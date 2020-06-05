import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import MovieLensDataset, SyntheticMovieLensDataset
from loss import loss_2s, loss_ce, loss_ips
from metric import Evaluator
from model import ImpressionSimulator, Nominator, Ranker
from six.moves import cPickle as pickle
from torch.utils.data import DataLoader, Subset
from torchnet.meter import AUCMeter

parser = argparse.ArgumentParser()
parser.add_argument("--verbose", type=int, default=1, help="Verbose.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--loss_type", default="loss_ce")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--alpha", type=float, default=1e-3, help="Loss ratio.")
parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
args = parser.parse_args()

torch.manual_seed(0)
torch.cuda.manual_seed(0)

filepath = "./"
device = args.device

dataset = MovieLensDataset(filepath, device=device)

NUM_ITEMS = dataset.NUM_ITEMS
NUM_YEARS = dataset.NUM_YEARS
NUM_GENRES = dataset.NUM_GENRES

NUM_USERS = dataset.NUM_USERS
NUM_OCCUPS = dataset.NUM_OCCUPS
NUM_AGES = dataset.NUM_AGES
NUM_ZIPS = dataset.NUM_ZIPS

simulator_path = os.path.join(filepath, "simulator.pt")
if os.path.exists(simulator_path):
    simulator = ImpressionSimulator(use_impression_feats=True)
    simulator.load_state_dict(torch.load(simulator_path))
    simulator.to(device)
    simulator.eval()
else:
    # train a simulator model on the original ML-1M dataset
    # the simulator will be used to generate synthetic labels later

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    num_samples = len(dataset)
    train_loader = DataLoader(
        Subset(dataset, list(range(num_samples * 3 // 5))),
        batch_size=128,
        shuffle=True)
    val_loader = DataLoader(
        Subset(dataset,
               list(range(num_samples * 3 // 5, num_samples * 4 // 5))),
        batch_size=128)
    test_loader = DataLoader(
        Subset(dataset, list(range(num_samples * 4 // 5, num_samples))),
        batch_size=128)

    simulator = ImpressionSimulator(use_impression_feats=True).to(device)
    opt = torch.optim.Adagrad(
        simulator.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    simulator.train()
    for epoch in range(4):
        print("---epoch {}---".format(epoch))
        for step, batch in enumerate(train_loader):
            feats, labels = batch
            logits = simulator(**feats)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    simulator.eval()
                    auc = AUCMeter()
                    for feats, labels in val_loader:
                        outputs = torch.sigmoid(simulator(**feats))
                        auc.add(outputs, labels)
                    print(step, auc.value()[0])
                    if auc.value()[0] > 0.735:
                        break
                simulator.train()

    simulator.to("cpu")
    torch.save(simulator.state_dict(), simulator_path)

# create a torch dataset class that adopt the simulator and generate the synthetic dataset
synthetic_data_path = os.path.join(filepath, "full_impression_feats.pt")
syn = SyntheticMovieLensDataset(
    filepath, simulator_path, synthetic_data_path, device=device)

logging_policy_path = os.path.join(filepath, "logging_policy.pt")
if os.path.exists(logging_policy_path):
    logging_policy = Nominator()
    logging_policy.load_state_dict(torch.load(logging_policy_path))
    logging_policy.to(device)
    logging_policy.eval()
else:
    # train a logging policy using the synthetic dataset
    num_samples = len(syn)
    idx_list = list(range(num_samples))
    rs = np.random.RandomState(0)
    rs.shuffle(idx_list)
    train_idx = idx_list[:10000]
    val_idx = idx_list[10000:20000]
    test_idx = idx_list[-100000:]
    train_loader = DataLoader(
        Subset(syn, train_idx), batch_size=128, shuffle=True)
    val_loader = DataLoader(Subset(syn, val_idx), batch_size=128)
    test_loader = DataLoader(Subset(syn, test_idx), batch_size=128)

    logging_policy = Nominator().to(device)
    opt = torch.optim.Adagrad(
        logging_policy.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    logging_policy.train()
    for epoch in range(40):
        print("---epoch {}---".format(epoch))
        for step, batch in enumerate(train_loader):
            feats, labels = batch
            feats = syn.to_device(feats)
            labels = syn.to_device(labels)
            logits = logging_policy(feats["user_feats"], feats["item_feats"])
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            logging_policy.eval()
            auc = AUCMeter()
            for feats, labels in val_loader:
                feats = syn.to_device(feats)
                labels = syn.to_device(labels)
                outputs = torch.sigmoid(
                    logging_policy(feats["user_feats"], feats["item_feats"]))
                auc.add(outputs, labels)
            print(step, auc.value()[0])

            logging_policy.train()

    logging_policy.eval()

    logging_policy.to("cpu")
    torch.save(logging_policy.state_dict(), logging_policy_path)
    logging_policy.to(device)


def generate_bandit_samples(logging_policy, syn, k=5):
    """Generates partial-labeled bandit samples with the logging policy.

    Arguments:
        k: The number of items to be sampled for each user.
    """
    logging_policy.set_binary(False)
    with torch.no_grad():
        feats = {}
        feats["user_feats"] = syn.user_feats
        feats["item_feats"] = syn.item_feats
        feats = syn.to_device(feats)
        probs = F.softmax(logging_policy(**feats), dim=1)

    sampled_users = []
    sampled_actions = []
    sampled_probs = []
    sampled_rewards = []
    for i in range(probs.size(0)):
        sampled_users.append([i] * k)
        sampled_actions.append(
            torch.multinomial(probs[i], k).cpu().numpy().tolist())
        sampled_probs.append(
            probs[i, sampled_actions[-1]].cpu().numpy().tolist())
        sampled_rewards.append(syn.impression_feats["labels"][[
            i * probs.size(1) + j for j in sampled_actions[-1]
        ]].numpy().tolist())
    return np.array(sampled_users).reshape(-1), np.array(
        sampled_actions).reshape(-1), np.array(sampled_probs).reshape(
            -1), np.array(sampled_rewards).reshape(-1)


torch.manual_seed(0)
torch.cuda.manual_seed(0)

u, a, p, r = generate_bandit_samples(
    logging_policy, syn,
    k=5)  # u: user, a: item, p: logging policy probability, r: reward/label

ev = Evaluator(u[r > 0], a[r > 0], simulator, syn)

all_user_feats = syn.to_device(syn.user_feats)
all_item_feats = syn.to_device(syn.item_feats)
all_impression_feats = syn.to_device({
    "real_feats":
    torch.mean(
        syn.impression_feats["real_feats"].view(NUM_USERS, NUM_ITEMS),
        dim=1).view(-1, 1)
})

# Split validation/test users.
num_val_users = 2000
val_user_list = list(range(0, num_val_users))
test_user_list = list(range(num_val_users, NUM_USERS))

test_item_feats = all_item_feats
test_user_feats = syn.to_device(
    {key: value[test_user_list]
     for key, value in all_user_feats.items()})
test_impression_feats = syn.to_device({
    key: value[test_user_list]
    for key, value in all_impression_feats.items()
})

val_item_feats = all_item_feats
val_user_feats = syn.to_device(
    {key: value[val_user_list]
     for key, value in all_user_feats.items()})
val_impression_feats = syn.to_device(
    {key: value[val_user_list]
     for key, value in all_impression_feats.items()})

ranker_path = os.path.join(filepath, "ranker.pt")
if os.path.exists(ranker_path):
    ranker = Ranker()
    ranker.load_state_dict(torch.load(ranker_path))
    ranker.to(device)
    ranker.eval()
    ranker.set_binary(False)
else:
    # train the ranker with binary cross-entropy
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    batch_size = 128
    neg_sample_size = 29

    ranker = Ranker().to(device)
    opt = torch.optim.Adagrad(ranker.parameters(), lr=0.05, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    rs = np.random.RandomState(0)
    ranker.train()
    for epoch in range(10):
        print("---epoch {}---".format(epoch))
        for step in range(len(u) // batch_size):
            user_list = u[step * batch_size:(step + 1) * batch_size]
            item_list = a[step * batch_size:(step + 1) * batch_size]
            user_feats = syn.to_device({
                key: value[user_list]
                for key, value in syn.user_feats.items()
            })
            item_feats = syn.to_device({
                key: value[item_list]
                for key, value in syn.item_feats.items()
            })
            impression_list = [
                user_id * NUM_ITEMS + item_id
                for user_id, item_id in zip(user_list, item_list)
            ]
            impression_feats = syn.to_device({
                "real_feats":
                syn.impression_feats["real_feats"][impression_list]
            })

            labels = torch.FloatTensor(
                r[step * batch_size:(step + 1) * batch_size]).to(device)

            logits = ranker(user_feats, item_feats, impression_feats)
            loss = criterion(logits, labels)

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            ranker.eval()
            ranker.set_binary(False)
            logits = ranker(all_user_feats, all_item_feats,
                            all_impression_feats)

            print(step, ev.one_stage_eval(logits))

            # Evaluate ranking metrics on validation users.
            logits = ranker(val_user_feats, val_item_feats,
                            val_impression_feats)
            print(step, ev.one_stage_ranking_eval(logits, val_user_list))

            ranker.train()
            ranker.set_binary(True)

    ranker.eval()
    ranker.set_binary(False)

    ranker.to("cpu")
    torch.save(ranker.state_dict(), ranker_path)
    ranker.to(device)

u = u[r > 0]
a = a[r > 0]
p = p[r > 0]

batch_size = 128
check_metric = "Precision@10"

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

nominator = Nominator().to(device)
nominator.set_binary(False)

opt = torch.optim.Adagrad(
    nominator.parameters(), lr=args.lr, weight_decay=1e-4)

rs = np.random.RandomState(0)
nominator.train()

# Init results.
best_epoch = 0
best_result = 0.0
val_results, test_results = [], []

if args.loss_type == "loss_2s":
    with torch.no_grad():
        ranker.eval()
        ranker.set_binary(False)
        ranker_logits = ranker(all_user_feats, all_item_feats,
                               all_impression_feats)

for epoch in range(20):
    print("---epoch {}---".format(epoch))
    for step in range(len(u) // batch_size):
        item_ids = torch.LongTensor(
            a[step * batch_size:(step + 1) * batch_size]).to(device)
        item_probs = torch.FloatTensor(
            p[step * batch_size:(step + 1) * batch_size]).to(device)

        user_ids = u[step * batch_size:(step + 1) * batch_size]
        user_feats = {
            key: value[user_ids]
            for key, value in syn.user_feats.items()
        }
        user_feats = syn.to_device(user_feats)

        logits = nominator(user_feats, val_item_feats)

        if args.loss_type == "loss_ce":
            loss = loss_ce(logits, item_ids, item_probs)
        elif args.loss_type == "loss_ips":
            loss = loss_ips(logits, item_ids, item_probs, upper_limit=10)
        elif args.loss_type == "loss_2s":
            batch_ranker_logits = F.embedding(
                torch.LongTensor(user_ids).to(device), ranker_logits)
            loss = loss_2s(
                logits,
                item_ids,
                item_probs,
                batch_ranker_logits,
                upper_limit=10,
                alpha=args.alpha)
        else:
            raise NotImplementedError(
                "{} not supported.".format(args.loss_type))

        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        nominator.eval()

        logits = nominator(all_user_feats, all_item_feats)
        print("1 stage", ev.one_stage_eval(logits))
        print("2 stage", ev.two_stage_eval(logits, ranker))

        # Evaluate ranking metrics on validation users.
        logits = nominator(val_user_feats, val_item_feats)
        one_stage_results = ev.one_stage_ranking_eval(logits, val_user_list)
        print("1 stage (val)", one_stage_results)
        two_stage_results = ev.two_stage_ranking_eval(logits, ranker,
                                                      val_user_list)
        print("2 stage (val)", two_stage_results)
        val_results.append((one_stage_results, two_stage_results))
        # Log best epoch
        if two_stage_results[check_metric] > best_result:
            best_epoch = epoch
            best_result = two_stage_results[check_metric]
        # Evaluate ranking metrics on test users.
        logits = nominator(test_user_feats, test_item_feats)
        one_stage_results = ev.one_stage_ranking_eval(logits, test_user_list)
        print("1 stage (test)", one_stage_results)
        two_stage_results = ev.two_stage_ranking_eval(logits, ranker,
                                                      test_user_list)
        print("2 stage (test)", two_stage_results)
        test_results.append((one_stage_results, two_stage_results))

        nominator.train()

print("Best validation epoch: {}".format(best_epoch))
print("Best validation stage results\n 1 stage: {}\n 2 stage: {}".format(
    val_results[best_epoch][0], val_results[best_epoch][1]))
print("Best test results\n 1 stage: {}\n 2 stage: {}".format(
    test_results[best_epoch][0], test_results[best_epoch][1]))

pickle.dump((best_epoch, val_results, test_results),
            open("results/{}-a{}_{}.pkl".format(
                args.loss_type.split("_")[1], args.alpha, args.seed), "wb"))

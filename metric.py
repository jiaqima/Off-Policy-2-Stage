from collections import OrderedDict

import numpy as np
import torch
from model import NUM_ITEMS, NUM_USERS


class BaseMetric(object):
    def __init__(self, rel_threshold, k):
        self.rel_threshold = rel_threshold
        if np.isscalar(k):
            k = np.array([k])
        self.k = k

    def __len__(self):
        return len(self.k)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def _compute(self, *args, **kwargs):
        raise NotImplementedError


class PrecisionRecall(BaseMetric):
    def __init__(self, rel_threshold=0, k=10):
        super(PrecisionRecall, self).__init__(rel_threshold, k)

    def __len__(self):
        return 2 * len(self.k)

    def __str__(self):
        str_precision = [('Precision@%1.f' % x) for x in self.k]
        str_recall = [('Recall@%1.f' % x) for x in self.k]
        return (','.join(str_precision)) + ',' + (','.join(str_recall))

    def __call__(self, targets, predictions):
        precision, recall = zip(
            *[self._compute(targets, predictions, x) for x in self.k])
        result = np.concatenate((precision, recall), axis=0)
        return result

    def _compute(self, targets, predictions, k):
        predictions = predictions[:k]
        num_hit = len(set(predictions).intersection(set(targets)))

        return float(num_hit) / len(predictions), float(num_hit) / len(targets)


class MeanAP(BaseMetric):
    def __init__(self, rel_threshold=0, k=np.inf):
        super(MeanAP, self).__init__(rel_threshold, k)

    def __call__(self, targets, predictions):
        result = [self._compute(targets, predictions, x) for x in self.k]
        return np.array(result)

    def __str__(self):
        return ','.join([('MeanAP@%1.f' % x) for x in self.k])

    def _compute(self, targets, predictions, k):
        if len(predictions) > k:
            predictions = predictions[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predictions):
            if p in targets and p not in predictions[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not list(targets):
            return 0.0

        return score / min(len(targets), k)


class NormalizedDCG(BaseMetric):
    def __init__(self, rel_threshold=0, k=10):
        super(NormalizedDCG, self).__init__(rel_threshold, k)

    def __call__(self, targets, predictions):
        result = [self._compute(targets, predictions, x) for x in self.k]
        return np.array(result)

    def __str__(self):
        return ','.join([('NDCG@%1.f' % x) for x in self.k])

    def _compute(self, targets, predictions, k):
        k = min(len(targets), k)

        if len(predictions) > k:
            predictions = predictions[:k]

        # compute idcg
        idcg = np.sum(1 / np.log2(np.arange(2, k + 2)))
        dcg = 0.0
        for i, p in enumerate(predictions):
            if p in targets:
                dcg += 1 / np.log2(i + 2)
        ndcg = dcg / idcg

        return ndcg


all_metrics = [PrecisionRecall(k=[1, 5, 10]), NormalizedDCG(k=[5, 10, 20])]


class Evaluator(object):
    """Evaluator for both one-stage and two-stage evaluations."""

    def __init__(self, u, a, simulator, syn):
        self.u = u
        self.a = a
        self.simulator = simulator
        self.syn = syn

        self.target_rankings = self.get_target_rankings()
        self.metrics = all_metrics

    def get_target_rankings(self):
        target_rankings = []
        with torch.no_grad():
            self.simulator.eval()
            for i in range(NUM_USERS):
                impression_ids = range(i * NUM_ITEMS, (i + 1) * NUM_ITEMS)
                feats, _ = self.syn[impression_ids]
                feats["impression_feats"]["real_feats"] = torch.mean(
                    feats["impression_feats"]["real_feats"],
                    dim=0,
                    keepdim=True).repeat([NUM_ITEMS, 1])
                feats = self.syn.to_device(feats)
                outputs = torch.sigmoid(self.simulator(**feats))
                user_target_ranking = (outputs >
                                       self.syn.cut).nonzero().view(-1)
                target_rankings.append(user_target_ranking.cpu().numpy())
        return target_rankings

    def one_stage_ranking_eval(self, logits, user_list):
        for i, user in enumerate(user_list):
            user_rated_items = self.a[self.u == user]
            logits[i, user_rated_items] = -np.inf
        sort_idx = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        # Init evaluation results.
        total_metrics_len = 0
        for metric in self.metrics:
            total_metrics_len += len(metric)

        total_val_metrics = np.zeros(
            [len(user_list), total_metrics_len], dtype=np.float32)
        valid_rows = []
        for i, user in enumerate(user_list):
            pred_ranking = sort_idx[i].tolist()
            target_ranking = self.target_rankings[user]
            if len(target_ranking) <= 0:
                continue
            metric_results = list()
            for j, metric in enumerate(self.metrics):
                result = metric(
                    targets=target_ranking, predictions=pred_ranking)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)
        # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()
        # Summary evaluation results into a dict.
        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind:ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = values
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = value
            ind += len(metric)
        return result

    def two_stage_ranking_eval(self, logits, ranker, user_list, k=30):
        sort_idx = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        topk_item_ids = []
        for i, user in enumerate(user_list):
            topk_item_ids.append([])
            for j in sort_idx[i]:
                if j not in self.a[self.u == user]:
                    topk_item_ids[-1].append(j)
                if len(topk_item_ids[-1]) == k:
                    break
        time_feats = self.syn.to_device(
            torch.mean(
                self.syn.impression_feats["real_feats"].view(
                    NUM_USERS, NUM_ITEMS),
                dim=1).view(-1, 1))
        # Init evaluation results.
        total_metrics_len = 0
        for metric in self.metrics:
            total_metrics_len += len(metric)

        total_val_metrics = np.zeros(
            [len(user_list), total_metrics_len], dtype=np.float32)
        valid_rows = []
        for i, user in enumerate(user_list):
            user_feats = {
                key: value[user].view(1, -1)
                for key, value in self.syn.user_feats.items()
            }
            item_feats = {
                key: value[topk_item_ids[i]]
                for key, value in self.syn.item_feats.items()
            }

            user_feats = self.syn.to_device(user_feats)
            item_feats = self.syn.to_device(item_feats)
            impression_feats = {"real_feats": time_feats[user].view(1, -1)}
            ranker_logits = ranker(user_feats, item_feats,
                                   impression_feats).view(1, -1)
            _, pred = ranker_logits.topk(k=k)
            pred = pred[0].cpu().numpy()
            pred_ranking = sort_idx[i][pred].tolist()
            target_ranking = self.target_rankings[user]
            if len(target_ranking) <= 0:
                continue
            metric_results = list()
            for j, metric in enumerate(self.metrics):
                result = metric(
                    targets=target_ranking, predictions=pred_ranking)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)
            # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()
        # Summary evaluation results into a dict.
        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind:ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = values
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = value
            ind += len(metric)
        return result

    def one_stage_eval(self, logits):
        sort_idx = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        impression_ids = []
        for i in range(NUM_USERS):
            for j in sort_idx[i]:
                if j not in self.a[self.u == i]:
                    break
            impression_ids.append(i * NUM_ITEMS + j)
        feats, labels = self.syn[impression_ids]
        feats["impression_feats"]["real_feats"] = torch.mean(
            self.syn.impression_feats["real_feats"].view(NUM_USERS, NUM_ITEMS),
            dim=1).view(-1, 1)
        with torch.no_grad():
            self.simulator.eval()
            feats = self.syn.to_device(feats)
            outputs = torch.sigmoid(self.simulator(**feats))
            return torch.mean(
                (outputs > self.syn.cut).to(dtype=torch.float32)).item()

    def two_stage_eval(self, logits, ranker, k=30):
        sort_idx = torch.argsort(logits, dim=1, descending=True).cpu().numpy()
        topk_item_ids = []
        for i in range(NUM_USERS):
            topk_item_ids.append([])
            for j in sort_idx[i]:
                if j not in self.a[self.u == i]:
                    topk_item_ids[-1].append(j)
                if len(topk_item_ids[-1]) == k:
                    break
        time_feats = self.syn.to_device(
            torch.mean(
                self.syn.impression_feats["real_feats"].view(
                    NUM_USERS, NUM_ITEMS),
                dim=1).view(-1, 1))
        recommneded = []
        for i in range(NUM_USERS):
            user_feats = {
                key: value[i].view(1, -1)
                for key, value in self.syn.user_feats.items()
            }
            item_feats = {
                key: value[topk_item_ids[i]]
                for key, value in self.syn.item_feats.items()
            }

            user_feats = self.syn.to_device(user_feats)
            item_feats = self.syn.to_device(item_feats)
            impression_feats = {"real_feats": time_feats[i].view(1, -1)}
            ranker_logits = ranker(user_feats, item_feats,
                                   impression_feats).view(1, -1)
            _, pred = torch.max(ranker_logits, 1)
            pred = pred.squeeze().item()
            recommneded.append(topk_item_ids[i][pred])

        impression_ids = []
        for i in range(NUM_USERS):
            impression_ids.append(i * NUM_ITEMS + recommneded[i])
        feats, labels = self.syn[impression_ids]
        feats["impression_feats"]["real_feats"] = torch.mean(
            self.syn.impression_feats["real_feats"].view(NUM_USERS, NUM_ITEMS),
            dim=1).view(-1, 1)
        with torch.no_grad():
            self.simulator.eval()
            feats = self.syn.to_device(feats)
            outputs = torch.sigmoid(self.simulator(**feats))
            return torch.mean(
                (outputs > self.syn.cut).to(dtype=torch.float32)).item()

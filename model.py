import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_ITEMS = 3883
NUM_YEARS = 81
NUM_GENRES = 18
NUM_USERS = 6040
NUM_OCCUPS = 21
NUM_AGES = 7
NUM_ZIPS = 3439


class ItemRep(nn.Module):
    """Item representation layer."""

    def __init__(self, item_emb_size=10, year_emb_size=5, genre_hidden=5):
        super(ItemRep, self).__init__()
        self.item_embedding = nn.Embedding(
            NUM_ITEMS + 1, item_emb_size, padding_idx=0)
        self.year_embedding = nn.Embedding(NUM_YEARS, year_emb_size)
        self.genre_linear = nn.Linear(NUM_GENRES, genre_hidden)
        self.rep_dim = item_emb_size + year_emb_size + genre_hidden

    def forward(self, categorical_feats, real_feats):
        out = torch.cat(
            [
                self.item_embedding(categorical_feats[:, 0]),
                self.year_embedding(categorical_feats[:, 1]),
                self.genre_linear(real_feats)
            ],
            dim=1)
        return out


class UserRep(nn.Module):
    """User representation layer."""

    def __init__(self, user_emb_size=10, feature_emb_size=5):
        super(UserRep, self).__init__()
        self.user_embedding = nn.Embedding(
            NUM_USERS + 1, user_emb_size, padding_idx=0)
        self.gender_embedding = nn.Embedding(2, feature_emb_size)
        self.age_embedding = nn.Embedding(NUM_AGES, feature_emb_size)
        self.occup_embedding = nn.Embedding(NUM_OCCUPS, feature_emb_size)
        self.zip_embedding = nn.Embedding(NUM_ZIPS, feature_emb_size)
        self.rep_dim = user_emb_size + feature_emb_size * 4

    def forward(self, categorical_feats, real_feats=None):
        reps = [
            self.user_embedding(categorical_feats[:, 0]),
            self.gender_embedding(categorical_feats[:, 1]),
            self.age_embedding(categorical_feats[:, 2]),
            self.occup_embedding(categorical_feats[:, 3]),
            self.zip_embedding(categorical_feats[:, 4])
        ]
        out = torch.cat(reps, dim=1)
        return out


class ImpressionSimulator(nn.Module):
    """Simulator model that predicts the outcome of impression."""

    def __init__(self, hidden=100, use_impression_feats=False):
        super(ImpressionSimulator, self).__init__()
        self.user_rep = UserRep()
        self.item_rep = ItemRep()
        self.use_impression_feats = use_impression_feats
        input_dim = self.user_rep.rep_dim + self.item_rep.rep_dim
        if use_impression_feats:
            input_dim += 1
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(), nn.Linear(hidden, 50), nn.ReLU(), nn.Linear(50, 1))

    def forward(self, user_feats, item_feats, impression_feats=None):
        users = self.user_rep(**user_feats)
        items = self.item_rep(**item_feats)
        inputs = torch.cat([users, items], dim=1)
        if self.use_impression_feats:
            inputs = torch.cat([inputs, impression_feats["real_feats"]], dim=1)
        return self.linear(inputs).squeeze()


class Nominator(nn.Module):
    """Two tower nominator model."""

    def __init__(self):
        super(Nominator, self).__init__()
        self.item_rep = ItemRep()
        self.user_rep = UserRep()
        self.linear = nn.Linear(self.user_rep.rep_dim, self.item_rep.rep_dim)
        self.binary = True

    def forward(self, user_feats, item_feats):
        users = self.linear(F.relu(self.user_rep(**user_feats)))
        users = torch.unsqueeze(users, 2)  # (b, h) -> (b, h, 1)
        items = self.item_rep(**item_feats)
        if self.binary:
            items = torch.unsqueeze(items, 1)  # (b, h) -> (b, 1, h)
        else:
            items = torch.unsqueeze(items, 0).expand(users.size(0), -1,
                                                     -1)  # (c, h) -> (b, c, h)
        logits = torch.bmm(items, users).squeeze()
        return logits

    def set_binary(self, binary=True):
        self.binary = binary


class Ranker(nn.Module):
    """Ranker model."""

    def __init__(self):
        super(Ranker, self).__init__()
        self.item_rep = ItemRep()
        self.user_rep = UserRep()
        self.linear = nn.Linear(self.user_rep.rep_dim + 1,
                                self.item_rep.rep_dim)
        self.binary = True

    def forward(self, user_feats, item_feats, impression_feats):
        users = self.user_rep(**user_feats)
        context_users = torch.cat(
            [users, impression_feats["real_feats"]], dim=1)
        context_users = self.linear(context_users)
        context_users = torch.unsqueeze(context_users,
                                        2)  # (b, h) -> (b, h, 1)
        items = self.item_rep(**item_feats)
        if self.binary:
            items = torch.unsqueeze(items, 1)  # (b, h) -> (b, 1, h)
        else:
            items = torch.unsqueeze(items, 0).expand(
                users.size(0), -1, -1)  # (c, h) -> (b, c, h), c=#items
        logits = torch.bmm(items, context_users).squeeze()
        return logits

    def set_binary(self, binary=True):
        self.binary = binary

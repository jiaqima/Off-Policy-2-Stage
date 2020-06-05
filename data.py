import os

import numpy as np
import pandas as pd
import torch
from model import ImpressionSimulator
from torch.utils.data import Dataset
from tqdm import tqdm


# Create a torch dataset class for ML-1M; convert the label into binary labels
# (positive if rating > 3)
class MovieLensDataset(Dataset):
    def __init__(self, filepath, device="cuda:0"):
        self.device = device
        ratings, users, items = self.load_data(filepath)

        self.user_feats = {}
        self.item_feats = {}
        self.impression_feats = {}

        self.user_feats["categorical_feats"] = torch.LongTensor(
            users.values).to(device)
        self.item_feats["categorical_feats"] = torch.LongTensor(
            items.values[:, :2]).to(device)
        self.item_feats["real_feats"] = torch.FloatTensor(
            items.values[:, 2:]).to(device)
        self.impression_feats["user_ids"] = torch.LongTensor(
            ratings.values[:, 0]).to(device)
        self.impression_feats["item_ids"] = torch.LongTensor(
            ratings.values[:, 1]).to(device)
        self.impression_feats["real_feats"] = torch.FloatTensor(
            ratings.values[:, 3]).view(-1, 1).to(device)
        self.impression_feats["labels"] = torch.FloatTensor(
            ratings.values[:, 2]).to(device)

    def __len__(self):
        return len(self.impression_feats["user_ids"])

    def __getitem__(self, idx):
        labels = self.impression_feats["labels"][idx]
        feats = {}
        feats["impression_feats"] = {}
        feats["impression_feats"]["real_feats"] = self.impression_feats[
            "real_feats"][idx]
        user_id = self.impression_feats["user_ids"][idx]
        item_id = self.impression_feats["item_ids"][idx]
        feats["user_feats"] = {
            key: value[user_id - 1]
            for key, value in self.user_feats.items()
        }
        feats["item_feats"] = {
            key: value[item_id - 1]
            for key, value in self.item_feats.items()
        }
        return feats, labels

    def load_data(self, filepath):
        names = "UserID::MovieID::Rating::Timestamp".split("::")
        ratings = pd.read_csv(
            os.path.join(filepath, "ratings.dat"),
            sep="::",
            names=names,
            engine="python")
        ratings["Rating"] = (ratings["Rating"] > 3).astype(int)
        ratings["Timestamp"] = (
            ratings["Timestamp"] - ratings["Timestamp"].min()
        ) / float(ratings["Timestamp"].max() - ratings["Timestamp"].min())

        names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
        users = pd.read_csv(
            os.path.join(filepath, "users.dat"),
            sep="::",
            names=names,
            engine="python")
        for i in range(1, users.shape[1]):
            users.iloc[:, i] = pd.factorize(users.iloc[:, i])[0]

        names = "MovieID::Title::Genres".split("::")
        Genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western"
        ]
        movies = pd.read_csv(
            os.path.join(filepath, "movies.dat"),
            sep="::",
            names=names,
            engine="python")
        movies["Year"] = movies["Title"].apply(lambda x: x[-5:-1])
        for genre in Genres:
            movies[genre] = movies["Genres"].apply(lambda x: genre in x)
        movies.iloc[:, 3] = pd.factorize(movies.iloc[:, 3])[0]
        movies.iloc[:, 4:] = movies.iloc[:, 4:].astype(float)
        movies = movies.loc[:, ["MovieID", "Year"] + Genres]
        movies.iloc[:, 2:] = movies.iloc[:, 2:].div(
            movies.iloc[:, 2:].sum(axis=1), axis=0)

        movie_id_map = {}
        for i in range(movies.shape[0]):
            movie_id_map[movies.loc[i, "MovieID"]] = i + 1

        movies["MovieID"] = movies["MovieID"].apply(lambda x: movie_id_map[x])
        ratings["MovieID"] = ratings["MovieID"].apply(
            lambda x: movie_id_map[x])

        self.NUM_ITEMS = len(movies.MovieID.unique())
        self.NUM_YEARS = len(movies.Year.unique())
        self.NUM_GENRES = movies.shape[1] - 2

        self.NUM_USERS = len(users.UserID.unique())
        self.NUM_OCCUPS = len(users.Occupation.unique())
        self.NUM_AGES = len(users.Age.unique())
        self.NUM_ZIPS = len(users["Zip-code"].unique())

        return ratings, users, movies


class SyntheticMovieLensDataset(Dataset):
    def __init__(self, filepath, simulator_path, synthetic_data_path, cut=0.764506,
                 device="cuda:0"):
        self.device = device
        self.cut = cut
        self.simulator = None

        ratings, users, items = self.load_data(filepath)

        self.user_feats = {}
        self.item_feats = {}
        self.impression_feats = {}

        self.user_feats["categorical_feats"] = torch.LongTensor(users.values)
        self.item_feats["categorical_feats"] = torch.LongTensor(
            items.values[:, :2])
        self.item_feats["real_feats"] = torch.FloatTensor(items.values[:, 2:])

        if os.path.exists(synthetic_data_path):
            self.impression_feats = torch.load(synthetic_data_path)
            self.impression_feats["labels"] = (
                self.impression_feats["label_probs"] >= cut).to(
                    dtype=torch.float32)
            print("loaded full_impression_feats.pt")
        else:
            print("generating impression_feats")
            self.simulator = ImpressionSimulator(use_impression_feats=True)
            self.simulator.load_state_dict(torch.load(simulator_path))
            self.simulator = self.simulator.to(device)

            impressions = self.get_full_impressions(ratings)

            self.impression_feats["user_ids"] = torch.LongTensor(
                impressions[:, 0])
            self.impression_feats["item_ids"] = torch.LongTensor(
                impressions[:, 1])
            self.impression_feats["real_feats"] = torch.FloatTensor(
                impressions[:, 2]).view(-1, 1)
            self.impression_feats["labels"] = torch.zeros_like(
                self.impression_feats["real_feats"])

            self.impression_feats["label_probs"] = self.generate_labels()
            self.impression_feats["labels"] = (
                self.impression_feats["label_probs"] >= cut).to(
                    dtype=torch.float32)

            torch.save(self.impression_feats, synthetic_data_path)
            print("saved impression_feats")

    def __len__(self):
        return len(self.impression_feats["user_ids"])

    def __getitem__(self, idx):
        labels = self.impression_feats["labels"][idx]
        feats = {}
        feats["impression_feats"] = {}
        feats["impression_feats"]["real_feats"] = self.impression_feats[
            "real_feats"][idx]
        user_id = self.impression_feats["user_ids"][idx]
        item_id = self.impression_feats["item_ids"][idx]
        feats["user_feats"] = {
            key: value[user_id - 1]
            for key, value in self.user_feats.items()
        }
        feats["item_feats"] = {
            key: value[item_id - 1]
            for key, value in self.item_feats.items()
        }
        return feats, labels

    def load_data(self, filepath):
        names = "UserID::MovieID::Rating::Timestamp".split("::")
        ratings = pd.read_csv(
            os.path.join(filepath, "ratings.dat"),
            sep="::",
            names=names,
            engine="python")
        ratings["Rating"] = (ratings["Rating"] > 3).astype(int)
        ratings["Timestamp"] = (
            ratings["Timestamp"] - ratings["Timestamp"].min()
        ) / float(ratings["Timestamp"].max() - ratings["Timestamp"].min())

        names = "UserID::Gender::Age::Occupation::Zip-code".split("::")
        users = pd.read_csv(
            os.path.join(filepath, "users.dat"),
            sep="::",
            names=names,
            engine="python")
        for i in range(1, users.shape[1]):
            users.iloc[:, i] = pd.factorize(users.iloc[:, i])[0]

        names = "MovieID::Title::Genres".split("::")
        Genres = [
            "Action", "Adventure", "Animation", "Children's", "Comedy",
            "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
            "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War",
            "Western"
        ]
        movies = pd.read_csv(
            os.path.join(filepath, "movies.dat"),
            sep="::",
            names=names,
            engine="python")
        movies["Year"] = movies["Title"].apply(lambda x: x[-5:-1])
        for genre in Genres:
            movies[genre] = movies["Genres"].apply(lambda x: genre in x)
        movies.iloc[:, 3] = pd.factorize(movies.iloc[:, 3])[0]
        movies.iloc[:, 4:] = movies.iloc[:, 4:].astype(float)
        movies = movies.loc[:, ["MovieID", "Year"] + Genres]
        movies.iloc[:, 2:] = movies.iloc[:, 2:].div(
            movies.iloc[:, 2:].sum(axis=1), axis=0)

        movie_id_map = {}
        for i in range(movies.shape[0]):
            movie_id_map[movies.loc[i, "MovieID"]] = i + 1

        movies["MovieID"] = movies["MovieID"].apply(lambda x: movie_id_map[x])
        ratings["MovieID"] = ratings["MovieID"].apply(
            lambda x: movie_id_map[x])

        self.NUM_ITEMS = len(movies.MovieID.unique())
        self.NUM_YEARS = len(movies.Year.unique())
        self.NUM_GENRES = movies.shape[1] - 2

        self.NUM_USERS = len(users.UserID.unique())
        self.NUM_OCCUPS = len(users.Occupation.unique())
        self.NUM_AGES = len(users.Age.unique())
        self.NUM_ZIPS = len(users["Zip-code"].unique())

        return ratings, users, movies

    def get_full_impressions(self, ratings):
        """Gets NUM_USERS x NUM_ITEMS impression features by iterating the user and item ids.
        The impression-level feature, i.e. the timestamp, is sampled from a normal distribution with
        mean and std as the empirical mean and std of each user's recorded timestamps in the real data.
        """
        timestamps = {}
        for i in range(len(ratings)):
            u_id = ratings.loc[i, "UserID"]
            timestamps[u_id] = timestamps.get(u_id, [])
            timestamps[u_id].append(ratings.loc[i, "Timestamp"])
        rs = np.random.RandomState(0)
        t_samples = []
        for i in range(self.NUM_USERS):
            u_id = i + 1
            t_samples.append(
                rs.normal(
                    loc=np.mean(timestamps[u_id]),
                    scale=np.std(timestamps[u_id]),
                    size=(self.NUM_ITEMS, )))
        t_samples = np.array(t_samples)

        impressions = []
        for i in range(self.NUM_USERS):
            for j in range(self.NUM_ITEMS):
                impressions.append([i + 1, j + 1, t_samples[i, j]])
        impressions = np.array(impressions)
        return impressions

    def to_device(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, dict):
            transformed_data = {}
            for key in data:
                transformed_data[key] = self.to_device(data[key])
        elif type(data) == list:
            transformed_data = []
            for x in data:
                transformed_data.append(self.to_device(x))
        else:
            raise NotImplementedError(
                "Type {} not supported.".format(type(data)))
        return transformed_data

    def generate_labels(self):
        """Generates the binary labels using the simulator on every user-item pair."""
        with torch.no_grad():
            self.simulator.eval()
            preds = []
            for i in tqdm(range(len(self.impression_feats["labels"]) // 500)):
                feats, _ = self.__getitem__(
                    list(range(i * 500, (i + 1) * 500)))
                feats = self.to_device(feats)
                outputs = torch.sigmoid(self.simulator(**feats))
                preds += list(outputs.squeeze().cpu().numpy())
            if (i + 1) * 500 < len(self.impression_feats["labels"]):
                feats, _ = self.__getitem__(
                    list(
                        range((i + 1) * 500,
                              len(self.impression_feats["labels"]))))
                feats = self.to_device(feats)
                outputs = torch.sigmoid(self.simulator(**feats))
                preds += list(outputs.squeeze().cpu().numpy())
        return torch.FloatTensor(np.array(preds))

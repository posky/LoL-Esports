import sys
from functools import reduce
from collections import Counter
import datetime
import logging

import pandas as pd
import numpy as np

from google_sheet import Sheet
from lol_fandom import get_leagues

pd.set_option("display.max_columns", None)
with open("./sheet_id.txt", "r") as f:
    SHEET_ID = f.read()

START_YEAR = 2011
END_YEAR = datetime.datetime.now().year


class LoLStats:
    def __init__(self, games, players, players_id):
        assert games.shape[0] > 0 and players.shape[0] > 0

        self.games = games
        self.players = players
        self.players_id = players_id

        self.games["CKPM"] = self.games[["Team1Kills", "Team2Kills"]].sum(axis=1)
        self.games["Team1GPM"] = self.games["Team1Gold"]
        self.games["Team2GPM"] = self.games["Team2Gold"]
        self.games["Team1GDPM"] = self.games["Team1Gold"] - self.games["Team2Gold"]
        self.games["Team2GDPM"] = self.games["Team2Gold"] - self.games["Team1Gold"]
        self.games["Team1KPM"] = self.games["Team1Kills"]
        self.games["Team2KPM"] = self.games["Team2Kills"]
        columns = [
            "CKPM",
            "Team1GPM",
            "Team2GPM",
            "Team1GDPM",
            "Team2GDPM",
            "Team1KPM",
            "Team2KPM",
        ]
        self.games[columns] = self.games[columns].divide(
            self.games["Gamelength Number"], axis=0
        )

        self.merged = pd.merge(players, games, how="inner", on="GameId")

        self.merged["player_id"] = self.merged["Link"].transform(
            lambda x: self.players_id.loc[players_id["player"] == x, "player_id"].iloc[
                0
            ]
        )
        self.merged["CSPM"] = self.merged["CS"]
        self.merged["GPM"] = self.merged["Gold"]
        self.merged["DPM"] = self.merged["DamageToChampions"]
        columns = ["CSPM", "GPM", "DPM"]
        self.merged[columns] = self.merged[columns].divide(
            self.merged["Gamelength Number"], axis=0
        )

    def get_teams_stats(self):
        teams_lst = self.games[["Team1", "Team2"]].unstack().unique()
        stats = pd.DataFrame(index=teams_lst)
        stats.index.set_names("Team", inplace=True)
        stats.sort_index(inplace=True)

        for team_name in stats.index:
            team1_games = self.games.loc[self.games["Team1"] == team_name]
            team2_games = self.games.loc[self.games["Team2"] == team_name]

            stats.loc[team_name, "Games"] = team1_games.shape[0] + team2_games.shape[0]
            stats.loc[team_name, "Win"] = (
                team1_games.loc[team1_games["Team1"] == team1_games["WinTeam"]].shape[0]
                + team2_games.loc[team2_games["Team2"] == team2_games["WinTeam"]].shape[
                    0
                ]
            )
            stats.loc[team_name, "Loss"] = (
                team1_games.loc[team1_games["Team1"] == team1_games["LossTeam"]].shape[
                    0
                ]
                + team2_games.loc[
                    team2_games["Team2"] == team2_games["LossTeam"]
                ].shape[0]
            )
            stats.loc[team_name, "WinRate"] = stats.loc[team_name, "Win"]
            stats.loc[team_name, "KD"] = (
                team1_games["Team1Kills"].sum() + team2_games["Team2Kills"].sum()
            ) / (team1_games["Team2Kills"].sum() + team2_games["Team1Kills"].sum())
            stats.loc[team_name, "CKPM"] = (
                team1_games["CKPM"].sum() + team2_games["CKPM"].sum()
            )
            stats.loc[team_name, "GameDuration"] = (
                team1_games["Gamelength Number"].sum()
                + team2_games["Gamelength Number"].sum()
            )
            stats.loc[team_name, "WinGameDuration"] = (
                team1_games.loc[
                    team1_games["Team1"] == team1_games["WinTeam"], "Gamelength Number"
                ].sum()
                + team2_games.loc[
                    team2_games["Team2"] == team2_games["WinTeam"], "Gamelength Number"
                ].sum()
            )
            stats.loc[team_name, "LossGameDuration"] = (
                team1_games.loc[
                    team1_games["Team1"] == team1_games["LossTeam"], "Gamelength Number"
                ].sum()
                + team2_games.loc[
                    team2_games["Team2"] == team2_games["LossTeam"], "Gamelength Number"
                ].sum()
            )
            stats.loc[team_name, "GPM"] = (
                team1_games["Team1GPM"].sum() + team2_games["Team2GPM"].sum()
            )
            stats.loc[team_name, "GDPM"] = (
                team1_games["Team1GDPM"].sum() + team2_games["Team2GDPM"].sum()
            )
            stats.loc[team_name, "KPM"] = (
                team1_games["Team1KPM"].sum() + team2_games["Team2KPM"].sum()
            )
        columns = ["WinRate", "CKPM", "GameDuration", "GPM", "GDPM", "KPM"]
        stats[columns] = stats[columns].divide(stats["Games"], axis=0)
        stats["WinGameDuration"] = stats["WinGameDuration"].divide(stats["Win"])
        stats["LossGameDuration"] = stats["LossGameDuration"].divide(stats["Loss"])

        return stats

    def get_champions_stats(self):
        ban_list = self.games[["Team1Bans", "Team2Bans"]].unstack().str.split(",")
        ban_list = list(reduce(lambda x, y: x + y, ban_list))
        champion_names = list(set(list(self.players["Champion"].unique()) + ban_list))
        while "None" in ban_list:
            ban_list.remove("None")
        champion_names.remove("None") if "None" in champion_names else None

        columns = [
            "Games",
            "BanPickRate",
            "Ban",
            "Blue Ban",
            "Red Ban",
            "GamesPlayed",
            "By",
            "Win",
            "Loss",
            "WinRate",
            "Kills",
            "Deaths",
            "Assists",
            "KDA",
            "CS",
            "CSPM",
            "Gold",
            "GPM",
            "Damage",
            "DPM",
        ]

        stats = pd.DataFrame(index=champion_names, columns=columns)
        stats.index.set_names("Champion", inplace=True)
        stats.sort_index(inplace=True)
        stats[columns] = 0

        for champ_name in stats.index:
            champions_df = self.merged.loc[self.merged["Champion"] == champ_name]

            stats.loc[champ_name, "GamesPlayed"] = champions_df.shape[0]
            stats.loc[champ_name, "By"] = len(champions_df["player_id"].unique())
            stats.loc[champ_name, "Win"] = champions_df.loc[
                champions_df["PlayerWin"] == "Yes"
            ].shape[0]
            stats.loc[champ_name, "Loss"] = champions_df.loc[
                champions_df["PlayerWin"] == "No"
            ].shape[0]
            stats.loc[champ_name, "WinRate"] = stats.loc[champ_name, "Win"]
            stats.loc[champ_name, "Kills"] = champions_df["Kills"].mean()
            stats.loc[champ_name, "Deaths"] = champions_df["Deaths"].mean()
            stats.loc[champ_name, "Assists"] = champions_df["Assists"].mean()
            stats.loc[champ_name, "KDA"] = (
                champions_df[["Kills", "Assists"]].unstack().sum()
                / champions_df["Deaths"].sum()
                if champions_df["Deaths"].sum() > 0
                else np.inf
            )
            stats.loc[champ_name, "CS"] = champions_df["CS"].mean()
            stats.loc[champ_name, "CSPM"] = champions_df["CSPM"].mean()
            stats.loc[champ_name, "Gold"] = champions_df["Gold"].mean()
            stats.loc[champ_name, "GPM"] = champions_df["GPM"].mean()
            stats.loc[champ_name, "Damage"] = champions_df["DamageToChampions"].mean()
            stats.loc[champ_name, "DPM"] = champions_df["DPM"].mean()

            stats.loc[champ_name, "Blue Ban"] = self.games.loc[
                self.games["Team1Bans"].str.contains(champ_name)
            ].shape[0]
            stats.loc[champ_name, "Red Ban"] = self.games.loc[
                self.games["Team2Bans"].str.contains(champ_name)
            ].shape[0]

        stats["Ban"] = 0
        ban_counter = Counter(ban_list)
        for key, value in ban_counter.items():
            stats.loc[key, "Ban"] = value
        stats["Games"] = stats[["GamesPlayed", "Ban"]].sum(axis=1)
        stats["BanPickRate"] = stats["Games"].divide(self.games.shape[0])
        stats["WinRate"] = stats["Win"].divide(stats["GamesPlayed"])

        assert len(columns) == len(stats.columns)

        return stats.sort_values(by="Games", ascending=False)

    def get_players_stats(self):
        idx = self.merged["player_id"].unique()
        stats = pd.DataFrame(index=idx)
        stats.index.set_names("id", inplace=True)
        stats.sort_index(inplace=True)

        for idx in stats.index:
            players_df = self.merged.loc[self.merged["player_id"] == idx]
            stats.loc[idx, "Player"] = players_df["Name"].iloc[-1]
            stats.loc[idx, "Team"] = players_df["Team"].iloc[-1]
            stats.loc[idx, "Games"] = players_df.shape[0]
            stats.loc[idx, "Win"] = players_df.loc[
                players_df["PlayerWin"] == "Yes"
            ].shape[0]
            stats.loc[idx, "Loss"] = players_df.loc[
                players_df["PlayerWin"] == "No"
            ].shape[0]
            stats.loc[idx, "WinRate"] = stats.loc[idx, "Win"]
            stats.loc[idx, "Kills"] = players_df["Kills"].mean()
            stats.loc[idx, "Deaths"] = players_df["Deaths"].mean()
            stats.loc[idx, "Assists"] = players_df["Assists"].mean()
            stats.loc[idx, "KDA"] = stats.loc[idx, ["Kills", "Assists"]].sum()
            stats.loc[idx, "DPM"] = players_df["DPM"].mean()
            stats.loc[idx, "CS"] = players_df["CS"].mean()
            stats.loc[idx, "CSPM"] = players_df["CSPM"].mean()
            stats.loc[idx, "Gold"] = players_df["Gold"].mean()
            stats.loc[idx, "GPM"] = players_df["GPM"].mean()
            stats.loc[idx, "KP"] = (
                players_df[["Kills", "Assists"]].unstack().sum()
                / players_df["TeamKills"].sum()
                if stats.loc[idx, ["Kills", "Assists"]].sum() > 0
                else 0
            )
            stats.loc[idx, "KS"] = (
                players_df["Kills"].sum() / players_df["TeamKills"].sum()
                if stats.loc[idx, "Kills"] > 0
                else 0
            )
            stats.loc[idx, "GS"] = (
                players_df["Gold"].sum() / players_df["TeamGold"].sum()
            )
            stats.loc[idx, "ChampionsPlayed"] = len(players_df["Champion"].unique())

        stats["WinRate"] = stats["WinRate"].divide(stats["Games"])
        stats["KDA"] = stats["KDA"].divide(stats["Deaths"])

        return stats.sort_values(by=["Team", "Player"])

    def get_player_by_champion_stats(self):
        idx = pd.MultiIndex.from_tuples(
            list(set(self.merged[["player_id", "Champion"]].itertuples(index=False))),
            names=["id", "Champion"],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for player_id, champion in stats.index:
            idx = (player_id, champion)
            players_df = self.merged.loc[
                (self.merged["player_id"] == player_id)
                & (self.merged["Champion"] == champion)
            ]

            stats.loc[idx, "Player"] = players_df["Name"].iloc[-1]
            stats.loc[idx, "Team"] = players_df["Team"].iloc[-1]
            stats.loc[idx, "Games"] = players_df.shape[0]
            stats.loc[idx, "Win"] = players_df.loc[
                players_df["PlayerWin"] == "Yes"
            ].shape[0]
            stats.loc[idx, "Loss"] = players_df.loc[
                players_df["PlayerWin"] == "No"
            ].shape[0]
            stats.loc[idx, "WinRate"] = (
                stats.loc[idx, "Win"] / stats.loc[idx, "Games"]
                if stats.loc[idx, "Win"] > 0
                else 0
            )
            stats.loc[idx, "Kills"] = players_df["Kills"].mean()
            stats.loc[idx, "Deaths"] = players_df["Deaths"].mean()
            stats.loc[idx, "Assists"] = players_df["Assists"].mean()
            stats.loc[idx, "KDA"] = (
                stats.loc[idx, ["Kills", "Assists"]].sum() / stats.loc[idx, "Deaths"]
                if stats.loc[idx, "Deaths"] > 0
                else np.inf
            )
            stats.loc[idx, "DPM"] = players_df["DPM"].mean()
            stats.loc[idx, "CSPM"] = players_df["CSPM"].mean()
            stats.loc[idx, "GPM"] = players_df["GPM"].mean()
            stats.loc[idx, "KP"] = (
                players_df[["Kills", "Assists"]].unstack().sum()
                / players_df["TeamKills"].sum()
                if stats.loc[idx, ["Kills", "Assists"]].sum() > 0
                else 0
            )
            stats.loc[idx, "KS"] = (
                players_df["Kills"].sum() / players_df["TeamKills"].sum()
                if stats.loc[idx, "Kills"] > 0
                else 0
            )
            stats.loc[idx, "GS"] = (
                players_df["Gold"].sum() / players_df["TeamGold"].sum()
            )

        stats.reset_index(level="id", drop=True, inplace=True)
        stats.reset_index(inplace=True)
        stats.sort_values(by=["Team", "Player", "Champion"], inplace=True)

        columns = [
            "Player",
            "Team",
            "Champion",
            "Games",
            "Win",
            "Loss",
            "WinRate",
            "Kills",
            "Deaths",
            "Assists",
            "KDA",
            "DPM",
            "CSPM",
            "GPM",
            "KP",
            "KS",
            "GS",
        ]
        assert len(stats.columns) == len(columns)

        return stats[columns]

    def get_duo_champions_stats(self, role1="Bot", role2="Support"):
        roles = self.players["Role"].unique()
        assert role1 != role2 and role1 in roles and role2 in roles

        role1_df = self.merged.loc[self.merged["IngameRole"] == role1]
        role2_df = self.merged.loc[self.merged["IngameRole"] == role2]
        merged = pd.merge(role1_df, role2_df, how="inner", on=["GameId", "Team"])
        assert merged.shape[0] == self.games.shape[0] * 2

        idx = pd.MultiIndex.from_tuples(
            list(set(merged[["Champion_x", "Champion_y"]].itertuples(index=False))),
            names=[role1, role2],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for champ1, champ2 in stats.index:
            idx = (champ1, champ2)
            duo_df = merged.loc[
                (merged["Champion_x"] == champ1) & (merged["Champion_y"] == champ2)
            ]
            stats.loc[idx, "Games"] = duo_df.shape[0]
            stats.loc[idx, "By"] = len(
                set(duo_df[["player_id_x", "player_id_y"]].itertuples(index=False))
            )
            stats.loc[idx, "Win"] = duo_df.loc[duo_df["PlayerWin_x"] == "Yes"].shape[0]
            stats.loc[idx, "Loss"] = duo_df.loc[duo_df["PlayerWin_x"] == "No"].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        return stats

    def get_vs_stats(self):
        merged = pd.merge(
            self.merged, self.merged, how="inner", on=["GameId", "IngameRole"]
        )
        merged = merged.loc[merged["Team_x"] != merged["Team_y"]]

        idx = pd.MultiIndex.from_tuples(
            list(
                set(
                    merged[["Champion_x", "Champion_y", "IngameRole"]].itertuples(
                        index=False
                    )
                )
            ),
            names=["Champion1", "Champion2", "As"],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for champ1, champ2, role in stats.index:
            idx = (champ1, champ2, role)
            vs_df = merged.loc[
                (merged["Champion_x"] == champ1)
                & (merged["Champion_y"] == champ2)
                & (merged["IngameRole"] == role)
            ]
            stats.loc[idx, "Games"] = vs_df.shape[0]
            stats.loc[idx, "Win"] = vs_df.loc[vs_df["PlayerWin_x"] == "Yes"].shape[0]
            stats.loc[idx, "Loss"] = vs_df.loc[vs_df["PlayerWin_x"] == "No"].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        return stats

    def get_player_by_champion_vs_stats(self):
        merged = pd.merge(
            self.merged, self.merged, how="inner", on=["GameId", "IngameRole"]
        )
        merged = merged.loc[merged["Team_x"] != merged["Team_y"]]

        idx = pd.MultiIndex.from_tuples(
            list(
                set(
                    merged[["player_id_x", "Champion_x", "Champion_y"]].itertuples(
                        index=False
                    )
                )
            ),
            names=["id", "Champion", "Opponent"],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for player_id, champ1, champ2 in stats.index:
            idx = (player_id, champ1, champ2)
            partial_df = merged.loc[
                (merged["player_id_x"] == player_id)
                & (merged["Champion_x"] == champ1)
                & (merged["Champion_y"] == champ2)
            ]
            stats.loc[idx, "Player"] = partial_df["Name_x"].iloc[-1]
            stats.loc[idx, "Team"] = partial_df["Team_x"].iloc[-1]
            stats.loc[idx, "Games"] = partial_df.shape[0]
            stats.loc[idx, "Win"] = partial_df.loc[
                partial_df["PlayerWin_x"] == "Yes"
            ].shape[0]
            stats.loc[idx, "Loss"] = partial_df.loc[
                partial_df["PlayerWin_x"] == "No"
            ].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        stats.reset_index(level="id", drop=True, inplace=True)
        stats.reset_index(inplace=True)
        columns = [
            "Player",
            "Team",
            "Champion",
            "Opponent",
            "Games",
            "Win",
            "Loss",
            "WinRate",
        ]
        assert len(stats.columns) == len(columns)

        return stats[columns].sort_values(by=["Team", "Player", "Champion"])

    def get_duo_player_by_champion_stats(self, role1="Bot", role2="Support"):
        roles = self.players["Role"].unique()
        assert role1 != role2 and role1 in roles and role2 in roles

        role1_df = self.merged.loc[self.merged["IngameRole"] == role1]
        role2_df = self.merged.loc[self.merged["IngameRole"] == role2]
        merged = pd.merge(role1_df, role2_df, how="inner", on=["GameId", "Team"])
        assert merged.shape[0] == self.games.shape[0] * 2

        idx = pd.MultiIndex.from_tuples(
            list(
                set(
                    merged[
                        ["player_id_x", "player_id_y", "Champion_x", "Champion_y"]
                    ].itertuples(index=False)
                )
            ),
            names=["id1", "id2", role1, role2],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for id1, id2, champ1, champ2 in stats.index:
            idx = (id1, id2, champ1, champ2)
            partial_df = merged.loc[
                (merged["player_id_x"] == id1)
                & (merged["player_id_y"] == id2)
                & (merged["Champion_x"] == champ1)
                & (merged["Champion_y"] == champ2)
            ]

            stats.loc[idx, "Player1"] = partial_df["Name_x"].iloc[-1]
            stats.loc[idx, "Player2"] = partial_df["Name_y"].iloc[-1]
            stats.loc[idx, "Games"] = partial_df.shape[0]
            stats.loc[idx, "Win"] = partial_df.loc[
                partial_df["PlayerWin_x"] == "Yes"
            ].shape[0]
            stats.loc[idx, "Loss"] = partial_df.loc[
                partial_df["PlayerWin_x"] == "No"
            ].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        stats.reset_index(level=["id1", "id2"], drop=True, inplace=True)
        stats.reset_index(inplace=True)

        columns = [
            "Player1",
            "Player2",
            role1,
            role2,
            "Games",
            "Win",
            "Loss",
            "WinRate",
        ]
        assert len(columns) == len(stats.columns)

        return stats[columns].sort_values(by=["Player1", "Player2", role1, role2])

    def get_ban_stats(self):
        teams_lst = self.games[["Team1", "Team2"]].unstack().unique()
        index_lst = []
        for team_name in teams_lst:
            df = self.games.loc[
                (self.games["Team1"] == team_name) | (self.games["Team2"] == team_name)
            ]
            ban_lst = list(
                set(
                    reduce(
                        lambda x, y: x + y,
                        df[["Team1Bans", "Team2Bans"]]
                        .unstack()
                        .transform(lambda x: x.split(",")),
                    )
                )
            )
            ban_lst.remove("None") if "None" in ban_lst else None
            for champion in ban_lst:
                index_lst.append((team_name, champion))

        idx = pd.MultiIndex.from_tuples(index_lst, names=["Team", "Champion"])
        columns = [
            "Games",
            "By",
            "Against",
            "By Rate",
            "Against Rate",
            "By Blue",
            "By Red",
        ]
        stats = pd.DataFrame(index=idx, columns=columns).sort_index()
        stats[columns] = 0

        for match in self.games.itertuples():
            team1_name = match.Team1
            team2_name = match.Team2
            team1_bans = list(set(match.Team1Bans.split(",")))
            team2_bans = list(set(match.Team2Bans.split(",")))
            team1_bans.remove("None") if "None" in team1_bans else None
            team2_bans.remove("None") if "None" in team2_bans else None

            for champion in team1_bans:
                stats.loc[(team1_name, champion), "By"] += 1
                stats.loc[(team2_name, champion), "Against"] += 1
                stats.loc[(team1_name, champion), "By Blue"] += 1
            for champion in team2_bans:
                stats.loc[(team2_name, champion), "By"] += 1
                stats.loc[(team1_name, champion), "Against"] += 1
                stats.loc[(team2_name, champion), "By Red"] += 1
        for team_name in stats.index.get_level_values(0).unique():
            stats.loc[team_name, "Games"] = self.games.loc[
                (self.games["Team1"] == team_name) | (self.games["Team2"] == team_name)
            ].shape[0]
        stats[["By Rate", "Against Rate"]] = stats[["By", "Against"]].divide(
            stats["Games"], axis=0
        )

        return stats

    def get_champions_by_position_stats(self):
        idx = list(set(self.merged[["Champion", "IngameRole"]].itertuples(index=False)))
        idx = pd.MultiIndex.from_tuples(idx, names=["Champion", "Position"])
        columns = ["Games", "Win", "Loss", "WinRate"]
        stats = pd.DataFrame(index=idx, columns=columns).sort_index()
        stats[columns] = 0

        for champion, position in stats.index:
            idx = (champion, position)
            df = self.merged.loc[
                (self.merged["Champion"] == champion)
                & (self.merged["IngameRole"] == position)
            ]
            stats.loc[idx, "Games"] = df.shape[0]
            stats.loc[idx, "Win"] = df.loc[df["PlayerWin"] == "Yes"].shape[0]
            stats.loc[idx, "Loss"] = df.loc[df["PlayerWin"] == "No"].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        return stats

    def get_players_by_position_stats(self):
        idx = list(
            set(self.merged[["player_id", "IngameRole"]].itertuples(index=False))
        )
        idx = pd.MultiIndex.from_tuples(idx, names=["Player_id", "Position"])
        columns = [
            "Player",
            "Team",
            "Games",
            "Win",
            "Loss",
            "WinRate",
            "Kills",
            "Deaths",
            "Assists",
            "KDA",
            "DPM",
            "CS",
            "CSPM",
            "Gold",
            "GPM",
            "KP",
            "KS",
            "GS",
            "ChampionsPlayed",
        ]
        stats = pd.DataFrame(index=idx, columns=columns).sort_index()
        stats[columns] = 0

        for player_id, position in stats.index:
            idx = (player_id, position)
            df = self.merged.loc[
                (self.merged["player_id"] == player_id)
                & (self.merged["IngameRole"] == position)
            ]
            stats.loc[idx, "Player"] = df["Name"].values[-1]
            stats.loc[idx, "Team"] = df["Team"].values[-1]
            stats.loc[idx, "Games"] = df.shape[0]
            stats.loc[idx, "Win"] = df.loc[df["PlayerWin"] == "Yes"].shape[0]
            stats.loc[idx, "Loss"] = df.loc[df["PlayerWin"] == "No"].shape[0]
            mean_columns = [
                "Kills",
                "Deaths",
                "Assists",
                "DPM",
                "CS",
                "CSPM",
                "Gold",
                "GPM",
            ]
            stats.loc[idx, mean_columns] = df[mean_columns].mean()
            stats.loc[idx, "KDA"] = df[["Kills", "Assists"]].unstack().sum()
            deaths = df["Deaths"].sum()
            if deaths > 0:
                stats.loc[idx, "KDA"] = stats.loc[idx, "KDA"] / deaths
            stats.loc[idx, "KP"] = (
                df[["Kills", "Assists"]].unstack().sum() / df["TeamKills"].sum()
                if stats.loc[idx, ["Kills", "Assists"]].sum() > 0
                else 0
            )
            stats.loc[idx, "KS"] = (
                df["Kills"].sum() / df["TeamKills"].sum()
                if stats.loc[idx, "Kills"] > 0
                else 0
            )
            stats.loc[idx, "GS"] = df["Gold"].sum() / df["TeamGold"].sum()
            stats.loc[idx, "ChampionsPlayed"] = len(df["Champion"].unique())

        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        stats.reset_index(inplace=True)
        stats.drop(columns=["Player_id"], inplace=True)
        stats.set_index(["Player", "Position"], inplace=True, drop=True)
        stats.sort_values(by=["Team", "Position", "Player"], inplace=True)

        return stats

    def get_duo_champions_vs_stats(self, role1="Bot", role2="Support"):
        roles = self.players["Role"].unique()
        assert role1 != role2 and role1 in roles and role2 in roles

        role1_df = self.merged.loc[self.merged["IngameRole"] == role1]
        role2_df = self.merged.loc[self.merged["IngameRole"] == role2]
        columns = ["Champion", "Team", "PlayerWin", "IngameRole", "GameId"]
        merged = pd.merge(
            role1_df[columns], role2_df[columns], how="inner", on=["GameId", "Team"]
        )
        merged = pd.merge(merged, merged, how="inner", on="GameId")
        merged = merged.loc[merged["Team_x"] != merged["Team_y"]]

        idx_lst = list(
            set(
                merged[
                    ["Champion_x_x", "Champion_y_x", "Champion_x_y", "Champion_y_y"]
                ].itertuples(index=False)
            )
        )
        idx = pd.MultiIndex.from_tuples(
            idx_lst, names=[f"{role1} 1", f"{role2} 1", f"{role1} 2", f"{role2} 2"]
        )
        columns = ["Games", "Win", "Loss", "WinRate"]
        stats = pd.DataFrame(index=idx, columns=columns).sort_index()
        stats[columns] = 0

        for champ1, champ2, champ3, champ4 in stats.index:
            idx = (champ1, champ2, champ3, champ4)
            df = merged.loc[
                (merged["Champion_x_x"] == champ1)
                & (merged["Champion_y_x"] == champ2)
                & (merged["Champion_x_y"] == champ3)
                & (merged["Champion_y_y"] == champ4)
            ]
            stats.loc[idx, "Games"] = df.shape[0]
            stats.loc[idx, "Win"] = df.loc[df["PlayerWin_x_x"] == "Yes"].shape[0]
            stats.loc[idx, "Loss"] = df.loc[df["PlayerWin_x_x"] == "No"].shape[0]
        stats["WinRate"] = stats["Win"].divide(stats["Games"])

        return stats


def parse_input(input_string, option):
    if option == 0:
        return list(map(int, input_string.split()))
    elif option == 1:
        return list(map(int, input_string.split("-")))
    elif option == 2:
        return int(input_string[1:])
    else:
        return int(input_string)


def parse_years(years):
    if years == "":
        years = list(range(START_YEAR, END_YEAR + 1))
    elif " " in years:
        years = parse_input(years, 0)
    elif "-" in years:
        years = parse_input(years, 1)
        years = list(range(years[0], years[1] + 1))
    elif years.startswith("^"):
        excluded = parse_input(years, 2)
        years = list(range(START_YEAR, excluded)) + list(
            range(excluded + 1, END_YEAR + 1)
        )
    else:
        years = [parse_input(years, 3)]

    return years


def parse_leagues(leagues):
    candidate_leagues = ["LCK", "LPL", "LEC", "LCS", "WCS", "MSI"]
    mapping = lambda x: candidate_leagues[x]
    if leagues == "":
        leagues = candidate_leagues.copy()
    elif " " in leagues:
        leagues = list(map(mapping, parse_input(leagues, 0)))
    elif leagues.startswith("^"):
        excluded = parse_input(leagues, 2)
        leagues = list(
            map(
                mapping,
                list(range(excluded))
                + list(range(excluded + 1, len(candidate_leagues) + 1)),
            )
        )
    else:
        leagues = [candidate_leagues[parse_input(leagues, 3)]]

    if "LCK" in leagues:
        leagues.append("LTC")
    if "LEC" in leagues:
        leagues.append("EU LCS")
    if "LCS" in leagues:
        leagues.append("NA LCS")

    return leagues


def parse_pages(pages, target_pages):
    mapping = lambda x: pages[x]
    if target_pages == "":
        target_pages = pages.copy()
    elif " " in target_pages:
        target_pages = list(map(mapping, parse_input(target_pages, 0)))
    elif target_pages.startswith("^"):
        excluded = parse_input(target_pages, 2)
        target_pages = list(
            map(
                mapping,
                list(range(excluded)) + list(range(excluded + 1, len(pages) + 1)),
            )
        )
    else:
        target_pages = [pages[parse_input(target_pages, 3)]]

    return target_pages


def select_options():
    logging.info("Get leagues ...")
    leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    logging.info("Get league complete")

    years = input("Input years (2011 ~ current year)\n")
    years = parse_years(years)
    logging.info("%s years", years)

    logging.info("Get tournaments ...")
    tournaments = pd.DataFrame()
    for year in years:
        df = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        tournaments = pd.concat([tournaments, df], ignore_index=True)
    logging.info("Get tournaments complete")

    candidate_leagues = ["LCK", "LPL", "LEC", "LCS", "WCS", "MSI"]
    target_leagues = input(
        ", ".join(map(lambda x: f"{x[0]}: {x[1]}", enumerate(candidate_leagues))) + "\n"
    )
    target_leagues = parse_leagues(target_leagues)
    logging.info("%s leagues", target_leagues)
    target_condition = leagues["League Short"].isin(target_leagues)
    league = leagues.loc[target_condition, "League"].values
    target_condition = tournaments["League"].isin(league)
    pages = tournaments.loc[target_condition, "OverviewPage"].values

    target_pages = input(
        ", ".join(map(lambda x: f"{x[0]}: {x[1]}", enumerate(pages))) + "\n"
    )
    target_pages = parse_pages(pages, target_pages)

    logging.info("Read scoreboard games ...")
    scoreboard_games = pd.DataFrame()
    for year in years:
        games = pd.read_csv(f"./csv/scoreboard_games/{year}_scoreboard_games.csv")
        scoreboard_games = pd.concat([scoreboard_games, games], ignore_index=True)
    scoreboard_games = scoreboard_games.loc[
        scoreboard_games["OverviewPage"].isin(target_pages)
    ]
    logging.info("Read scoreboard games complete")

    logging.info("Read scoreboard players ...")
    scoreboard_players = pd.DataFrame()
    for year in years:
        players = pd.read_csv(f"./csv/scoreboard_players/{year}_scoreboard_players.csv")
        scoreboard_players = pd.concat([scoreboard_players, players], ignore_index=True)
    scoreboard_players = scoreboard_players.loc[
        scoreboard_players["OverviewPage"].isin(target_pages)
    ]
    logging.info("Read scoreboard players complete")

    patch_version = input("Patch version: ")
    if patch_version != "":
        scoreboard_games = scoreboard_games.loc[
            scoreboard_games["Patch"] == patch_version
        ]
        scoreboard_players = scoreboard_players.loc[
            scoreboard_players["Patch"] == patch_version
        ]

    logging.info(
        "scoreboard games: %d | scoreboard players: %d",
        scoreboard_games.shape[0],
        scoreboard_players.shape[0],
    )

    if scoreboard_games.shape[0] * 10 != scoreboard_players.shape[0]:
        logging.error("%s", target_pages)
        sys.exit(-1)

    return scoreboard_games, scoreboard_players


def select_roles(scoreboard_players):
    roles = scoreboard_players["Role"].unique().tolist()

    role1 = int(
        input(", ".join(map(lambda x: f"{x[0]}: {x[1]}", enumerate(roles))) + "\n")
    )
    role1 = roles.pop(role1)
    role2 = int(
        input(", ".join(map(lambda x: f"{x[0]}: {x[1]}", enumerate(roles))) + "\n")
    )
    role2 = roles[role2]

    return role1, role2


def main():
    sheet = Sheet(SHEET_ID)
    sheet.connect_sheet()

    scoreboard_games, scoreboard_players = select_options()
    role1, role2 = select_roles(scoreboard_players)

    players_id = pd.read_csv("./csv/players_id.csv")

    print()

    stats = LoLStats(scoreboard_games, scoreboard_players, players_id)
    print("Team stats ... ", end="")
    teams_stats = stats.get_teams_stats()
    teams_stats.to_csv("./csv/stats/teams.csv")
    sheet.update_sheet("teams", teams_stats)
    print("Complete")

    print("Champion stats ... ", end="")
    champions_stats = stats.get_champions_stats()
    champions_stats.to_csv("./csv/stats/champions.csv")
    sheet.update_sheet("champions", champions_stats)
    print("Complete")

    print("Players Stats ... ", end="")
    players_stats = stats.get_players_stats()
    players_stats.to_csv("./csv/stats/players.csv", index=False)
    sheet.update_sheet("players", players_stats, index=False)
    print("Complete")

    print("Ban stats ... ", end="")
    ban_stats = stats.get_ban_stats()
    ban_stats.to_csv("./csv/stats/ban.csv")
    sheet.update_sheet("ban", ban_stats)
    print("Complete")

    print("Champions by position stats ... ", end="")
    champions_by_position_stats = stats.get_champions_by_position_stats()
    champions_by_position_stats.to_csv("./csv/stats/champions_by_position.csv")
    sheet.update_sheet("champions_by_position", champions_by_position_stats)
    print("Complete")

    print("Duo Stats ... ", end="")
    duo_stats = stats.get_duo_champions_stats(role1, role2)
    duo_stats.to_csv("./csv/stats/duo.csv")
    sheet.update_sheet("duo", duo_stats)
    print("Complete")

    print("Vs Stats ... ", end="")
    vs_stats = stats.get_vs_stats()
    vs_stats.to_csv("./csv/stats/vs.csv")
    sheet.update_sheet("vs", vs_stats)
    print("Complete")

    print("Duo vs Stats ... ", end="")
    duo_champions_vs_stats = stats.get_duo_champions_vs_stats(role1, role2)
    duo_champions_vs_stats.to_csv("./csv/stats/duo_champions_vs_stats.csv")
    sheet.update_sheet("duo_vs", duo_champions_vs_stats)
    print("Complete")

    print("Player by Champion Stats ... ", end="")
    player_by_champion_stats = stats.get_player_by_champion_stats()
    player_by_champion_stats.to_csv("./csv/stats/player_by_champion.csv", index=False)
    sheet.update_sheet("player_by_champion", player_by_champion_stats, index=False)
    print("Complete")

    print("Player by champion vs stats ... ", end="")
    player_by_champion_vs_stats = stats.get_player_by_champion_vs_stats()
    player_by_champion_vs_stats.to_csv(
        "./csv/stats/player_by_champion_vs.csv", index=False
    )
    sheet.update_sheet(
        "player_by_champion_vs", player_by_champion_vs_stats, index=False
    )
    print("Complete")

    print("Duo player by champion stats ... ", end="")
    duo_player_by_champion_stats = stats.get_duo_player_by_champion_stats(role1, role2)
    duo_player_by_champion_stats.to_csv(
        "./csv/stats/duo_player_by_champion.csv", index=False
    )
    sheet.update_sheet(
        "duo_player_by_champion", duo_player_by_champion_stats, index=False
    )
    print("Complete")

    print("Players by position stats")
    players_by_position_stats = stats.get_players_by_position_stats()
    players_by_position_stats.to_csv("./csv/stats/players_by_position.csv")
    sheet.update_sheet("players_by_position", players_by_position_stats)
    print("Complete")


if __name__ == "__main__":
    main()

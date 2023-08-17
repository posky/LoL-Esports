# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import logging
import webbrowser
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from lol_fandom import get_leagues, get_player_redirects

pd.set_option("display.max_columns", None)
pd.set_option("display.max_columns", None)


# %%
def get_wiki_url(link):
    return f'https://lol.fandom.com/wiki/{link.replace(" ", "_")}'


def get_new_id(ids):
    if "team_id" in ids.columns:
        id_list = sorted(ids["team_id"].unique())
    else:
        id_list = sorted(ids["player_id"].unique())

    if len(id_list) == 0:
        return 1
    prev = id_list[0]
    for id in id_list[1:]:
        if prev + 1 != id:
            return prev + 1
        prev = id
    return prev + 1


def get_team_id(teams, name):
    return teams.loc[teams["team"] == name, "team_id"].iloc[0]


def get_player_id(players, name):
    return players.loc[players["player"] == name, "player_id"].iloc[0]


def get_player_redirects_list(player_link):
    pr = get_player_redirects(where=f'PR.AllName="{player_link}"')
    if pr is None:
        return []
    link = pr["OverviewPage"].iloc[0]
    lst = get_player_redirects(where=f'PR.OverviewPage="{link}"')["AllName"].values
    return list(map(lambda x: x.lower(), lst))


def get_players_id(players, player_link):
    return players.loc[
        players["player"].str.lower().isin(get_player_redirects_list(player_link))
    ]


def concat_teams(teams, name, new_id=-1):
    if new_id == -1:
        new_id = get_new_id(teams)
    df = pd.concat(
        [teams, pd.Series({"team": name, "team_id": new_id}).to_frame().T],
        ignore_index=True,
    )
    return df


def concat_players(players, name, new_id=-1):
    if new_id == -1:
        new_id = get_new_id(players)
    df = pd.concat(
        [players, pd.Series({"player": name, "player_id": new_id}).to_frame().T],
        ignore_index=True,
    )
    return df


def split_string(string, delimiter=";;"):
    return list(map(lambda x: x.strip(), string.split(delimiter)))


def extract_players(row):
    if pd.isna(row.RosterLinks):
        return []

    roles = ["top", "jungle", "mid", "bot", "support"]

    player_names = []
    roster = split_string(row.RosterLinks)
    role = split_string(row.Roles)
    for i in range(len(roster)):
        positions = split_string(role[i], ",")
        is_player = False
        for pos in positions:
            if pos.lower() in roles:
                is_player = True
                break
        if is_player:
            player_names.append(roster[i])
    return list(set(player_names))


# %%
leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
leagues

# %% [markdown]
# ## Teams
#

# %%
teams = pd.read_csv("./csv/teams_id.csv")
names = []

for year in tqdm(range(2011, 2024)):
    tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
    scoreboard_games = pd.read_csv(
        f"./csv/scoreboard_games/{year}_scoreboard_games.csv"
    )
    tournament_rosters = pd.read_csv(
        f"./csv/tournament_rosters/{year}_tournament_rosters.csv"
    )
    logging.info("%d - tournament %d", year, tournaments.shape[0])
    logging.info("%d - scoreboard games %d", year, scoreboard_games.shape[0])
    logging.info("%d - tournament rosters %d", year, tournament_rosters.shape[0])
    for page in tqdm(tournaments["OverviewPage"]):
        logging.info("\t%s", page)
        sg = scoreboard_games.loc[scoreboard_games["OverviewPage"] == page]
        tr = tournament_rosters.loc[tournament_rosters["OverviewPage"] == page]
        team_names = list(
            set(
                list(sg[["Team1", "Team2"]].unstack().unique())
                + list(tr["Team"].unique())
            )
        )
        names = []
        for name in team_names:
            if name not in teams["team"].values:
                names.append(name)
        if len(names) > 0:
            names = sorted(names, key=lambda x: x.lower())
            break
    if len(names) > 0:
        url = f'https://lol.fandom.com/wiki/{page.replace(" ", "_")}'
        webbrowser.open(url)
        print(url)
        print(f"{page}\n{names}")
        for link in names:
            url = f'https://lol.fandom.com/wiki/{link.replace(" ", "_")}'
            webbrowser.open(url)
        break
if len(names) == 0:
    logging.info("Completed")

# %%
teams.loc[teams["team"].str.contains("point", case=False)]

# %%
if len(names) > 0:
    teams = concat_teams(teams, names.pop(0), new_id=-1)
names

# %%
teams = teams.sort_values(by=["team_id", "team"]).reset_index(drop=True)
teams

# %%
teams["lower"] = teams["team"].str.lower()
teams["lower"] = teams["lower"].str.replace(" ", "")
teams["lower"] = teams["lower"].str.replace("-", "")
lst = sorted(list(teams[["lower", "team_id"]].itertuples(index=False)))

is_complete = True
prev = lst[0]
for cur in lst[1:]:
    if prev[0] == cur[0] and prev[1] != cur[1]:
        is_complete = False
        break
    prev = cur

if not is_complete:
    print("incomplete")
    candidate = teams.loc[teams["lower"] == prev[0]]
    print(candidate)
else:
    print("completed")

# %%
if is_complete:
    teams[["team", "team_id"]].to_csv("./csv/teams_id.csv", index=False)
    print("Save complete")

# %% [markdown]
# ## Players
#

# %%
roles = ["top", "jungle", "mid", "bot", "support"]
is_exception = False

while not is_exception:
    players = pd.read_csv("./csv/players_id.csv")
    names = []

    for year in tqdm(range(2011, 2024)):
        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv"
        )
        scoreboard_players = pd.read_csv(
            f"./csv/scoreboard_players/{year}_scoreboard_players.csv"
        )
        tournament_rosters = pd.read_csv(
            f"./csv/tournament_rosters/{year}_tournament_rosters.csv"
        )
        logging.info("%d", year)
        logging.info(
            (
                "%d - %d tournaments | %d scoreboard games | %d scoreboard players | %d"
                " tournament rosters"
            ),
            year,
            tournaments.shape[0],
            scoreboard_games.shape[0],
            scoreboard_players.shape[0],
            tournament_rosters.shape[0],
        )
        for page in tqdm(tournaments["OverviewPage"]):
            logging.info("\t%s", page)
            sp = scoreboard_players.loc[scoreboard_players["OverviewPage"] == page]
            rosters = tournament_rosters.loc[tournament_rosters["OverviewPage"] == page]
            player_names = []
            for row in rosters.itertuples():
                player_names += extract_players(row)
            player_names = list(set(player_names + list(sp["Link"].unique())))
            for name in player_names:
                if name not in players["player"].values:
                    names.append(name)
            names = sorted(names)
            if len(names) > 0:
                break
        if len(names) > 0:
            print(f"{page}")
            print(f"{names}")
            break
    if len(names) == 0:
        logging.info("Completed")
        break

    while not is_exception and len(names) > 0:
        name = names[0]
        df = get_players_id(players, name)
        logging.info("%s", name)
        logging.info("%s", df)

        is_concatenated = False
        if df.shape[0] == 0:
            players = concat_players(players, name)
            is_concatenated = True
            player_id = get_player_id(players, name)
            logging.info("%s - %d", name, player_id)
        else:
            id_lst = df["player_id"].unique()
            if len(id_lst) == 1:
                players = concat_players(players, name, id_lst[0])
                is_concatenated = True
                player_id = get_player_id(players, name)
                logging.info("%s - %d", name, player_id)

        if is_concatenated:
            del names[0]
        else:
            logging.error("There are several ids")
            logging.error("%s\n%s", name, df)
            is_exception = True
            break

    if is_exception:
        break

    players = players.sort_values(by=["player_id", "player"]).reset_index(drop=True)

    while True:
        players["lower"] = players["player"].str.lower()
        lst = sorted(list(players[["lower", "player_id"]].itertuples(index=False)))

        is_complete = True
        prev = lst[0]
        for cur in lst[1:]:
            if prev[0] == cur[0] and prev[1] != cur[1]:
                is_complete = False
                break
            prev = cur

        if not is_complete:
            print("incomplete")
            candidate = players.loc[players["lower"] == prev[0]]
            print(candidate)
            df = players.loc[players["player_id"].isin(candidate["player_id"].values)]
            if candidate.shape[0] != df.shape[0]:
                print("improper id")
                print(df)
                break
            else:
                players.loc[players["lower"] == prev[0], "player_id"] = min(
                    players.loc[players["lower"] == prev[0], "player_id"]
                )
                print("after correction")
                print(players.loc[players["lower"] == prev[0]])
                print("\n")
        else:
            print("completed")
            break

    while True:
        is_complete = True

        counter = Counter(players[["player", "player_id"]].itertuples(index=False))
        lst = [key for key, value in counter.items() if value > 1]
        if len(lst) > 0:
            is_complete = False
            print("incomplete")
            print(lst)
            for name, player_id in lst:
                idx = players.loc[players["player"] == name].index
                players = players.drop(idx[1:])
        else:
            print("completed")
            break

    if is_complete:
        players[["player", "player_id"]].to_csv("./csv/players_id.csv", index=False)
        print("Save complete")

# %%

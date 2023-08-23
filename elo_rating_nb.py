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
#     display_name: Python 3.10.4 64-bit
#     language: python
#     name: python3
# ---

# %%
from itertools import permutations

import pandas as pd

from elo_rating import Team, get_team_id

pd.set_option("display.max_columns", None)


# %%
teams_id = pd.read_csv("./csv/teams_id.csv")
ratings = pd.read_csv(
    "./csv/elo_rating/2023_elo_rating.csv", parse_dates=["last_game_date"]
)
teams_id.shape, ratings.shape


# %%
ratings["mean_error"] = ratings["error"].divide(ratings[["win", "loss"]].sum(axis=1))
ratings


# %%
ratings.loc[(ratings["league"] == "LCK") & (ratings["last_game_date"].dt.year == 2023)]


# %%
ratings.loc[
    (ratings["league"].isin(["LCK", "LPL", "LEC", "LCS"]))
    & (ratings["last_game_date"].dt.year == 2023)
]


# %%
leagues = ["LCK", "LPL", "LEC", "LCS"]
for league in leagues:
    print(league)
    lst = ratings.loc[
        (ratings["league"] == league) & (ratings["last_game_date"].dt.year == 2023),
        "team",
    ].unique()
    print(", ".join(map(lambda x: f"'{x}'", lst)))


# %%
team_names = {
    "T1": "T1",
    "GEN": "Gen.G",
    "DK": "Dplus KIA",
    "LSB": "Liiv SANDBOX",
    "KT": "KT Rolster",
    "HLE": "Hanwha Life Esports",
    "BRO": "BRION",
    "DRX": "DRX",
    "NS": "Nongshim RedForce",
    "KDF": "Kwangdong Freecs",
}
teams = {}
lst_team = []
for name in team_names.values():
    team_id = get_team_id(teams_id, name)
    row = ratings.loc[ratings["team"] == name].iloc[0]
    teams[team_id] = Team(
        name=name,
        league=row["league"],
        win=row["win"],
        loss=row["loss"],
        streak=row["streak"],
        rating=row["rating"],
        last_game_date=row["last_game_date"],
    )
    lst_team.append(teams[team_id])

# %%
matches = [
    ('T1', 'GEN'),
    ('KT', 'HLE'),
]

columns = ["Team1", "Rating1", "Winprob1", "sum", "Winprob2", "Rating2", "Team2"]
probs = pd.DataFrame(columns=columns)
for team1_name, team2_name in matches:
    team1 = teams[get_team_id(teams_id, team_names[team1_name])]
    team2 = teams[get_team_id(teams_id, team_names[team2_name])]
    lst = [
        team1.name,
        team1.rating.rating,
        team1.get_win_probability(team2) * 100,
        0,
        team2.get_win_probability(team1) * 100,
        team2.rating.rating,
        team2.name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs["sum"] = probs[["Winprob1", "Winprob2"]].sum(axis=1)
probs


# %%
columns = ["Team1", "Rating1", "2:0", "2:1", "1:2", "0:2", "Rating2", "Team2"]
probs = pd.DataFrame(columns=columns)
for team1_name, team2_name in matches:
    team1 = teams[get_team_id(teams_id, team_names[team1_name])]
    team2 = teams[get_team_id(teams_id, team_names[team2_name])]
    team1_prob = team1.get_win_probability(team2)
    team2_prob = team2.get_win_probability(team1)
    lst = [
        team1.name,
        team1.rating.rating,
        team1_prob**2 * 100,
        (team1_prob**2) * team2_prob * 2 * 100,
        (team2_prob**2) * team1_prob * 2 * 100,
        team2_prob**2 * 100,
        team2.rating.rating,
        team2_name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs["Winprob1"] = probs[["2:0", "2:1"]].sum(axis=1)
probs["Winprob2"] = probs[["1:2", "0:2"]].sum(axis=1)
columns = [
    "Team1",
    "Rating1",
    "Winprob1",
    "2:0",
    "2:1",
    "1:2",
    "0:2",
    "Winprob2",
    "Rating2",
    "Team2",
]
probs[columns]


# %%
columns = [
    "Team1",
    "Rating1",
    "Winprob1",
    "3:0",
    "3:1",
    "3:2",
    "2:3",
    "1:3",
    "0:3",
    "Winprob2",
    "Rating2",
    "Team2",
]
probs = pd.DataFrame(columns=columns)
for team1_name, team2_name in matches:
    team1 = teams[get_team_id(teams_id, team_names[team1_name])]
    team2 = teams[get_team_id(teams_id, team_names[team2_name])]
    team1_prob = team1.get_win_probability(team2)
    team2_prob = team2.get_win_probability(team1)
    lst = [
        team1.name,
        team1.rating.rating,
        0,
        team1_prob**3 * 100,
        (team1_prob**3) * team2_prob * 3 * 100,
        (team1_prob**3) * (team2_prob**2) * 6 * 100,
        (team2_prob**3) * (team1_prob**2) * 6 * 100,
        (team2_prob**3) * team1_prob * 3 * 100,
        team2_prob**3 * 100,
        0,
        team2.rating.rating,
        team2.name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs["Winprob1"] = probs[["3:0", "3:1", "3:2"]].sum(axis=1)
probs["Winprob2"] = probs[["2:3", "1:3", "0:3"]].sum(axis=1)
probs


# %%

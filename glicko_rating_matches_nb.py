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
from itertools import permutations

import pandas as pd

from glicko_rating_matches import GlickoSystem, Team
from lol_fandom import get_tournaments, get_match_schedule

pd.set_option("display.max_columns", None)


# %%
def get_team_id(teams_id, team_name):
    return teams_id.loc[teams_id["team"] == team_name, "team_id"].iloc[0]


# %%
teams_id = pd.read_csv("./csv/teams_id.csv")
ratings = pd.read_csv(
    "./csv/glicko_rating/glicko_rating_2023.csv", parse_dates=["last_game_date"]
)
scoreboard_games = pd.read_csv("./csv/scoreboard_games/2023_scoreboard_games.csv")
teams_id.shape, ratings.shape, scoreboard_games.shape

# %%
scoreboard_games.loc[scoreboard_games["League"] == "LCK"].tail(6)

# %%
ratings.loc[(ratings["league"] == "LCK") & (ratings["last_game_date"].dt.year == 2023)]

# %%
scoreboard_games.loc[scoreboard_games["League"] == "LPL"].tail(9)

# %%
ratings.loc[(ratings["league"] == "LPL") & (ratings["last_game_date"].dt.year == 2023)]

# %%
scoreboard_games.loc[scoreboard_games["League"] == "LEC"].tail()

# %%
ratings.loc[(ratings["league"] == "LEC") & (ratings["last_game_date"].dt.year == 2023)]

# %%
scoreboard_games.loc[scoreboard_games["League"] == "LCS"].tail()

# %%
ratings.loc[(ratings["league"] == "LCS") & (ratings["last_game_date"].dt.year == 2023)]

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
leagues = {
    "LCK": "LoL Champions Korea",
    "LPL": "Tencent LoL Pro League",
    "LEC": "LoL EMEA Championship",
    "LCS": "League of Legends Championship Series",
    "MSI": "Mid-Season Invitational",
    "WCS": "World Championship",
}

tournaments = get_tournaments(where=f'T.Year=2023 and T.League="{leagues["WCS"]}"')
tournaments

# %%
page = "LCK/2023 Season/Regional Finals"
match_schedules = get_match_schedule(where=f'MS.OverviewPage="{page}"').sort_values(
    by=["DateTime UTC"], ignore_index=True
)


teams = {}
lst_team = []
for team_name in match_schedules[["Team1", "Team2"]].unstack().unique():
    try:
        team_id = get_team_id(teams_id, team_name)
    except Exception as e:
        print(e)
        continue
    row = ratings.loc[ratings["team"] == team_name].iloc[0]
    teams[team_id] = Team(
        name=team_name,
        league=row["league"],
        team_id=row["team_id"],
        win=row["win"],
        loss=row["loss"],
        streak=row["streak"],
        point=row["point"],
        rd=row["rd"],
        sigma=row["sigma"],
        last_game_date=row["last_game_date"],
    )
    lst_team.append(teams[team_id])

# %%
match_schedules = match_schedules.loc[
    match_schedules["Winner"].isna()
    & ~(match_schedules["Team1"] == "TBD")
    & ~(match_schedules["Team2"] == "TBD")
].reset_index(drop=True)
match_schedules.head()

# %%
glicko = GlickoSystem()

# %%
matches = match_schedules.head(10)
matches.head()

# %%
columns = ["Team1", "Point1", "Winprob1", "Winprob2", "Point2", "Team2"]
probs = pd.DataFrame(columns=columns)
for row in matches.itertuples():
    team1_name = row.Team1
    team2_name = row.Team2
    team1 = teams[get_team_id(teams_id, team1_name)]
    team2 = teams[get_team_id(teams_id, team2_name)]
    lst = [
        team1.name,
        team1.point,
        team1.get_win_probability(glicko, team2) * 100,
        team2.get_win_probability(glicko, team1) * 100,
        team2.point,
        team2.name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs

# %%
matches_bo3 = matches.loc[matches["BestOf"] == "3"]
columns = ["Team1", "Point1", "2:0", "2:1", "1:2", "0:2", "Point2", "Team2"]
probs = pd.DataFrame(columns=columns)
for row in matches_bo3.itertuples():
    team1_name = row.Team1
    team2_name = row.Team2
    team1 = teams[get_team_id(teams_id, team1_name)]
    team2 = teams[get_team_id(teams_id, team2_name)]
    team1_prob = team1.get_win_probability(glicko, team2)
    team2_prob = team2.get_win_probability(glicko, team1)
    lst = [
        team1.name,
        team1.point,
        team1_prob**2 * 100,
        (team1_prob**2) * team2_prob * 2 * 100,
        (team2_prob**2) * team1_prob * 2 * 100,
        team2_prob**2 * 100,
        team2.point,
        team2_name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs["Winprob1"] = probs[["2:0", "2:1"]].sum(axis=1)
probs["Winprob2"] = probs[["1:2", "0:2"]].sum(axis=1)
columns = [
    "Team1",
    "Point1",
    "Winprob1",
    "2:0",
    "2:1",
    "1:2",
    "0:2",
    "Winprob2",
    "Point2",
    "Team2",
]
probs[columns]

# %%
matches_bo5 = matches.loc[matches["BestOf"] == "5"]
columns = [
    "Team1",
    "Point1",
    "Winprob1",
    "3:0",
    "3:1",
    "3:2",
    "2:3",
    "1:3",
    "0:3",
    "Winprob2",
    "Point2",
    "Team2",
]
probs = pd.DataFrame(columns=columns)
for row in matches_bo5.itertuples():
    team1_name = row.Team1
    team2_name = row.Team2
    team1 = teams[get_team_id(teams_id, team1_name)]
    team2 = teams[get_team_id(teams_id, team2_name)]
    team1_prob = team1.get_win_probability(glicko, team2)
    team2_prob = team2.get_win_probability(glicko, team1)
    lst = [
        team1.name,
        team1.point,
        0,
        team1_prob**3 * 100,
        (team1_prob**3) * team2_prob * 3 * 100,
        (team1_prob**3) * (team2_prob**2) * 6 * 100,
        (team2_prob**3) * (team1_prob**2) * 6 * 100,
        (team2_prob**3) * team1_prob * 3 * 100,
        team2_prob**3 * 100,
        0,
        team2.point,
        team2.name,
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs["Winprob1"] = probs[["3:0", "3:1", "3:2"]].sum(axis=1)
probs["Winprob2"] = probs[["2:3", "1:3", "0:3"]].sum(axis=1)
probs

# %%
print()

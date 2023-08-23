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
from functools import reduce

import pandas as pd

from elo_rating_player import Team, Player


pd.set_option('display.max_columns', None)


# %%
def get_team_id(teams_id, team_name):
    return teams_id.loc[teams_id['team'] == team_name, 'team_id'].iloc[0]


# %%
teams_id = pd.read_csv('./csv/teams_id.csv')
team_rating = pd.read_csv('./csv/team_elo_rating.csv', parse_dates=['last_game_date'])
player_rating = pd.read_csv('./csv/player_elo_rating.csv', parse_dates=['last_game_date'])
teams_id.shape, team_rating.shape, player_rating.shape

# %%
team_rating

# %%
player_rating

# %%
team_rating.loc[
    (team_rating['League'] == 'LCK') &
    (team_rating['last_game_date'].dt.year == 2023)
]

# %%
player_rating.loc[
    (player_rating['League'] == 'LCK') &
    (player_rating['last_game_date'].dt.year == 2023)
]

# %%
player_rating.loc[
    (player_rating['League'] == 'LPL') &
    (player_rating['last_game_date'].dt.year == 2023)
]

# %%
player_rating.loc[
    (player_rating['League'] == 'LEC') &
    (player_rating['last_game_date'].dt.year == 2023)
]

# %%
player_rating.loc[
    (player_rating['League'] == 'LCS') &
    (player_rating['last_game_date'].dt.year == 2023)
]

# %%
player_rating.loc[
    (player_rating['League'].isin(['LCK', 'LPL', 'LEC', 'LCS'])) &
    (player_rating['last_game_date'].dt.year == 2023)
]

# %%
team_rating.loc[
    (team_rating['League'] == 'LCK'), 'Team'
].unique()

# %%
team_names = {
    'T1': 'T1',
    'GEN': 'Gen.G',
    'DK': 'Dplus KIA',
    'LSB': 'Liiv SANDBOX',
    'KT': 'KT Rolster',
    'HLE': 'Hanwha Life Esports',
    'BRO': 'BRION',
    'DRX': 'DRX',
    'NS': 'Nongshim RedForce',
    'KDF': 'Kwangdong Freecs',
}
teams = {}
lst_team = []
for name in team_names.values():
    team_id = get_team_id(teams_id, name)
    row = team_rating.loc[team_rating['Team'] == name].iloc[0]
    teams[team_id] = Team(
        name, row['League'], row['Point'], row['Win'], row['Loss'],
        row['Streak'], row['last_game_date']
    )
    lst_team.append(teams[team_id])

# %%
columns = ['Team1', 'Rating1', 'Winprob1', 'Winprob2', 'Rating2', 'Team2']
probs = pd.DataFrame(columns=columns)
for (team1, team2) in permutations(lst_team, 2):
    lst = [
        team1.name, team1.point, team1.get_win_prob(team2) * 100,
        team2.get_win_prob(team1) * 100, team2.point, team2.name
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs

# %%
matches = [
    ('KDF', 'NS'),
    ('KT', 'HLE'),
    ('LSB', 'GEN'),
    ('DRX', 'T1'),
    ('DK', 'NS'),
    ('BRO', 'KT'),
    ('DRX', 'LSB'),
    ('T1', 'HLE'),
    ('DK', 'BRO'),
    ('GEN', 'KDF'),
]

columns = ['Team1', 'Rating1', 'Winprob1', 'Winprob2', 'Rating2', 'Team2']
probs = pd.DataFrame(columns=columns)
for (team1_name, team2_name) in matches:
    team1 = teams[get_team_id(teams_id, team_names[team1_name])]
    team2 = teams[get_team_id(teams_id, team_names[team2_name])]
    lst = [
        team1.name, team1.point, team1.get_win_prob(team2) * 100,
        team2.get_win_prob(team1) * 100, team2.point, team2.name
    ]
    df = pd.DataFrame(data=[lst], columns=columns)
    probs = pd.concat([probs, df], ignore_index=True)
probs

# %%
team_name_lst = [[team_names[name1]] + [team_names[name2]] for name1, name2 in matches]
team_name_lst = list(set(reduce(lambda x, y: x + y, team_name_lst)))

player_rating.loc[player_rating['Team'].isin(team_name_lst)]

# %%

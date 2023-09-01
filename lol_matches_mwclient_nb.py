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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google_sheet import Sheet


pd.set_option('display.max_columns', None)
with open('./sheet_id.txt', 'r') as f:
    SHEET_ID = f.read()

# %%
sheet = Sheet(SHEET_ID)
sheet.connect_sheet()

# %%
tournaments = pd.read_csv(f'./csv/tournaments/2023_tournaments.csv')
tournaments

# %%
page = tournaments.loc[tournaments['League'] == 'LoL EMEA Championship', 'OverviewPage'].values[0]
page

# %%
scoreboard_games = pd.read_csv(f'./csv/scoreboard_games/2023_scoreboard_games.csv')
scoreboard_games = scoreboard_games.loc[scoreboard_games['OverviewPage'] == page]
scoreboard_games = scoreboard_games.sort_values(by='DateTime UTC').reset_index(drop=True)
scoreboard_games

# %%
scoreboard_players = pd.read_csv('./csv/scoreboard_players/2023_scoreboard_players.csv')
scoreboard_players = scoreboard_players.loc[scoreboard_players['OverviewPage'] == page]
scoreboard_players = scoreboard_players.sort_values(by='DateTime UTC').reset_index()
scoreboard_players

# %%
for id in sorted(scoreboard_games['GameId'].unique()):
    print(id, scoreboard_players.loc[scoreboard_players['GameId'] == id].shape[0])
scoreboard_games.shape[0] * 10 == scoreboard_players.shape[0]


# %%
def champions_stats(games, players):
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby('Champion')

    champions = pd.DataFrame()
    champions[[
        'Kills', 'Deaths', 'Assists', 'Gold',
        'CS', 'DamageToChampions', 'VisionScore']
    ] = grouped[[
        'Kills', 'Deaths', 'Assists', 'Gold',
        'CS', 'DamageToChampions', 'VisionScore'
    ]].mean()
    champions['GamesPlayed'] = grouped['Name'].count()
    champions['By'] = grouped['Name'].nunique()
    champions[['Win', 'Loss']] = grouped['PlayerWin'].value_counts().unstack(fill_value=0).rename(columns={'Yes': 'Win', 'No': 'Loss'})
    champions['WinRate'] = champions['Win'] / champions['GamesPlayed']
    champions['KDA'] = champions[['Kills', 'Assists']].sum(axis=1) / champions['Deaths']
    champions['CS_M'] = champions['CS'] / grouped['Gamelength Number'].mean()
    champions['GoldM'] = champions['Gold'] / grouped['Gamelength Number'].mean()
    champions['KillParticipation'] = champions[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
    champions['KillShare'] = champions['Kills'] / grouped['TeamKills'].mean()
    champions['GoldShare'] = champions['Gold'] / grouped['TeamGold'].mean()
    champions['As'] = grouped['IngameRole'].unique()

    ban_list = games[['Team1Bans', 'Team2Bans']].unstack().str.split(',')
    banned = {}
    for bans in ban_list:
        for b in bans:
            if b not in banned:
                banned[b] = 0
            banned[b] += 1
    for champ, ban in banned.items():
        champions.loc[champ, 'Banned'] = ban

    champions['Games'] = champions[['GamesPlayed', 'Banned']].sum(axis=1)
    champions['PickBanRate'] = champions['Games'] / games.shape[0]
    
    int_types = ['GamesPlayed', 'Win', 'Loss', 'Banned', 'Games']
    champions.loc[:, int_types] = champions.loc[:, int_types].fillna(0)
    champions[int_types] = champions[int_types].astype('int')

    columns = [
        'Games', 'PickBanRate', 'Banned', 'GamesPlayed', 'By', 'Win', 'Loss',
        'WinRate', 'Kills', 'Deaths', 'Assists', 'KDA', 'DamageToChampions',
        'CS', 'CS_M', 'Gold', 'GoldM', 'VisionScore',
        'KillParticipation', 'KillShare', 'GoldShare', 'As'
    ]
    champions = champions[columns]
    
    return champions


champions = champions_stats(scoreboard_games, scoreboard_players)
champions['As'] = champions['As'].str.join(', ')
sheet.update_sheet('champions', champions)


# %%
def players_stats(games, _players):
    merged = pd.merge(_players, games, how='left', on='GameId')
    grouped = merged.groupby('Name')

    players = pd.DataFrame(index=_players['Name'].unique())
    players['Team'] = grouped.tail(1).set_index('Name')['Team']
    players['Games'] = grouped['Champion'].count()
    players[['Win', 'Loss']] = grouped['PlayerWin'].value_counts().unstack(
        fill_value=0).rename(columns={'Yes': 'Win', 'No': 'Loss'})[['Win', 'Loss']]
    players['WinRate'] = players['Win'] / players['Games']
    players[['Kills', 'Deaths', 'Assists', 'CS', 'Gold']] = grouped[['Kills', 'Deaths', 'Assists', 'CS', 'Gold']].mean()
    players['KDA'] = players[['Kills', 'Assists']].sum(axis=1) / players['Deaths']
    players['CS_M'] = players['CS'] / grouped['Gamelength Number'].mean()
    players['GoldM'] = players['Gold'] / grouped['Gamelength Number'].mean()
    players['KillParticipation'] = players[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
    players['KillShare'] = players['Kills'] / grouped['TeamKills'].mean()
    players['GoldShare'] = players['Gold'] / grouped['TeamGold'].mean()
    players['DPM'] = grouped['DamageToChampions'].sum() / grouped['Gamelength Number'].sum()
    players['VisionScoreM'] = grouped['VisionScore'].sum() / grouped['Gamelength Number'].sum()
    players['ChampionsPlayed'] = grouped['Champion'].nunique()
    champs = grouped['Champion'].value_counts(sort=True, ascending=False)
    keys = players.index
    values = []
    for key in keys:
        values.append(list(champs[key].index))
    players['Champs'] = pd.Series(data=values, index=keys)

    columns = [
        'Team', 'Games', 'Win', 'Loss', 'WinRate', 'Kills', 'Deaths', 'Assists',
        'KDA', 'DPM', 'CS', 'CS_M', 'Gold', 'GoldM', 'VisionScoreM',
        'KillParticipation', 'KillShare', 'GoldShare', 'ChampionsPlayed', 'Champs'
    ]
    players = players[columns]
    players.index.name = 'Player'

    return players


players = players_stats(scoreboard_games, scoreboard_players)
players['Champs'] = players['Champs'].str.join(', ')
sheet.update_sheet('players', players)


# %%
def champions_comp_stats(games, players, _comps=['Top', 'Jungle', 'Mid', 'Bot', 'Support']):
    assert isinstance(_comps, list)
    positions = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    for pos in _comps:
        assert pos in positions

    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['GameId', 'Team'])
    data = {}
    for df in grouped:
        df = df[1]
        idx = []
        by = []
        for pos in _comps:
            idx.append(df.loc[df['IngameRole'] == pos, 'Champion'].iloc[0])
            by.append(df.loc[df['IngameRole'] == pos, 'Name'].iloc[0])
        idx = tuple(idx)
        by = tuple(by)
        if idx not in data:
            data[idx] = {
                'Win': 0,
                'Loss': 0,
                'By': []
            }
        result = 'Win' if df.iloc[0]['PlayerWin'] == 'Yes' else 'Loss'
        data[idx][result] += 1
        data[idx]['By'].append(by)
    for idx in data.keys():
        data[idx]['By'] = len(set(data[idx]['By']))
    comps = pd.DataFrame(data=data.values(), index=data.keys())
    comps['Games'] = comps[['Win', 'Loss']].sum(axis=1)
    comps['WinRate'] = comps['Win'] / comps['Games']

    comps = comps[['Games', 'By', 'Win', 'Loss', 'WinRate']]
    comps.index = comps.index.set_names(_comps)
    
    return comps

positions = ['Bot', 'Support']
comps = champions_comp_stats(scoreboard_games, scoreboard_players, positions)
comps = comps.sort_values(by='Games', ascending=False)
sheet.update_sheet('Bot_Support_stats', comps)


# %%
def champions_vs_stats(games, players, champs=None):
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['GameId', 'IngameRole'])

    data = {}
    for df in grouped:
        df = df[1]
        champions = [df.iloc[0], df.iloc[1]]
        idx = [
            (champions[0]['Champion'], champions[1]['Champion']),
            (champions[1]['Champion'], champions[0]['Champion'])
        ]
        for i in idx:
            if i not in data:
                data[i] = {
                    'Win': 0,
                    'Loss': 0,
                    'As': []
                }
        for i, champ in zip(idx, champions):
            result = 'Win' if champ['PlayerWin'] == 'Yes' else 'Loss'
            data[i][result] += 1
            data[i]['As'].append(champ['IngameRole'])
    for key in data.keys():
        data[key]['As'] = list(set(data[key]['As']))
    vs_stats = pd.DataFrame(data=data.values(), index=data.keys())
    vs_stats['Games'] = vs_stats[['Win', 'Loss']].sum(axis=1)
    vs_stats['WinRate'] = vs_stats['Win'] / vs_stats['Games']
    vs_stats = vs_stats[['Games', 'Win', 'Loss', 'WinRate', 'As']]

    if champs is not None:
        champions = []
        for champ in champs:
            if champ in merged['Champion']:
                champions.append(champ)

    vs_stats.index = vs_stats.index.set_names(['Champion1', 'Champion2'])
    
    return vs_stats if champs is None else vs_stats[champions]


vs_stats = champions_vs_stats(scoreboard_games, scoreboard_players)
vs_stats = vs_stats.sort_values(by='Games', ascending=False)
vs_stats['As'] = vs_stats['As'].str.join(', ')
sheet.update_sheet('vs_stats', vs_stats)


# %%
def player_by_champions(games, players, champs=None):
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['Name', 'Champion'])

    players_by = pd.DataFrame()
    players_by['Team'] = grouped.tail(1).set_index(['Name', 'Champion'])['Team']
    players_by['Games'] = grouped['Team'].count()
    players_by[['Win', 'Loss']] = grouped['PlayerWin'].value_counts().unstack(
        fill_value=0).rename(columns={'Yes': 'Win', 'No': 'Loss'})[['Win', 'Loss']]
    players_by['WinRate'] = players_by['Win'] / players_by['Games']
    players_by[['Kills', 'Deaths', 'Assists', 'CS', 'Gold']] = grouped[
        ['Kills', 'Deaths', 'Assists', 'CS', 'Gold']].mean()
    players_by['KDA'] = players_by[['Kills', 'Assists']].sum(axis=1) / players_by['Deaths']
    players_by['CS_M'] = players_by['CS'] / grouped['Gamelength Number'].mean()
    players_by['GoldM'] = players_by['Gold'] / grouped['Gamelength Number'].mean()
    players_by['KillParticipation'] = players_by[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
    players_by['KillShare'] = players_by['Kills'] / grouped['TeamKills'].mean()
    players_by['GoldShare'] = players_by['Gold'] / grouped['TeamGold'].mean()
    players_by['DPM'] = grouped['DamageToChampions'].sum() / grouped['Gamelength Number'].sum()
    players_by['VisionScoreM'] = grouped['VisionScore'].sum() / grouped['Gamelength Number'].sum()

    columns = [
        'Team', 'Games', 'Win', 'Loss', 'WinRate', 'Kills', 'Deaths', 'Assists',
        'KDA', 'DPM', 'CS', 'CS_M', 'Gold', 'GoldM', 'VisionScoreM',
        'KillParticipation', 'KillShare', 'GoldShare'
    ]
    players_by = players_by[columns]

    return players_by

players_by = player_by_champions(scoreboard_games, scoreboard_players)
sheet.update_sheet('player_by_champions', players_by)


# %%
def teams_stats(games, players):
    games['Team1Win'] = games['Winner'].transform(lambda x: x == 1)
    games['Team2Win'] = games['Winner'].transform(lambda x: x == 2)
    games['GPM1'] = games['Team1Gold'] / games['Gamelength Number']
    games['GPM2'] = games['Team2Gold'] / games['Gamelength Number']
    games['KPM1'] = games['Team1Kills'] / games['Gamelength Number']
    games['KPM2'] = games['Team2Kills'] / games['Gamelength Number']
    games['GDPM1'] = (games['Team1Gold'] - games['Team2Gold']) / games['Gamelength Number']
    games['GDPM2'] = (games['Team2Gold'] - games['Team1Gold']) / games['Gamelength Number']
    grouped1 = games.groupby('Team1')
    grouped2 = games.groupby('Team2')

    teams = pd.DataFrame(index=pd.concat([games['Team1'], games['Team2']]).unique())
    teams.index = teams.index.set_names('Team')

    teams['Games'] = grouped1['Winner'].count() + grouped2['Winner'].count()
    teams['Win'] = grouped1['Team1Win'].sum() + grouped2['Team2Win'].sum()
    teams['Loss'] = grouped1['Team2Win'].sum() + grouped2['Team1Win'].sum()
    teams['WinRate'] = teams['Win'] / teams['Games']
    teams['GameDuration'] = (grouped1['Gamelength Number'].sum() + 
        grouped2['Gamelength Number'].sum()) / teams['Games']
    teams['GPM'] = (grouped1['GPM1'].sum() + grouped2['GPM2'].sum()) / teams['Games']
    teams['GDPM'] = (grouped1['GDPM1'].sum() + grouped2['GDPM2'].sum()) / teams['Games']
    teams['KPM'] = (grouped1['KPM1'].sum() + grouped2['KPM2'].sum()) / teams['Games']
    teams['Out'] = teams['KPM'] * teams['GDPM']

    return teams

teams = teams_stats(scoreboard_games, scoreboard_players)
sheet.update_sheet('teams', teams)

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.set_xlim(teams['GDPM'].min() - 25, teams['GDPM'].max() + 25)
ax.set_ylim(teams['KPM'].min() - 0.01, teams['KPM'].max() + 0.01)
ax = sns.regplot(data=teams, x='GDPM', y='KPM')
y = np.linspace(0, 1.2)
x = 100 / y
sns.lineplot(x=x, y=y)
for index, i in zip(teams.index, range(teams.shape[0])):
    row = teams.iloc[i]
    plt.annotate(index, xy=(row['GDPM'], row['KPM']), xytext=(5, 5), textcoords='offset pixels')


# %%
def players_champions_comp_stats(games, players, _comps=['Top', 'Jungle', 'Mid', 'Bot', 'Support']):
    assert isinstance(_comps, list)
    positions = ['Top', 'Jungle', 'Mid', 'Bot', 'Support']
    for pos in _comps:
        assert pos in positions

    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['GameId', 'Team'])
    data = {}
    for df in grouped:
        df = df[1]
        player_idx = []
        champion_idx = []        
        for pos in _comps:
            player_idx.append(df.loc[df['IngameRole'] == pos, 'Name'].iloc[0])
            champion_idx.append(df.loc[df['IngameRole'] == pos, 'Champion'].iloc[0])
        idx = tuple([df['Team'].iloc[0]] + player_idx + champion_idx)
        if idx not in data:
            data[idx] = {
                'Win': 0,
                'Loss': 0
            }
        result = 'Win' if df.iloc[0]['PlayerWin'] == 'Yes' else 'Loss'
        data[idx][result] += 1
    comps = pd.DataFrame(data=data.values(), index=data.keys())
    comps['Games'] = comps[['Win', 'Loss']].sum(axis=1)
    comps['WinRate'] = comps['Win'] / comps['Games']

    comps = comps[['Games', 'Win', 'Loss', 'WinRate']]
    comps.index = comps.index.set_names(
        ['Team'] + \
        list(map(lambda x: x + 'Player', _comps)) + \
        _comps
    )

    return comps

positions = ['Bot', 'Support']
player_comps = players_champions_comp_stats(scoreboard_games, scoreboard_players, positions)
player_comps = player_comps.sort_values(by=['Games', 'WinRate'], ascending=False)
sheet.update_sheet('Bot_Support_Player_stats', player_comps)

# %%

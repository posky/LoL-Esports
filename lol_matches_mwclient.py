"""
LoLesports variety of stats
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google_sheet import Sheet

from lol_fandom import get_leagues, get_tournaments
from lol_fandom import get_scoreboard_games, get_scoreboard_players

pd.set_option('display.max_columns', None)
with open('./sheet_id.txt', 'r', encoding='utf8') as f:
    SHEET_ID = f.read()


def get_scoreboard_players_by_games(overview, games):
    """Get scoreboard of players with scoreboard of games

    Args:
        overview (str): OverviewPage of tournament
        games (DataFrame): Scoreboard of games

    Returns:
        DataFrame: Scoreboard of players with scoreboard of games
    """
    players = pd.DataFrame()
    for id in games['GameId']:
        temp = get_scoreboard_players(
            f'T.OverviewPage="{overview}" and SP.GameId="{id}"'
        )
        players = pd.concat([players, temp])
    return players.reset_index(drop=True)

def get_champions_stats(games, players):
    """Get statistics of champions

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players

    Returns:
        DataFrame: Statistics of champions
    """
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby('Champion')

    champions = pd.DataFrame()
    champions[[
        'Kills', 'Deaths', 'Assists', 'Gold',
        'CS', 'DamageToChampions', 'VisionScore'
    ]] = grouped[[
        'Kills', 'Deaths', 'Assists', 'Gold',
        'CS', 'DamageToChampions', 'VisionScore'
    ]].mean()
    champions['GamesPlayed'] = grouped['Name'].count()
    champions['By'] = grouped['Name'].nunique()
    champions[['Win', 'Loss']] = \
        grouped['PlayerWin'].value_counts().unstack(fill_value=0).rename(
            columns={'Yes': 'Win', 'No': 'Loss'}
        )[['Win', 'Loss']]
    champions['WinRate'] = champions['Win'] / champions['GamesPlayed']
    champions['KDA'] = \
        champions[['Kills', 'Assists']].sum(axis=1) / champions['Deaths']
    champions['CSPM'] = champions['CS'] / grouped['Gamelength Number'].mean()
    champions['GPM'] = champions['Gold'] / grouped['Gamelength Number'].mean()
    champions['KP'] = \
        champions[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
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
        'CS', 'CSPM', 'Gold', 'GPM', 'VisionScore',
        'KP', 'KillShare', 'GoldShare', 'As'
    ]
    champions = champions[columns]

    return champions

def get_players_stats(games, _players):
    """Get statistics of players

    Args:
        games (DataFrame): Scoreboard of games
        _players (DataFrame): Scoreboard of players

    Returns:
        DataFrame: Statistics of players
    """
    merged = pd.merge(_players, games, how='left', on='GameId')
    grouped = merged.groupby('Name')

    players = pd.DataFrame(index=_players['Name'].unique())
    players['Team'] = grouped.tail(1).set_index('Name')['Team']
    players['Games'] = grouped['Champion'].count()
    players[['Win', 'Loss']] = grouped['PlayerWin'].value_counts().unstack(
        fill_value=0
    ).rename(columns={'Yes': 'Win', 'No': 'Loss'})[['Win', 'Loss']]
    players['WinRate'] = players['Win'] / players['Games']
    players[['Kills', 'Deaths', 'Assists', 'CS', 'Gold', 'VisionScore']] = \
        grouped[['Kills', 'Deaths', 'Assists', 'CS', 'Gold', 'VisionScore']].mean()
    players['KDA'] = players[['Kills', 'Assists']].sum(axis=1) / players['Deaths']
    players['CSPM'] = players['CS'] / grouped['Gamelength Number'].mean()
    players['GPM'] = players['Gold'] / grouped['Gamelength Number'].mean()
    players['KP'] = \
        players[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
    players['KillShare'] = players['Kills'] / grouped['TeamKills'].mean()
    players['GoldShare'] = players['Gold'] / grouped['TeamGold'].mean()
    players['DPM'] = \
        grouped['DamageToChampions'].sum() / grouped['Gamelength Number'].sum()
    players['ChampionsPlayed'] = grouped['Champion'].nunique()
    champs = grouped['Champion'].value_counts(sort=True, ascending=False)
    keys = players.index
    values = []
    for key in keys:
        values.append(list(champs[key].index))
    players['Champs'] = pd.Series(data=values, index=keys)

    columns = [
        'Team', 'Games', 'Win', 'Loss', 'WinRate', 'Kills', 'Deaths', 'Assists',
        'KDA', 'DPM', 'CS', 'CSPM', 'Gold', 'GPM', 'VisionScore',
        'KP', 'KillShare', 'GoldShare', 'ChampionsPlayed', 'Champs'
    ]
    players = players[columns]
    players.index.name = 'Player'

    return players

def get_champions_comp_stats(
    games, players, _comps=('Top', 'Jungle', 'Mid', 'Bot', 'Support')):
    """Get statistics of champions comp

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players
        _comps (tuple, optional): A tuple of target positions.
            Defaults to ('Top', 'Jungle', 'Mid', 'Bot', 'Support').

    Returns:
        DataFrame: Statistics of champions comps
    """
    assert isinstance(_comps, list)
    positions = ('Top', 'Jungle', 'Mid', 'Bot', 'Support')
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
    # for idx in data.keys():
    #     data[idx]['By'] = len(set(data[idx]['By']))
    for key, value in data.items():
        data[key]['By'] = len(set(value['By']))
    comps = pd.DataFrame(data=data.values(), index=data.keys())
    comps['Games'] = comps[['Win', 'Loss']].sum(axis=1)
    comps['WinRate'] = comps['Win'] / comps['Games']

    comps = comps[['Games', 'By', 'Win', 'Loss', 'WinRate']]
    comps.index = comps.index.set_names(_comps)

    return comps

def get_champions_vs_stats(games, players, champs=None):
    """Get statistics of champions vs champions

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players
        champs (tuple or None, optional): Target of champions. Defaults to None.

    Returns:
        DataFrame: Statistics of champions vs champions
    """
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
    # for key in data.keys():
    #     data[key]['As'] = list(set(data[key]['As']))
    for key, value in data.items():
        data[key]['As'] = list(set(value['As']))
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

def get_player_by_champions_stats(games, players):
    """Get statistics of player by champions

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players

    Returns:
        DataFrame: Statistics of player by champions
    """
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['Name', 'Champion'])

    players_by = pd.DataFrame()
    players_by['Team'] = grouped.tail(1).set_index(['Name', 'Champion'])['Team']
    players_by['Games'] = grouped['Team'].count()
    players_by[['Win', 'Loss']] = \
        grouped['PlayerWin'].value_counts().unstack(
            fill_value=0).rename(
                columns={'Yes': 'Win', 'No': 'Loss'}
            )[['Win', 'Loss']]
    players_by['WinRate'] = players_by['Win'] / players_by['Games']
    players_by[['Kills', 'Deaths', 'Assists', 'CS', 'Gold', 'VisionScore']] = grouped[
        ['Kills', 'Deaths', 'Assists', 'CS', 'Gold', 'VisionScore']
    ].mean()
    players_by['KDA'] = \
        players_by[['Kills', 'Assists']].sum(axis=1) / players_by['Deaths']
    players_by['CSPM'] = players_by['CS'] / grouped['Gamelength Number'].mean()
    players_by['GPM'] = players_by['Gold'] / grouped['Gamelength Number'].mean()
    players_by['KP'] = \
        players_by[['Kills', 'Assists']].sum(axis=1) / grouped['TeamKills'].mean()
    players_by['KillShare'] = players_by['Kills'] / grouped['TeamKills'].mean()
    players_by['GoldShare'] = players_by['Gold'] / grouped['TeamGold'].mean()
    players_by['DPM'] = \
        grouped['DamageToChampions'].sum() / grouped['Gamelength Number'].sum()

    columns = [
        'Team', 'Games', 'Win', 'Loss', 'WinRate', 'Kills', 'Deaths', 'Assists',
        'KDA', 'DPM', 'CS', 'CSPM', 'Gold', 'GPM', 'VisionScore',
        'KP', 'KillShare', 'GoldShare'
    ]
    players_by = players_by[columns]

    return players_by

def get_teams_stats(games):
    """Get statistics of teams

    Args:
        games (DataFrame): Scoreboard of games

    Returns:
        DataFrame: Statistics of teams
    """
    games['Team1Win'] = games['Winner'].transform(lambda x: x == 1)
    games['Team2Win'] = games['Winner'].transform(lambda x: x == 2)
    games['Team1WinDuration'] = games['Team1Win'] * games['Gamelength Number']
    games['Team2WinDuration'] = games['Team2Win'] * games['Gamelength Number']
    games['Team1LossDuration'] = games['Team2Win'] * games['Gamelength Number']
    games['Team2LossDuration'] = games['Team1Win'] * games['Gamelength Number']
    games['GPM1'] = games['Team1Gold'] / games['Gamelength Number']
    games['GPM2'] = games['Team2Gold'] / games['Gamelength Number']
    games['GDPM1'] = (games['Team1Gold'] - games['Team2Gold']) / \
        games['Gamelength Number']
    games['GDPM2'] = (games['Team2Gold'] - games['Team1Gold']) / \
        games['Gamelength Number']
    games['KPM1'] = games['Team1Kills'] / games['Gamelength Number']
    games['KPM2'] = games['Team2Kills'] / games['Gamelength Number']
    games['KDPM1'] = (games['Team1Kills'] - games['Team2Kills']) / \
        games['Gamelength Number']
    games['KDPM2'] = (games['Team2Kills'] - games['Team1Kills']) / \
        games['Gamelength Number']

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
    teams['WinGameDuration'] = \
        (grouped1['Team1WinDuration'].sum() +
        grouped2['Team2WinDuration'].sum()) / teams['Win']
    teams['LossGameDuration'] = \
        (grouped1['Team2WinDuration'].sum() +
        grouped2['Team1WinDuration'].sum()) / teams['Loss']
    teams['Kills'] = \
        (grouped1['Team1Kills'].sum() + grouped2['Team2Kills'].sum()) / \
        teams['Games']
    teams['Deaths'] = \
        (grouped1['Team2Kills'].sum() + grouped2['Team1Kills'].sum()) / \
        teams['Games']
    teams['GPM'] = (grouped1['GPM1'].sum() + grouped2['GPM2'].sum()) / teams['Games']
    teams['GDPM'] = (grouped1['GDPM1'].sum() + grouped2['GDPM2'].sum()) / teams['Games']
    teams['KPM'] = (grouped1['KPM1'].sum() + grouped2['KPM2'].sum()) / teams['Games']
    teams['KDPM'] = (grouped1['KDPM1'].sum() + grouped2['KDPM2'].sum()) / teams['Games']

    teams['Outlier'] = teams['KPM'] * teams['GDPM']

    columns = [
        'Games', 'Win', 'Loss', 'WinRate', 'GameDuration',
        'WinGameDuration', 'LossGameDuration', 'Kills', 'Deaths',
        'GPM', 'GDPM', 'KPM', 'KDPM', 'Outlier'
    ]

    return teams[columns]

def get_player_champions_comp_stats(
    games, players, _comps=('Top', 'Jungle', 'Mid', 'Bot', 'Support')):
    """Get statistics of player champions comps

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players
        _comps (tuple, optional): Target of positions.
            Defaults to ('Top', 'Jungle', 'Mid', 'Bot', 'Support').

    Returns:
        DataFrame: Statistics of player champions comps
    """
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

def get_player_by_champions_vs_stats(games, players):
    """Get statistics of player by champions vs opponent champions

    Args:
        games (DataFrame): Scoreboard of games
        players (DataFrame): Scoreboard of players

    Returns:
        DataFrame: Statistics of player by champions vs opponent champions
    """
    merged = pd.merge(players, games, how='left', on='GameId')
    grouped = merged.groupby(['GameId', 'IngameRole'])

    data = {}
    for df in grouped:
        df = df[1]
        rows = [df.iloc[0], df.iloc[1]]
        idx = [
            (
                rows[0]['Team'], rows[0]['Name'],
                rows[0]['Champion'], rows[1]['Champion']
            ),
            (
                rows[1]['Team'], rows[1]['Name'],
                rows[1]['Champion'], rows[0]['Champion']
            )
        ]
        for i in idx:
            if i not in data:
                data[i] = {
                    'Win': 0,
                    'Loss': 0,
                    'Kills': 0,
                    'Deaths': 0,
                    'Assists': 0,
                    'CS': 0,
                    'Gold': 0,
                    'VisionScore': 0,
                    'DamageToChampions': 0,
                    'Gamelength Number': 0
                }
        for i, row in zip(idx, rows):
            result = 'Win' if row['PlayerWin'] == 'Yes' else 'Loss'
            data[i][result] += 1
            data[i]['Kills'] += row['Kills']
            data[i]['Deaths'] += row['Deaths']
            data[i]['Assists'] += row['Assists']
            data[i]['CS'] += row['CS']
            data[i]['Gold'] += row['Gold']
            data[i]['VisionScore'] += row['VisionScore']
            data[i]['DamageToChampions'] += row['DamageToChampions']
            data[i]['Gamelength Number'] += row['Gamelength Number']
    stats = pd.DataFrame(data=data.values(), index=data.keys())
    stats['Games'] = stats[['Win', 'Loss']].sum(axis=1)
    stats['WinRate'] = stats['Win'] / stats['Games']
    columns = [
        'Kills', 'Deaths', 'Assists', 'CS', 'Gold',
        'VisionScore', 'Gamelength Number'
    ]
    for col in columns:
        stats[col] = stats[col] / stats['Games']
    stats['KDA'] = stats[['Kills', 'Assists']].sum(axis=1) / stats['Deaths']
    stats['CSPM'] = stats['CS'] / stats['Gamelength Number']
    stats['GPM'] = stats['Gold'] / stats['Gamelength Number']
    stats['DPM'] = stats['DamageToChampions'] / stats['Gamelength Number']
    columns = [
        'Games', 'Win', 'Loss', 'WinRate', 'Kills', 'Deaths', 'Assists',
        'KDA', 'DPM', 'CS', 'CSPM', 'Gold', 'GPM', 'VisionScore'
    ]
    stats = stats[columns]

    stats.index = stats.index.set_names(['Team', 'Player', 'Champion', 'Opponent'])

    return stats


def main():
    """Various statistics"""
    sheet = Sheet(SHEET_ID)
    sheet.connect_sheet()

    print('Get leagues data ...')
    leagues = get_leagues(where='L.League_Short="WCS"')
    print(f'{len(leagues)} leagues')

    print('Get tournaments data ...')
    tournaments = get_tournaments(
        'L.League_Short="WCS" and T.Year=2022 and T.Region="international"'
    )
    tournaments = tournaments.sort_values(
        by=['Year', 'DateStart', 'Date']
    ).reset_index(drop=True)
    print(f'{len(tournaments)} tournaments')

    print('Get scoreboard games ...')
    scoreboard_games = get_scoreboard_games(
        f'T.OverviewPage="{tournaments["OverviewPage"][1]}"'
    )
    print(f'{len(scoreboard_games)} scoreboard games')

    print('Get scoreboard players ...')
    scoreboard_players = get_scoreboard_players_by_games(
        tournaments['OverviewPage'][1], scoreboard_games
    )
    print(f'{len(scoreboard_players)} scoreboard players')

    if scoreboard_games.shape[0] * 10 == scoreboard_players.shape[0]:
        print('Data is updated properly')
    else:
        print('Data is not updated properly')
        return None

    print('Get champions stats ...')
    champions_stats = get_champions_stats(scoreboard_games, scoreboard_players)
    champions_stats['As'] = champions_stats['As'].str.join(', ')
    sheet.update_sheet('champions', champions_stats)
    print('Updated champions stats on sheet')

    print('Get players stats ...')
    players_stats = get_players_stats(scoreboard_games, scoreboard_players)
    players_stats['Champs'] = players_stats['Champs'].str.join(', ')
    sheet.update_sheet('players', players_stats)
    print('Updated players stats on sheet')

    print('Get champions comp stats ...')
    positions = ['Bot', 'Support']
    comp_stats = get_champions_comp_stats(
        scoreboard_games, scoreboard_players, positions
    )
    comp_stats = comp_stats.sort_values(by='Games', ascending=False)
    sheet.update_sheet(f'{"_".join(positions)}_stats', comp_stats)
    print(f'Updated {" ".join(positions)} stats on sheet')

    print('Get champions vs stats ...')
    vs_stats = get_champions_vs_stats(scoreboard_games, scoreboard_players)
    vs_stats = vs_stats.sort_values(by='Games', ascending=False)
    vs_stats['As'] = vs_stats['As'].str.join(', ')
    sheet.update_sheet('vs_stats', vs_stats)
    print('Updated champions vs stats on sheet')

    print('Get player by champions stats ...')
    player_by_champions_stats = get_player_by_champions_stats(
        scoreboard_games, scoreboard_players
    )
    sheet.update_sheet('player_by_champions', player_by_champions_stats)
    print('Updated player by champions stats on sheet')

    print('Get teams stats ...')
    teams_stats = get_teams_stats(scoreboard_games)
    sheet.update_sheet('teams', teams_stats)
    print('Updated teams stats on sheet')

    print('Get player champions comp stats ...')
    positions = ['Bot', 'Support']
    player_comps_stats = get_player_champions_comp_stats(
        scoreboard_games, scoreboard_players, positions
    )
    player_comps_stats = player_comps_stats.sort_values(
        by=['Games', 'WinRate'], ascending=False
    )
    sheet.update_sheet(f'{"_".join(positions)}_player_stats', player_comps_stats)
    print(f'Updated {" ".join(positions)} stats on sheet')

    print('Get player by champions vs stats ...')
    player_champions_vs_stats = get_player_by_champions_vs_stats(
        scoreboard_games, scoreboard_players
    )
    player_champions_vs_stats = player_champions_vs_stats.sort_values(
        by=['Team', 'Player']
    )
    sheet.update_sheet('player_by_champions_vs', player_champions_vs_stats)
    print('Updated player by champions vs stats on sheet')


    _, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_xlim(teams_stats['GDPM'].min() - 25, teams_stats['GDPM'].max() + 25)
    ax.set_ylim(teams_stats['KPM'].min() - 0.01, teams_stats['KPM'].max() + 0.01)
    ax = sns.regplot(data=teams_stats, x='GDPM', y='KPM')
    y = np.linspace(0, 1.2)
    x = 100 / y
    sns.lineplot(x=x, y=y)
    for index, i in zip(teams_stats.index, range(teams_stats.shape[0])):
        row = teams_stats.iloc[i]
        plt.annotate(
            index, xy=(row['GDPM'], row['KPM']),
            xytext=(5, 5), textcoords='offset pixels'
        )
    plt.show()


if __name__ == '__main__':
    main()

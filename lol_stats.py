from functools import reduce
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google_sheet import Sheet
from lol_fandom import get_leagues

pd.set_option('display.max_columns', None)
with open('./sheet_id.txt', 'r') as f:
    SHEET_ID = f.read()


class LoLStats:
    def __init__(self, games, players, players_id):
        assert games.shape[0] > 0
        assert players.shape[0] > 0

        self.games = games
        self.players = players
        self.players_id = players_id

        self.games['CKPM'] = self.games[['Team1Kills', 'Team2Kills']].sum(axis=1)
        self.games['Team1GPM'] = self.games['Team1Gold']
        self.games['Team2GPM'] = self.games['Team2Gold']
        self.games['Team1GDPM'] = \
            (self.games['Team1Gold'] - self.games['Team2Gold'])
        self.games['Team2GDPM'] = \
            (self.games['Team2Gold'] - self.games['Team1Gold'])
        self.games['Team1KPM'] = self.games['Team1Kills']
        self.games['Team2KPM'] = self.games['Team2Kills']
        columns = [
            'CKPM', 'Team1GPM', 'Team2GPM', 'Team1GDPM', 'Team2GDPM',
            'Team1KPM', 'Team2KPM'
        ]
        self.games[columns] = self.games[columns].divide(
            self.games['Gamelength Number'], axis=0
        )

        self.merged = pd.merge(players, games, how='inner', on='GameId')

        self.merged['player_id'] = self.merged['Link'].transform(
            lambda x: self.players_id.loc[
                players_id['player'] == x, 'player_id'
            ].iloc[0]
        )
        self.merged['CSPM'] = self.merged['CS']
        self.merged['GPM'] = self.merged['Gold']
        self.merged['DPM'] = self.merged['DamageToChampions']
        columns = ['CSPM', 'GPM', 'DPM']
        self.merged[columns] = self.merged[columns].divide(
            self.merged['Gamelength Number'], axis=0
        )

    def get_teams_stats(self):
        teams_lst = self.games[['Team1', 'Team2']].unstack().unique()
        stats = pd.DataFrame(index=teams_lst)
        stats.index.set_names('Team', inplace=True)
        stats.sort_index(inplace=True)

        for team_name in stats.index:
            team1_games = self.games.loc[self.games['Team1'] == team_name]
            team2_games = self.games.loc[self.games['Team2'] == team_name]

            stats.loc[team_name, 'Games'] = \
                team1_games.shape[0] + team2_games.shape[0]
            stats.loc[team_name, 'Win'] = \
                team1_games.loc[
                    team1_games['Team1'] == team1_games['WinTeam']
                ].shape[0] + team2_games.loc[
                    team2_games['Team2'] == team2_games['WinTeam']
                ].shape[0]
            stats.loc[team_name, 'Loss'] = \
                team1_games.loc[
                    team1_games['Team1'] == team1_games['LossTeam']
                ].shape[0] + team2_games.loc[
                    team2_games['Team2'] == team2_games['LossTeam']
                ].shape[0]
            stats.loc[team_name, 'WinRate'] = \
                stats.loc[team_name, 'Win']
            stats.loc[team_name, 'KD'] = \
                (team1_games['Team1Kills'].sum() +
                team2_games['Team2Kills'].sum()) / \
                (team1_games['Team2Kills'].sum() +
                team2_games['Team1Kills'].sum())
            stats.loc[team_name, 'CKPM'] = \
                team1_games['CKPM'].sum() + team2_games['CKPM'].sum()
            stats.loc[team_name, 'GameDuration'] = \
                team1_games['Gamelength Number'].sum() + \
                team2_games['Gamelength Number'].sum()
            stats.loc[team_name, 'WinGameDuration'] = \
                team1_games.loc[
                    team1_games['Team1'] == team1_games['WinTeam'],
                    'Gamelength Number'
                ].sum() + team2_games.loc[
                    team2_games['Team2'] == team2_games['WinTeam'],
                    'Gamelength Number'
                ].sum()
            stats.loc[team_name, 'LossGameDuration'] = \
                team1_games.loc[
                    team1_games['Team1'] == team1_games['LossTeam'],
                    'Gamelength Number'
                ].sum() + team2_games.loc[
                    team2_games['Team2'] == team2_games['LossTeam'],
                    'Gamelength Number'
                ].sum()
            stats.loc[team_name, 'GPM'] = \
                team1_games['Team1GPM'].sum() + team2_games['Team2GPM'].sum()
            stats.loc[team_name, 'GDPM'] = \
                team1_games['Team1GDPM'].sum() + team2_games['Team2GDPM'].sum()
            stats.loc[team_name, 'KPM'] = \
                team1_games['Team1KPM'].sum() + team2_games['Team2KPM'].sum()
        columns = ['WinRate', 'CKPM', 'GameDuration', 'GPM', 'GDPM', 'KPM']
        stats[columns] = stats[columns].divide(
            stats['Games'], axis=0
        )
        stats['WinGameDuration'] = \
            stats['WinGameDuration'].divide(stats['Win'])
        stats['LossGameDuration'] = \
            stats['LossGameDuration'].divide(stats['Loss'])

        return stats

    def get_champions_stats(self):
        ban_list = self.games[['Team1Bans', 'Team2Bans']].unstack().str.split(',')
        ban_list = list(reduce(lambda x, y: x + y, ban_list))
        champion_names = list(set(
            list(self.players['Champion'].unique()) + ban_list
        ))
        while 'None' in ban_list:
            ban_list.remove('None')
        while 'None' in champion_names:
            champion_names.remove('None')

        stats = pd.DataFrame(index=champion_names)
        stats.index.set_names('Champion', inplace=True)
        stats.sort_index(inplace=True)

        for champ_name in stats.index:
            champions_df = self.merged.loc[self.merged['Champion'] == champ_name]

            stats.loc[champ_name, 'GamesPlayed'] = \
                champions_df.shape[0]
            stats.loc[champ_name, 'By'] = len(
                champions_df['player_id'].unique()
            )
            stats.loc[champ_name, 'Win'] = \
                champions_df.loc[champions_df['PlayerWin'] == 'Yes'].shape[0]
            stats.loc[champ_name, 'Loss'] = \
                champions_df.loc[champions_df['PlayerWin'] == 'No'].shape[0]
            stats.loc[champ_name, 'WinRate'] = \
                stats.loc[champ_name, 'Win']
            stats.loc[champ_name, 'Kills'] = \
                champions_df['Kills'].mean()
            stats.loc[champ_name, 'Deaths'] = \
                champions_df['Deaths'].mean()
            stats.loc[champ_name, 'Assists'] = \
                champions_df['Assists'].mean()
            stats.loc[champ_name, 'KDA'] = \
                champions_df[['Kills', 'Assists']].unstack().sum() / \
                champions_df['Deaths'].sum() \
                if champions_df['Deaths'].sum() > 0 else np.inf
            stats.loc[champ_name, 'CS'] = champions_df['CS'].mean()
            stats.loc[champ_name, 'CSPM'] = \
                champions_df['CSPM'].mean()
            stats.loc[champ_name, 'Gold'] = \
                champions_df['Gold'].mean()
            stats.loc[champ_name, 'GPM'] = champions_df['GPM'].mean()
            stats.loc[champ_name, 'Damage'] = \
                champions_df['DamageToChampions'].mean()
            stats.loc[champ_name, 'DPM'] = champions_df['DPM'].mean()

        stats['Ban'] = 0
        ban_counter = Counter(ban_list)
        for key, value in ban_counter.items():
            stats.loc[key, 'Ban'] = value
        stats['Games'] = stats[['GamesPlayed', 'Ban']].sum(axis=1)
        stats['BanPickRate'] = \
            stats['Games'].divide(self.games.shape[0])

        columns = [
            'Games', 'BanPickRate', 'Ban', 'GamesPlayed', 'By', 'Win', 'Loss',
            'WinRate', 'Kills', 'Deaths', 'Assists', 'KDA', 'CS', 'CSPM', 'Gold',
            'GPM', 'Damage', 'DPM'
        ]
        assert len(columns) == len(stats.columns)

        return stats[columns].sort_values(by='Games', ascending=False)

    def get_players_stats(self):
        idx = self.merged['player_id'].unique()
        stats = pd.DataFrame(index=idx)
        stats.index.set_names('id', inplace=True)
        stats.sort_index(inplace=True)

        for id in stats.index:
            players_df = self.merged.loc[self.merged['player_id'] == id]
            stats.loc[id, 'Player'] = players_df['Name'].iloc[-1]
            stats.loc[id, 'Team'] = players_df['Team'].iloc[-1]
            stats.loc[id, 'Games'] = players_df.shape[0]
            stats.loc[id, 'Win'] = players_df.loc[
                players_df['PlayerWin'] == 'Yes'
            ].shape[0]
            stats.loc[id, 'Loss'] = players_df.loc[
                players_df['PlayerWin'] == 'No'
            ].shape[0]
            stats.loc[id, 'WinRate'] = stats.loc[id, 'Win']
            stats.loc[id, 'Kills'] = players_df['Kills'].mean()
            stats.loc[id, 'Deaths'] = players_df['Deaths'].mean()
            stats.loc[id, 'Assists'] = players_df['Assists'].mean()
            stats.loc[id, 'KDA'] = stats.loc[
                id, ['Kills', 'Assists']
            ].sum()
            stats.loc[id, 'DPM'] = players_df['DPM'].mean()
            stats.loc[id, 'CS'] = players_df['CS'].mean()
            stats.loc[id, 'CSPM'] = players_df['CSPM'].mean()
            stats.loc[id, 'Gold'] = players_df['Gold'].mean()
            stats.loc[id, 'GPM'] = players_df['GPM'].mean()
            stats.loc[id, 'KP'] = players_df[['Kills', 'Assists']].unstack().sum() / \
                players_df['TeamKills'].sum() \
                if stats.loc[id, ['Kills', 'Assists']].sum() > 0 else 0
            stats.loc[id, 'KS'] = players_df['Kills'].sum() / \
                players_df['TeamKills'].sum() \
                if stats.loc[id, 'Kills'] > 0 else 0
            stats.loc[id, 'GS'] = \
                players_df['Gold'].sum() / players_df['TeamGold'].sum()
            stats.loc[id, 'ChampionsPlayed'] = \
                len(players_df['Champion'].unique())

        stats['WinRate'] = stats['WinRate'].divide(
            stats['Games']
        )
        stats['KDA'] = stats['KDA'].divide(stats['Deaths'])

        return stats.sort_values(by=['Team', 'Player'])

    def get_player_by_champion_stats(self):
        idx = pd.MultiIndex.from_tuples(
            list(set(self.merged[['player_id', 'Champion']].itertuples(index=False))),
            names=['id', 'Champion'],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for (id, champion) in stats.index:
            idx = (id, champion)
            players_df = self.merged.loc[
                (self.merged['player_id'] == id) &
                (self.merged['Champion'] == champion)
            ]

            stats.loc[idx, 'Player'] = players_df['Name'].iloc[-1]
            stats.loc[idx, 'Team'] = players_df['Team'].iloc[-1]
            stats.loc[idx, 'Games'] = players_df.shape[0]
            stats.loc[idx, 'Win'] = players_df.loc[
                players_df['PlayerWin'] == 'Yes'
            ].shape[0]
            stats.loc[idx, 'Loss'] = players_df.loc[
                players_df['PlayerWin'] == 'No'
            ].shape[0]
            stats.loc[idx, 'WinRate'] = stats.loc[idx, 'Win'] / \
                stats.loc[idx, 'Games'] if stats.loc[idx, 'Win'] > 0 else 0
            stats.loc[idx, 'Kills'] = players_df['Kills'].mean()
            stats.loc[idx, 'Deaths'] = players_df['Deaths'].mean()
            stats.loc[idx, 'Assists'] = players_df['Assists'].mean()
            stats.loc[idx, 'KDA'] = stats.loc[idx, ['Kills', 'Assists']].sum() / \
                stats.loc[idx, 'Deaths'] \
                if stats.loc[idx, 'Deaths'] > 0 else np.inf
            stats.loc[idx, 'DPM'] = players_df['DPM'].mean()
            stats.loc[idx, 'CSPM'] = players_df['CSPM'].mean()
            stats.loc[idx, 'GPM'] = players_df['GPM'].mean()
            stats.loc[idx, 'KP'] = \
                players_df[['Kills', 'Assists']].unstack().sum() / \
                players_df['TeamKills'].sum() \
                if stats.loc[idx, ['Kills', 'Assists']].sum() > 0 else 0
            stats.loc[idx, 'KS'] = players_df['Kills'].sum() / \
                players_df['TeamKills'].sum() \
                if stats.loc[idx, 'Kills'] > 0 else 0
            stats.loc[idx, 'GS'] = players_df['Gold'].sum() / \
                players_df['TeamGold'].sum()

        stats.reset_index(level='id', drop=True, inplace=True)
        stats.reset_index(inplace=True)
        stats.sort_values(by=['Team', 'Player', 'Champion'], inplace=True)

        columns = [
            'Player', 'Team', 'Champion', 'Games', 'Win', 'Loss', 'WinRate',
            'Kills', 'Deaths', 'Assists', 'KDA', 'DPM', 'CSPM', 'GPM',
            'KP', 'KS', 'GS'
        ]
        assert len(stats.columns) == len(columns)

        return stats[columns]

    def get_duo_champions_stats(self, role1='Bot', role2='Support'):
        roles = self.players['Role'].unique()
        assert role1 != role2 and role1 in roles and role2 in roles

        role1_df = self.merged.loc[self.merged['IngameRole'] == role1]
        role2_df = self.merged.loc[self.merged['IngameRole'] == role2]
        merged = pd.merge(role1_df, role2_df, how='inner', on=['GameId', 'Team'])
        assert merged.shape[0] == self.games.shape[0] * 2

        idx = pd.MultiIndex.from_tuples(
            list(set(merged[['Champion_x', 'Champion_y']].itertuples(index=False))),
            names=[role1, role2],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for (champ1, champ2) in stats.index:
            idx = (champ1, champ2)
            duo_df = merged.loc[
                (merged['Champion_x'] == champ1) &
                (merged['Champion_y'] == champ2)
            ]
            stats.loc[idx, 'Games'] = duo_df.shape[0]
            stats.loc[idx, 'By'] = len(set(
                duo_df[['player_id_x', 'player_id_y']].itertuples(index=False)
            ))
            stats.loc[idx, 'Win'] = \
                duo_df.loc[duo_df['PlayerWin_x'] == 'Yes'].shape[0]
            stats.loc[idx, 'Loss'] = \
                duo_df.loc[duo_df['PlayerWin_x'] == 'No'].shape[0]
        stats['WinRate'] = stats['Win'].divide(stats['Games'])

        return stats

    def get_vs_stats(self):
        merged = pd.merge(
            self.merged, self.merged, how='inner', on=['GameId', 'IngameRole']
        )
        merged = merged.loc[merged['Team_x'] != merged['Team_y']]

        idx = pd.MultiIndex.from_tuples(
            list(set(merged[
                ['Champion_x', 'Champion_y', 'IngameRole']
            ].itertuples(index=False))),
            names=['Champion1', 'Champion2', 'As']
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for (champ1, champ2, role) in stats.index:
            idx = (champ1, champ2, role)
            vs_df = merged.loc[
                (merged['Champion_x'] == champ1) &
                (merged['Champion_y'] == champ2) &
                (merged['IngameRole'] == role)
            ]
            stats.loc[idx, 'Games'] = vs_df.shape[0]
            stats.loc[idx, 'Win'] = \
                vs_df.loc[vs_df['PlayerWin_x'] == 'Yes'].shape[0]
            stats.loc[idx, 'Loss'] = \
                vs_df.loc[vs_df['PlayerWin_x'] == 'No'].shape[0]
        stats['WinRate'] = stats['Win'].divide(stats['Games'])

        return stats

    def get_player_by_champion_vs_stats(self):
        merged = pd.merge(
            self.merged, self.merged, how='inner', on=['GameId', 'IngameRole']
        )
        merged = merged.loc[merged['Team_x'] != merged['Team_y']]

        idx = pd.MultiIndex.from_tuples(
            list(set(merged[
                ['player_id_x', 'Champion_x', 'Champion_y']
            ].itertuples(index=False))),
            names=['id', 'Champion', 'Opponent'],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for (id, champ1, champ2) in stats.index:
            idx = (id, champ1, champ2)
            partial_df = merged.loc[
                (merged['player_id_x'] == id) &
                (merged['Champion_x'] == champ1) &
                (merged['Champion_y'] == champ2)
            ]
            stats.loc[idx, 'Player'] = partial_df['Name_x'].iloc[-1]
            stats.loc[idx, 'Team'] = partial_df['Team_x'].iloc[-1]
            stats.loc[idx, 'Games'] = partial_df.shape[0]
            stats.loc[idx, 'Win'] = \
                partial_df.loc[partial_df['PlayerWin_x'] == 'Yes'].shape[0]
            stats.loc[idx, 'Loss'] = \
                partial_df.loc[partial_df['PlayerWin_x'] == 'No'].shape[0]
        stats['WinRate'] = stats['Win'].divide(stats['Games'])

        stats.reset_index(level='id', drop=True, inplace=True)
        stats.reset_index(inplace=True)
        columns = [
            'Player', 'Team', 'Champion', 'Opponent', 'Games',
            'Win', 'Loss', 'WinRate'
        ]
        assert len(stats.columns) == len(columns)

        return stats[columns].sort_values(by=['Team', 'Player', 'Champion'])

    def get_duo_player_by_champion_stats(self, role1='Bot', role2='Support'):
        roles = self.players['Role'].unique()
        assert role1 != role2 and role1 in roles and role2 in roles

        role1_df = self.merged.loc[self.merged['IngameRole'] == role1]
        role2_df = self.merged.loc[self.merged['IngameRole'] == role2]
        merged = pd.merge(role1_df, role2_df, how='inner', on=['GameId', 'Team'])
        assert merged.shape[0] == self.games.shape[0] * 2

        idx = pd.MultiIndex.from_tuples(
            list(set(merged[
                ['player_id_x', 'player_id_y', 'Champion_x', 'Champion_y']
            ].itertuples(index=False))),
            names=['id1', 'id2', role1, role2],
        )
        stats = pd.DataFrame(index=idx).sort_index()

        for (id1, id2, champ1, champ2) in stats.index:
            idx = (id1, id2, champ1, champ2)
            partial_df = merged.loc[
                (merged['player_id_x'] == id1) &
                (merged['player_id_y'] == id2) &
                (merged['Champion_x'] == champ1) &
                (merged['Champion_y'] == champ2)
            ]

            stats.loc[idx, 'Player1'] = partial_df['Name_x'].iloc[-1]
            stats.loc[idx, 'Player2'] = partial_df['Name_y'].iloc[-1]
            stats.loc[idx, 'Games'] = partial_df.shape[0]
            stats.loc[idx, 'Win'] = partial_df.loc[partial_df['PlayerWin_x'] == 'Yes'].shape[0]
            stats.loc[idx, 'Loss'] = partial_df.loc[partial_df['PlayerWin_x'] == 'No'].shape[0]
        stats['WinRate'] = stats['Win'].divide(stats['Games'])

        stats.reset_index(level=['id1', 'id2'], drop=True, inplace=True)
        stats.reset_index(inplace=True)

        columns = [
            'Player1', 'Player2', role1, role2, 'Games',
            'Win', 'Loss', 'WinRate'
        ]
        assert len(columns) == len(stats.columns)

        return stats[columns].sort_values(by=['Player1', 'Player2', role1, role2])


def main():
    sheet = Sheet(SHEET_ID)
    sheet.connect_sheet()

    print('Get leagues ... ', end='')
    leagues = get_leagues(where=f'L.Level="Primary" and L.IsOfficial="Yes"')
    print('Complete')
    print('Get tournaments ... ', end='')
    tournaments = pd.read_csv('./csv/tournaments/2023_tournaments.csv')
    print('Complete')

    league = leagues.loc[
        leagues['League Short'] == 'LCK', 'League'
    ].iloc[0]
    page = tournaments.loc[
        tournaments['League'] == league, 'OverviewPage'
    ].iloc[0]

    print('Read scoreboard games ... ', end='')
    scoreboard_games = \
        pd.read_csv('./csv/scoreboard_games/2023_scoreboard_games.csv')
    scoreboard_games = \
        scoreboard_games.loc[scoreboard_games['OverviewPage'] == page]
    scoreboard_games = scoreboard_games.sort_values(
        by='DateTime UTC'
    ).reset_index(drop=True)
    print('Complete')

    print('Read scoreboard players ... ', end='')
    scoreboard_players = pd.read_csv(
        './csv/scoreboard_players/2023_scoreboard_players.csv'
    )
    scoreboard_players = scoreboard_players.loc[
        scoreboard_players['OverviewPage'] == page
    ]
    scoreboard_players = scoreboard_players.sort_values(
        by='DateTime UTC'
    ).reset_index(drop=True)
    print('Complete')


    # scoreboard_games = scoreboard_games.loc[(scoreboard_games['Patch'] == '13.5')]
    # scoreboard_players = scoreboard_players.loc[
    #     scoreboard_players['GameId'].isin(scoreboard_games['GameId'].unique())
    # ]

    print(
        f'scoreboard games: {scoreboard_games.shape[0]} | ' +
        f'scoreboard players: {scoreboard_players.shape[0]}'
    )
    assert scoreboard_games.shape[0] * 10 == scoreboard_players.shape[0]
    print(f'\n{page}')

    players_id = pd.read_csv('./csv/players_id.csv')

    stats = LoLStats(scoreboard_games, scoreboard_players, players_id)
    print('Team stats ... ', end='')
    teams_stats = stats.get_teams_stats()
    teams_stats.to_csv('./csv/stats/teams.csv')
    sheet.update_sheet('teams', teams_stats)
    print('Complete')

    print('Champion stats ... ', end='')
    champions_stats = stats.get_champions_stats()
    champions_stats.to_csv('./csv/stats/champions.csv')
    sheet.update_sheet('champions', champions_stats)
    print('Complete')

    print('Players Stats ... ', end='')
    players_stats = stats.get_players_stats()
    players_stats.to_csv('./csv/stats/players.csv', index=False)
    sheet.update_sheet('players', players_stats, index=False)
    print('Complete')

    print('Player by Champion Stats ... ', end='')
    player_by_champion_stats = stats.get_player_by_champion_stats()
    player_by_champion_stats.to_csv(
        './csv/stats/player_by_champion.csv', index=False
    )
    sheet.update_sheet(
        'player_by_champion', player_by_champion_stats, index=False
    )
    print('Complete')

    print('Duo Stats ... ', end='')
    duo_stats = stats.get_duo_champions_stats()
    duo_stats.to_csv('./csv/stats/duo.csv')
    sheet.update_sheet('duo', duo_stats)
    print('Complete')

    print('Vs Stats ... ', end='')
    vs_stats = stats.get_vs_stats()
    vs_stats.to_csv('./csv/stats/vs.csv')
    sheet.update_sheet('vs', vs_stats)
    print('Complete')

    print('Player by champion vs stats ... ', end='')
    player_by_champion_vs_stats = stats.get_player_by_champion_vs_stats()
    player_by_champion_vs_stats.to_csv(
        './csv/stats/player_by_champion_vs.csv', index=False
    )
    sheet.update_sheet(
        'player_by_champion_vs', player_by_champion_vs_stats, index=False
    )
    print('Complete')

    print('Duo player by champion stats ... ', end='')
    duo_player_by_champion_stats = stats.get_duo_player_by_champion_stats()
    duo_player_by_champion_stats.to_csv(
        './csv/stats/duo_player_by_champion.csv', index=False
    )
    sheet.update_sheet(
        'duo_player_by_champion', duo_player_by_champion_stats, index=False
    )
    print('Complete')


if __name__ == '__main__':
    main()
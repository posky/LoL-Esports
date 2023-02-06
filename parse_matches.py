from collections import Counter
import datetime

import pandas as pd
import numpy as np

from lol_fandom import SITE, set_default_delay
from lol_fandom import get_leagues, get_tournaments
from lol_fandom import get_scoreboard_games, get_scoreboard_players
from lol_fandom import get_tournament_rosters
from lol_fandom import from_response

pd.set_option('display.max_columns', None)
# set_default_delay(0.5)


def parse_tournaments(start=2011, end=datetime.datetime.now().year):
    print('=========== Tournaments ===========')
    leagues = get_leagues(where=f'L.Level="Primary" and L.IsOfficial="Yes"')
    for year in range(start, end + 1):
        print(f'{year} tournaments')
        tournaments = pd.DataFrame()
        for league in leagues['League Short']:
            t = get_tournaments(where=f'L.League_Short="{league}" and T.Year={year}')
            print(f'\t{league} - {t.shape[0]}')
            tournaments = pd.concat([tournaments, t], ignore_index=True)
        tournaments = tournaments.sort_values(
            by=['Year', 'DateStart', 'Date']
        ).reset_index(drop=True)

        tournaments.to_csv(f'./csv/tournaments/{year}_tournaments.csv', index=False)
        print(f'{year} tournaments - {tournaments.shape}')

def parse_scoreboard_games(start=2011, end=datetime.datetime.now().year):
    print('=========== Scoreboard Games ===========')
    leagues = get_leagues(where=f'L.Level="Primary" and L.IsOfficial="Yes"')
    for year in range(start, end + 1):
        tournaments = pd.read_csv(f'./csv/tournaments/{year}_tournaments.csv')
        print(f'{year} - tournament {tournaments.shape}')
        scoreboard_games = pd.DataFrame()
        for page in tournaments['OverviewPage']:
            sg = get_scoreboard_games(where=f'T.OverviewPage="{page}"')
            if sg is None:
                print(f'\t{page} - drop')
                tournaments.drop(
                    tournaments.loc[tournaments['OverviewPage'] == page].index,
                    inplace=True
                )
                continue
            league = tournaments.loc[
                tournaments['OverviewPage'] == page, 'League'
            ].iloc[0]
            league = leagues.loc[leagues['League'] == league, 'League Short'].iloc[0]
            sg['League'] = league
            print(f'\t{page} - {sg.shape[0]}')
            scoreboard_games = pd.concat([scoreboard_games, sg], ignore_index=True)
        scoreboard_games = scoreboard_games.sort_values(
            by='DateTime UTC'
        ).reset_index(drop=True)
        scoreboard_games.to_csv(
            f'./csv/scoreboard_games/{year}_scoreboard_games.csv', index=False
        )
        print(f'{year} scoreboard_games {scoreboard_games.shape}')
        tournaments.to_csv(f'./csv/tournaments/{year}_tournaments.csv', index=False)
        print(f'{year} tournaments {tournaments.shape}')

def parse_scoreboard_players(start=2011, end=datetime.datetime.now().year):
    print('=========== Scoreboard Players ===========')
    for year in range(start, end + 1):
        tournaments = pd.read_csv(f'./csv/tournaments/{year}_tournaments.csv')
        scoreboard_games = pd.read_csv(f'./csv/scoreboard_games/{year}_scoreboard_games.csv')
        print(f'{year} - tournament {tournaments.shape}')
        print(f'{year} - scoreboard games {scoreboard_games.shape}')
        scoreboard_players = pd.DataFrame()
        for page in tournaments['OverviewPage']:
            print(f'\t{page}', end='')
            teams = scoreboard_games.loc[
                scoreboard_games['OverviewPage'] == page, ['Team1', 'Team2']
            ].unstack().unique()
            len_sp = 0
            for i, team in enumerate(teams, start=1):
                print(f'\r\t{page} - ({i}/{len(teams)})', end='')
                sp = get_scoreboard_players(
                    where=f'T.OverviewPage="{page}" and SP.Team="{team}"'
                )
                len_sp += sp.shape[0]
                scoreboard_players = pd.concat([scoreboard_players, sp])
            len_sg = scoreboard_games.loc[scoreboard_games['OverviewPage'] == page].shape[0]
            print(f'\n\t\tscoreboard games - {len_sg} | scoreboard players - {len_sp} | {len_sg * 10 == len_sp}')
        scoreboard_players = scoreboard_players.sort_values(
            by=['DateTime UTC', 'Team', 'Role Number']
        ).reset_index(drop=True)
        scoreboard_players.to_csv(
            f'./csv/scoreboard_players/{year}_scoreboard_players.csv',
            index=False
        )
        print(f'{year} scoreboard_players {scoreboard_players.shape}')

def parse_tournament_rosters(start=2011, end=datetime.datetime.now().year):
    print('=========== Tournament Rosters ===========')
    for year in range(start, end + 1):
        tournaments = pd.read_csv(f'./csv/tournaments/{year}_tournaments.csv')
        print(f'{year} - tournament {tournaments.shape}')
        tournament_rosters = pd.DataFrame()
        for page in tournaments['OverviewPage']:
            tr = get_tournament_rosters(where=f'T.OverviewPage="{page}"')
            print(f'\t{page} - {tr.shape[0]}')
            tournament_rosters = pd.concat([tournament_rosters, tr], ignore_index=True)
        tournament_rosters.to_csv(
            f'./csv/tournament_rosters/{year}_tournament_rosters.csv',
            index=False
        )
        print(f'{year} tournament rosters {tournament_rosters.shape}')

def main():
    parse_tournaments(start=2023)
    parse_scoreboard_games(start=2023)
    parse_scoreboard_players(start=2023)
    parse_tournament_rosters(start=2023)


if __name__ == '__main__':
    main()
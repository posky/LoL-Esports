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
import pandas as pd
import numpy as np

from lol_fandom import get_leagues, get_tournaments
from lol_fandom import get_scoreboard_games, get_scoreboard_players

pd.set_option('display.max_columns', None)

# %%
OVERVIEWS = {
    # Worlds
    'Season 1 World Championship': 'Season 1 World Championship',
    'Season 2 World Championship': 'Season 2 World Championship',
    'Season 3 World Championship': 'Season 3 World Championship',
    '2014 Season World Championship': '2014 Season World Championship',
    '2015 Season World Championship': '2015 Season World Championship',
    '2016 Season World Championship': '2016 Season World Championship',
    '2017 Season World Championship/Play-In': '2017 Season World Championship',
    '2017 Season World Championship/Main Event': '2017 Season World Championship',
    '2018 Season World Championship/Play-In': '2018 Season World Championship',
    '2018 Season World Championship/Main Event': '2018 Season World Championship',
    '2019 Season World Championship/Play-In': '2019 Season World Championship',
    '2019 Season World Championship/Main Event': '2019 Season World Championship',
    '2020 Season World Championship/Play-In': '2020 Season World Championship',
    '2020 Season World Championship/Main Event': '2020 Season World Championship',
    '2021 Season World Championship/Play-In': '2021 Season World Championship',
    '2021 Season World Championship/Main Event': '2021 Season World Championship',
    # MSI
    '2015 Mid-Season Invitational': '2015 Mid-Season Invitational',
    '2016 Mid-Season Invitational': '2016 Mid-Season Invitational',
    '2017 Mid-Season Invitational/Play-In': '2017 Mid-Season Invitational',
    '2017 Mid-Season Invitational/Main Event': '2017 Mid-Season Invitational',
    '2018 Mid-Season Invitational/Play-In': '2018 Mid-Season Invitational',
    '2018 Mid-Season Invitational/Main Event': '2018 Mid-Season Invitational',
    '2019 Mid-Season Invitational/Play-In': '2019 Mid-Season Invitational',
    '2019 Mid-Season Invitational/Main Event': '2019 Mid-Season Invitational',
    '2021 Mid-Season Invitational': '2021 Mid-Season Invitational',
    '2022 Mid-Season Invitational': '2022 Mid-Season Invitational',
    # LCK
    'Champions/2012 Season/Spring': 'Champions/2012 Season/Spring',
    'Champions/2012 Season/Summer': 'Champions/2012 Season/Summer',
    'Champions/2013 Season/Winter': 'Champions/2013 Season/Winter',
    'Champions/2013 Season/Spring': 'Champions/2013 Season/Spring',
    'Champions/2013 Season/Summer': 'Champions/2013 Season/Summer',
    'Champions/2014 Season/Winter Season': 'Champions/2014 Season/Winter',
    'Champions/2014 Season/Spring Season': 'Champions/2014 Season/Spring',
    'Champions/2014 Season/Summer Season': 'Champions/2014 Season/Summer',
    'Champions/2015 Season/Spring Season': 'Champions/2015 Season/Spring',
    'Champions/2015 Season/Spring Playoffs': 'Champions/2015 Season/Spring',
    'Champions/2015 Season/Summer Season': 'Champions/2015 Season/Summer',
    'Champions/2015 Season/Summer Playoffs': 'Champions/2015 Season/Summer',
    'LCK/2016 Season/Spring Season': 'LCK/2016 Season/Spring',
    'LCK/2016 Season/Spring Playoffs': 'LCK/2016 Season/Spring',
    'LCK/2016 Season/Summer Season': 'LCK/2016 Season/Summer',
    'LCK/2016 Season/Summer Playoffs': 'LCK/2016 Season/Summer',
    'LCK/2017 Season/Spring Season': 'LCK/2017 Season/Spring',
    'LCK/2017 Season/Spring Playoffs': 'LCK/2017 Season/Spring',
    'LCK/2017 Season/Summer Season': 'LCK/2017 Season/Summer',
    'LCK/2017 Season/Summer Playoffs': 'LCK/2017 Season/Summer',
    'LCK/2018 Season/Spring Season': 'LCK/2018 Season/Spring',
    'LCK/2018 Season/Spring Playoffs': 'LCK/2018 Season/Spring',
    'LCK/2018 Season/Summer Season': 'LCK/2018 Season/Summer',
    'LCK/2018 Season/Summer Playoffs': 'LCK/2018 Season/Summer',
    'LCK/2019 Season/Spring Season': 'LCK/2019 Season/Spring',
    'LCK/2019 Season/Spring Playoffs': 'LCK/2019 Season/Spring',
    'LCK/2019 Season/Summer Season': 'LCK/2019 Season/Summer',
    'LCK/2019 Season/Summer Playoffs': 'LCK/2019 Season/Summer',
    'LCK/2020 Season/Spring Season': 'LCK/2020 Season/Spring',
    'LCK/2020 Season/Spring Playoffs': 'LCK/2020 Season/Spring',
    'LCK/2020 Season/Summer Season': 'LCK/2020 Season/Summer',
    'LCK/2020 Season/Summer Playoffs': 'LCK/2020 Season/Summer',
    'LCK/2021 Season/Spring Season': 'LCK/2021 Season/Spring',
    'LCK/2021 Season/Spring Playoffs': 'LCK/2021 Season/Spring',
    'LCK/2021 Season/Summer Season': 'LCK/2021 Season/Summer',
    'LCK/2021 Season/Summer Playoffs': 'LCK/2021 Season/Summer',
    'LCK/2022 Season/Spring Season': 'LCK/2022 Season/Spring',
    'LCK/2022 Season/Spring Playoffs': 'LCK/2022 Season/Spring',
    'LCK/2022 Season/Summer Season': 'LCK/2022 Season/Summer',
    'LCK/2022 Season/Summer Playoffs': 'LCK/2022 Season/Summer',
    # LPL
    'LPL/2013 Season/Spring Season': 'LPL/2013 Season/Spring',
    'LPL/2013 Season/Spring Playoffs': 'LPL/2013 Season/Spring',
    'LPL/2013 Season/Summer Season': 'LPL/2013 Season/Summer',
    'LPL/2013 Season/Summer Playoffs': 'LPL/2013 Season/Summer',
    'LPL/2014 Season/Spring Season': 'LPL/2014 Season/Spring',
    'LPL/2014 Season/Spring Playoffs': 'LPL/2014 Season/Spring',
    'LPL/2014 Season/Summer Season': 'LPL/2014 Season/Summer',
    'LPL/2014 Season/Summer Playoffs': 'LPL/2014 Season/Summer',
    'LPL/2015 Season/Spring Season': 'LPL/2015 Season/Spring',
    'LPL/2015 Season/Spring Playoffs': 'LPL/2015 Season/Spring',
    'LPL/2015 Season/Summer Season': 'LPL/2015 Season/Summer',
    'LPL/2015 Season/Summer Playoffs': 'LPL/2015 Season/Summer',
    'LPL/2016 Season/Spring Season': 'LPL/2016 Season/Spring',
    'LPL/2016 Season/Spring Playoffs': 'LPL/2016 Season/Spring',
    'LPL/2016 Season/Summer Season': 'LPL/2016 Season/Summer',
    'LPL/2016 Season/Summer Playoffs': 'LPL/2016 Season/Summer',
    'LPL/2017 Season/Spring Season': 'LPL/2017 Season/Spring',
    'LPL/2017 Season/Spring Playoffs': 'LPL/2017 Season/Spring',
    'LPL/2017 Season/Summer Season': 'LPL/2017 Season/Summer',
    'LPL/2017 Season/Summer Playoffs': 'LPL/2017 Season/Summer',
    'LPL/2018 Season/Spring Season': 'LPL/2018 Season/Spring',
    'LPL/2018 Season/Spring Playoffs': 'LPL/2018 Season/Spring',
    'LPL/2018 Season/Summer Season': 'LPL/2018 Season/Summer',
    'LPL/2018 Season/Summer Playoffs': 'LPL/2018 Season/Summer',
    'LPL/2019 Season/Spring Season': 'LPL/2019 Season/Spring',
    'LPL/2019 Season/Spring Playoffs': 'LPL/2019 Season/Spring',
    'LPL/2019 Season/Summer Season': 'LPL/2019 Season/Summer',
    'LPL/2019 Season/Summer Playoffs': 'LPL/2019 Season/Summer',
    'LPL/2020 Season/Spring Season': 'LPL/2020 Season/Spring',
    'LPL/2020 Season/Spring Playoffs': 'LPL/2020 Season/Spring',
    'LPL/2020 Season/Summer Season': 'LPL/2020 Season/Summer',
    'LPL/2020 Season/Summer Playoffs': 'LPL/2020 Season/Summer',
    'LPL/2021 Season/Spring Season': 'LPL/2021 Season/Spring',
    'LPL/2021 Season/Spring Playoffs': 'LPL/2021 Season/Spring',
    'LPL/2021 Season/Summer Season': 'LPL/2021 Season/Summer',
    'LPL/2021 Season/Summer Playoffs': 'LPL/2021 Season/Summer',
    'LPL/2022 Season/Spring Season': 'LPL/2022 Season/Spring',
    'LPL/2022 Season/Spring Playoffs': 'LPL/2022 Season/Spring',
    'LPL/2022 Season/Summer Season': 'LPL/2022 Season/Summer',
    'LPL/2022 Season/Summer Playoffs': 'LPL/2022 Season/Summer',
    # LEC
    'EU LCS/Season 3/Spring Season': 'EU LCS/Season 3/Spring',
    'EU LCS/Season 3/Spring Playoffs': 'EU LCS/Season 3/Spring',
    'EU LCS/Season 3/Summer Season': 'EU LCS/Season 3/Summer',
    'EU LCS/Season 3/Summer Playoffs': 'EU LCS/Season 3/Summer',
    'EU LCS/2014 Season/Spring Season': 'EU LCS/2014 Season/Spring',
    'EU LCS/2014 Season/Spring Playoffs': 'EU LCS/2014 Season/Spring',
    'EU LCS/2014 Season/Summer Season': 'EU LCS/2014 Season/Summer',
    'EU LCS/2014 Season/Summer Playoffs': 'EU LCS/2014 Season/Summer',
    'EU LCS/2015 Season/Spring Season': 'EU LCS/2015 Season/Spring',
    'EU LCS/2015 Season/Spring Playoffs': 'EU LCS/2015 Season/Spring',
    'EU LCS/2015 Season/Summer Season': 'EU LCS/2015 Season/Summer',
    'EU LCS/2015 Season/Summer Playoffs': 'EU LCS/2015 Season/Summer',
    'EU LCS/2016 Season/Spring Season': 'EU LCS/2016 Season/Spring',
    'EU LCS/2016 Season/Spring Playoffs': 'EU LCS/2016 Season/Spring',
    'EU LCS/2016 Season/Summer Season': 'EU LCS/2016 Season/Summer',
    'EU LCS/2016 Season/Summer Playoffs': 'EU LCS/2016 Season/Summer',
    'EU LCS/2017 Season/Spring Season': 'EU LCS/2017 Season/Spring',
    'EU LCS/2017 Season/Spring Playoffs': 'EU LCS/2017 Season/Spring',
    'EU LCS/2017 Season/Summer Season': 'EU LCS/2017 Season/Summer',
    'EU LCS/2017 Season/Summer Playoffs': 'EU LCS/2017 Season/Summer',
    'EU LCS/2018 Season/Spring Season': 'EU LCS/2018 Season/Spring',
    'EU LCS/2018 Season/Spring Playoffs': 'EU LCS/2018 Season/Spring',
    'EU LCS/2018 Season/Summer Season': 'EU LCS/2018 Season/Summer',
    'EU LCS/2018 Season/Summer Playoffs': 'EU LCS/2018 Season/Summer',
    'LEC/2019 Season/Spring Season': 'LEC/2019 Season/Spring',
    'LEC/2019 Season/Spring Playoffs': 'LEC/2019 Season/Spring',
    'LEC/2019 Season/Summer Season': 'LEC/2019 Season/Summer',
    'LEC/2019 Season/Summer Playoffs': 'LEC/2019 Season/Summer',
    'LEC/2020 Season/Spring Season': 'LEC/2020 Season/Spring',
    'LEC/2020 Season/Spring Playoffs': 'LEC/2020 Season/Spring',
    'LEC/2020 Season/Summer Season': 'LEC/2020 Season/Summer',
    'LEC/2020 Season/Summer Playoffs': 'LEC/2020 Season/Summer',
    'LEC/2021 Season/Spring Season': 'LEC/2021 Season/Spring',
    'LEC/2021 Season/Spring Playoffs': 'LEC/2021 Season/Spring',
    'LEC/2021 Season/Summer Season': 'LEC/2021 Season/Summer',
    'LEC/2021 Season/Summer Playoffs': 'LEC/2021 Season/Summer',
    'LEC/2022 Season/Spring Season': 'LEC/2022 Season/Spring',
    'LEC/2022 Season/Spring Playoffs': 'LEC/2022 Season/Spring',
    'LEC/2022 Season/Summer Season': 'LEC/2022 Season/Summer',
    'LEC/2022 Season/Summer Playoffs': 'LEC/2022 Season/Summer',
    # LCS
    'NA LCS/Season 3/Spring Season': 'NA LCS/Season 3/Spring',
    'NA LCS/Season 3/Spring Playoffs': 'NA LCS/Season 3/Spring',
    'NA LCS/Season 3/Summer Season': 'NA LCS/Season 3/Summer',
    'NA LCS/Season 3/Summer Playoffs': 'NA LCS/Season 3/Summer',
    'NA LCS/2014 Season/Spring Season': 'NA LCS/2014 Season/Spring',
    'NA LCS/2014 Season/Spring Playoffs': 'NA LCS/2014 Season/Spring',
    'NA LCS/2014 Season/Summer Season': 'NA LCS/2014 Season/Summer',
    'NA LCS/2014 Season/Summer Playoffs': 'NA LCS/2014 Season/Summer',
    'NA LCS/2015 Season/Spring Season': 'NA LCS/2015 Season/Spring',
    'NA LCS/2015 Season/Spring Playoffs': 'NA LCS/2015 Season/Spring',
    'NA LCS/2015 Season/Summer Season': 'NA LCS/2015 Season/Summer',
    'NA LCS/2015 Season/Summer Playoffs': 'NA LCS/2015 Season/Summer',
    'NA LCS/2016 Season/Spring Season': 'NA LCS/2016 Season/Spring',
    'NA LCS/2016 Season/Spring Playoffs': 'NA LCS/2016 Season/Spring',
    'NA LCS/2016 Season/Summer Season': 'NA LCS/2016 Season/Summer',
    'NA LCS/2016 Season/Summer Playoffs': 'NA LCS/2016 Season/Summer',
    'NA LCS/2017 Season/Spring Season': 'NA LCS/2017 Season/Spring',
    'NA LCS/2017 Season/Spring Playoffs': 'NA LCS/2017 Season/Spring',
    'NA LCS/2017 Season/Summer Season': 'NA LCS/2017 Season/Summer',
    'NA LCS/2017 Season/Summer Playoffs': 'NA LCS/2017 Season/Summer',
    'NA LCS/2018 Season/Spring Season': 'NA LCS/2018 Season/Spring',
    'NA LCS/2018 Season/Spring Playoffs': 'NA LCS/2018 Season/Spring',
    'NA LCS/2018 Season/Summer Season': 'NA LCS/2018 Season/Summer',
    'NA LCS/2018 Season/Summer Playoffs': 'NA LCS/2018 Season/Summer',
    'LCS/2019 Season/Spring Season': 'LCS/2019 Season/Spring',
    'LCS/2019 Season/Spring Playoffs': 'LCS/2019 Season/Spring',
    'LCS/2019 Season/Summer Season': 'LCS/2019 Season/Summer',
    'LCS/2019 Season/Summer Playoffs': 'LCS/2019 Season/Summer',
    'LCS/2020 Season/Spring Season': 'LCS/2020 Season/Spring',
    'LCS/2020 Season/Spring Playoffs': 'LCS/2020 Season/Spring',
    'LCS/2020 Season/Summer Season': 'LCS/2020 Season/Summer',
    'LCS/2020 Season/Summer Playoffs': 'LCS/2020 Season/Summer',
    'LCS/2021 Season/Spring Season': 'LCS/2021 Season/Spring',
    'LCS/2021 Season/Mid-Season Showdown': 'LCS/2021 Season/Spring',
    'LCS/2021 Season/Summer Season': 'LCS/2021 Season/Summer',
    'LCS/2021 Season/Championship': 'LCS/2021 Season/Summer',
    'LCS/2022 Season/Spring Season': 'LCS/2022 Season/Spring',
    'LCS/2022 Season/Spring Playoffs': 'LCS/2022 Season/Spring',
    'LCS/2022 Season/Summer Season': 'LCS/2022 Season/Summer',
    'LCS/2022 Season/Championship': 'LCS/2022 Season/Summer',
}


# %%
def change_data(games):
    df = pd.DataFrame()
    columns = [
        'OverviewPage', 'Team', 'Opponent', 'Win', 'Loss', 'DateTime UTC',
        'Gamelength', 'Gamelength Number', 'Gold', 'Kills'
    ]
    for i, row in enumerate(games.itertuples()):
        data = {
            'OverviewPage': [row.OverviewPage] * 2,
            'Team': [row.Team1, row.Team2],
            'Opponent': [row.Team2, row.Team1],
            'Win': [],
            'Loss': [],
            'DateTime UTC': [row._6] * 2,
            'Gamelength': [row.Gamelength] * 2,
            'Gamelength Number': [row._11] * 2,
            'TeamBans': [row.Team1Bans, row.Team2Bans],
            'OpponentBans': [row.Team2Bans, row.Team1Bans],
            'TeamPicks': [row.Team1Picks, row.Team2Picks],
            'OpponentPicks': [row.Team2Picks, row.Team1Picks],
            'TeamPlayers': [row.Team1Players, row.Team2Players],
            'OpponentPlayers': [row.Team2Players, row.Team1Players],
            'TeamDragons': [row.Team1Dragons, row.Team2Dragons],
            'OpponentDragons': [row.Team2Dragons, row.Team1Dragons],
            'TeamBarons': [row.Team1Barons, row.Team2Barons],
            'OpponentBarons': [row.Team2Barons, row.Team1Barons],
            'TeamTowers': [row.Team1Towers, row.Team2Towers],
            'OpponentTowers': [row.Team2Towers, row.Team1Towers],
            'TeamGold': [row.Team1Gold, row.Team2Gold],
            'OpponentGold': [row.Team2Gold, row.Team1Gold],
            'TeamKills': [row.Team1Kills, row.Team2Kills],
            'OpponentKills': [row.Team2Kills, row.Team1Kills],
            'TeamRiftHeralds': [row.Team1RiftHeralds, row.Team2RiftHeralds],
            'OpponentHeralds': [row.Team2RiftHeralds, row.Team1RiftHeralds],
            'TeamInhibitors': [row.Team1Inhibitors, row.Team2Inhibitors],
            'OpponentInhibitors': [row.Team2Inhibitors, row.Team1Inhibitors],
        }
        win = 1 if row.Team1 == row.WinTeam else 0
        data['Win'].extend([win, 1 - win])
        data['Loss'].extend([1 - win, win])

        df = pd.concat([df, pd.DataFrame(data)])
    return df.reset_index(drop=True)

def get_team_stats(games):
    grouped = games.groupby('Team')
    df = pd.DataFrame()
    df['Win'] = grouped['Win'].sum()
    df['Loss'] = grouped['Loss'].sum()
    df['Gold'] = grouped['Gold'].mean()
    df['Kills'] = grouped['Kills'].mean()
    df['Deaths'] = grouped['Deaths'].mean()
    df['Gamelength Number'] = grouped['Gamelength Number'].mean()
    df['Win Gamelength Number'] = \
        grouped[['Win', 'Gamelength Number']].apply(
            lambda x: (x['Win'] * x['Gamelength Number']).sum()
        ) / df['Win']
    df['Loss Gamelength Number'] = \
        grouped[['Loss', 'Gamelength Number']].apply(
            lambda x: (x['Loss'] * x['Gamelength Number']).sum()
        ) / df['Loss']
    df['Games'] = grouped['Win'].count()
    df['WinRate'] = df['Win'] / df['Games']
    df['KPM'] = grouped['KPM'].mean()
    df['KDPM'] = grouped['KDPM'].mean()
    df['GPM'] = grouped['GPM'].mean()
    df['GDPM'] = grouped['GDPM'].mean()
    df['OverviewPage'] = grouped['OverviewPage'].last()
    df['Winner'] = 0

    columns = [
        'OverviewPage', 'Games', 'Win', 'Loss', 'WinRate', 'Gamelength Number',
        'Win Gamelength Number', 'Loss Gamelength Number', 'Gold', 'Kills',
        'GPM', 'GDPM', 'KPM', 'KDPM', 'Winner'
    ]

    return df[columns]


# %%
LEAGUES = [
    'WCS', 'MSI', 'LTC', 'LCK', 'LPL', 'EU LCS', 'LEC', 'NA LCS', 'LCS'
]

# %%
leagues = pd.DataFrame()
for league in LEAGUES:
    temp = get_leagues(where=f'L.League_Short="{league}"')
    leagues = pd.concat([leagues, temp])
leagues = leagues.reset_index(drop=True)
leagues


# %%
def filter_tournaments(tournaments):
    for row in tournaments.itertuples():
        if row.OverviewPage not in OVERVIEWS.keys():
            tournaments = tournaments.drop(index=row.Index)
    return tournaments.reset_index(drop=True)

tournaments = pd.DataFrame()
for league in leagues['League Short']:
    if league in ['WCS']:
        tournament = get_tournaments(
            where=f'L.League_Short="{league}" and T.Region="International"'
        )
    elif league == 'MSI':
        tournament = get_tournaments(
            where=f'L.League_Short="{league}"'
        )
    else:
        tournament = get_tournaments(
            where=f'L.League_Short="{league}" and T.IsQualifier=0'
        )
    tournaments = pd.concat([tournaments, tournament])
    
tournaments = tournaments.sort_values(
    by=['Year', 'DateStart', 'Date']
).reset_index(drop=True)
tournaments = filter_tournaments(tournaments)
tournaments


# %%
def change_overviews(games):
    games['OverviewPage'] = games['OverviewPage'].apply(lambda x: OVERVIEWS[x])
    return games

scoreboard_games = pd.DataFrame()
for overview in tournaments['OverviewPage']:
    temp = get_scoreboard_games(where=f'T.OverviewPage="{overview}"')
    if temp is None:
        print(f'{overview} is None')
        break
    scoreboard_games = pd.concat([scoreboard_games, temp])
scoreboard_games = scoreboard_games.reset_index(drop=True)
scoreboard_games = change_overviews(scoreboard_games)
scoreboard_games

# %%
games = change_data(scoreboard_games)
print(scoreboard_games.shape[0] * 2 == games.shape[0])
games

# %%
games.to_csv('./csv/major_matches.csv', index=False)

# %%
WINNERS = {
    # Worlds
    'Season 1 World Championship': 'Fnatic',
    'Season 2 World Championship': 'Taipei Assassins',
    'Season 3 World Championship': 'SK Telecom T1',
    '2014 Season World Championship': 'Samsung White',
    '2015 Season World Championship': 'SK Telecom T1',
    '2016 Season World Championship': 'SK Telecom T1',
    '2017 Season World Championship': 'Samsung Galaxy',
    '2018 Season World Championship': 'Invictus Gaming',
    '2019 Season World Championship': 'FunPlus Phoenix',
    '2020 Season World Championship': 'DAMWON Gaming',
    '2021 Season World Championship': 'EDward Gaming',
    # MSI
    '2015 Mid-Season Invitational': 'EDward Gaming',
    '2016 Mid-Season Invitational': 'SK Telecom T1',
    '2017 Mid-Season Invitational': 'SK Telecom T1',
    '2018 Mid-Season Invitational': 'Royal Never Give Up',
    '2019 Mid-Season Invitational': 'G2 Esports',
    '2021 Mid-Season Invitational': 'Royal Never Give Up',
    '2022 Mid-Season Invitational': 'Royal Never Give Up',
    # LCK
    'Champions/2012 Season/Spring': 'MiG Blaze',
    'Champions/2012 Season/Summer': 'Azubu Frost',
    'Champions/2013 Season/Winter': 'NaJin Sword',
    'Champions/2013 Season/Spring': 'MVP Ozone',
    'Champions/2013 Season/Summer': 'SK Telecom T1 2',
    'Champions/2014 Season/Winter': 'SK Telecom T1 K',
    'Champions/2014 Season/Spring': 'Samsung Blue',
    'Champions/2014 Season/Summer': 'KT Rolster Arrows',
    'Champions/2015 Season/Spring': 'SK Telecom T1',
    'Champions/2015 Season/Summer': 'SK Telecom T1',
    'LCK/2016 Season/Spring': 'SK Telecom T1',
    'LCK/2016 Season/Summer': 'ROX Tigers',
    'LCK/2017 Season/Spring': 'SK Telecom T1',
    'LCK/2017 Season/Summer': 'Longzhu Gaming',
    'LCK/2018 Season/Spring': 'Kingzone DragonX',
    'LCK/2018 Season/Summer': 'KT Rolster',
    'LCK/2019 Season/Spring': 'SK Telecom T1',
    'LCK/2019 Season/Summer': 'SK Telecom T1',
    'LCK/2020 Season/Spring': 'T1',
    'LCK/2020 Season/Summer': 'DAMWON Gaming',
    'LCK/2021 Season/Spring': 'DWG KIA',
    'LCK/2021 Season/Summer': 'DWG KIA',
    'LCK/2022 Season/Spring': 'T1',
    'LCK/2022 Season/Summer': 'Gen.G',
    # LPL
    'LPL/2013 Season/Spring': 'Oh My God',
    'LPL/2013 Season/Summer': 'Positive Energy',
    'LPL/2014 Season/Spring': 'EDward Gaming',
    'LPL/2014 Season/Summer': 'EDward Gaming',
    'LPL/2015 Season/Spring': 'EDward Gaming',
    'LPL/2015 Season/Summer': 'LGD Gaming',
    'LPL/2016 Season/Spring': 'Royal Never Give Up',
    'LPL/2016 Season/Summer': 'EDward Gaming',
    'LPL/2017 Season/Spring': 'Team WE',
    'LPL/2017 Season/Summer': 'EDward Gaming',
    'LPL/2018 Season/Spring': 'Royal Never Give Up',
    'LPL/2018 Season/Summer': 'Royal Never Give Up',
    'LPL/2019 Season/Spring': 'Invictus Gaming',
    'LPL/2019 Season/Summer': 'FunPlus Phoenix',
    'LPL/2020 Season/Spring': 'JD Gaming',
    'LPL/2020 Season/Summer': 'Top Esports',
    'LPL/2021 Season/Spring': 'Royal Never Give Up',
    'LPL/2021 Season/Summer': 'EDward Gaming',
    'LPL/2022 Season/Spring': 'Royal Never Give Up',
    'LPL/2022 Season/Summer': 'JD Gaming',
    # LEC
    'EU LCS/Season 3/Spring': 'Fnatic',
    'EU LCS/Season 3/Summer': 'Fnatic',
    'EU LCS/2014 Season/Spring': 'Fnatic',
    'EU LCS/2014 Season/Summer': 'Alliance',
    'EU LCS/2015 Season/Spring': 'Fnatic',
    'EU LCS/2015 Season/Summer': 'Fnatic',
    'EU LCS/2016 Season/Spring': 'G2 Esports',
    'EU LCS/2016 Season/Summer': 'G2 Esports',
    'EU LCS/2017 Season/Spring': 'G2 Esports',
    'EU LCS/2017 Season/Summer': 'G2 Esports',
    'EU LCS/2018 Season/Spring': 'Fnatic',
    'EU LCS/2018 Season/Summer': 'Fnatic',
    'LEC/2019 Season/Spring': 'G2 Esports',
    'LEC/2019 Season/Summer': 'G2 Esports',
    'LEC/2020 Season/Spring': 'G2 Esports',
    'LEC/2020 Season/Summer': 'G2 Esports',
    'LEC/2021 Season/Spring': 'MAD Lions',
    'LEC/2021 Season/Summer': 'MAD Lions',
    'LEC/2022 Season/Spring': 'G2 Esports',
    'LEC/2022 Season/Summer': 'Rogue (European Team)',
    # LCS
    'NA LCS/Season 3/Spring': 'TSM',
    'NA LCS/Season 3/Summer': 'Cloud9',
    'NA LCS/2014 Season/Spring': 'Cloud9',
    'NA LCS/2014 Season/Summer': 'TSM',
    'NA LCS/2015 Season/Spring': 'TSM',
    'NA LCS/2015 Season/Summer': 'Counter Logic Gaming',
    'NA LCS/2016 Season/Spring': 'Counter Logic Gaming',
    'NA LCS/2016 Season/Summer': 'TSM',
    'NA LCS/2017 Season/Spring': 'TSM',
    'NA LCS/2017 Season/Summer': 'TSM',
    'NA LCS/2018 Season/Spring': 'Team Liquid',
    'NA LCS/2018 Season/Summer': 'Team Liquid',
    'LCS/2019 Season/Spring': 'Team Liquid',
    'LCS/2019 Season/Summer': 'Team Liquid',
    'LCS/2020 Season/Spring': 'Cloud9',
    'LCS/2020 Season/Summer': 'TSM',
    'LCS/2021 Season/Spring': 'Cloud9',
    'LCS/2021 Season/Summer': '100 Thieves',
    'LCS/2022 Season/Spring': 'Evil Geniuses.NA',
    'LCS/2022 Season/Summer': 'Cloud9',
}

# %%
winners = pd.DataFrame(data=WINNERS.items(), columns=['OverviewPage', 'Winner'])
winners

# %%
winners.to_csv('./csv/major_matches_winners.csv', index=False)

# %%
games = pd.read_csv('./csv/major_matches.csv')
games

# %%
winners = pd.read_csv('./csv/major_matches_winners.csv')
winners

# %%
games = pd.merge(games, winners, how='left', left_on=['OverviewPage', 'Team'], right_on=['OverviewPage', 'Winner'])
games['Winner'] = games['Winner'].apply(lambda x: 1 if isinstance(x, str) else 0)
games

# %%

# %%
leagues = get_leagues(where='L.League_Short in ("LTC")')
# leagues = get_leagues(where='L.Region="Korea"')
leagues

# %%
tournaments = pd.DataFrame()
for league in leagues['League Short']:
    # tournament = get_tournaments(where=f'L.League_Short="{league}"')
    # tournament = get_tournaments(where=f'L.League_Short="{league}" and T.Region="International"')
    tournament = get_tournaments(where=f'L.League_Short="{league}" and T.IsQualifier=0')
    tournaments = pd.concat([tournaments, tournament])
    
tournaments = tournaments.sort_values(by=['Year', 'DateStart', 'Date']).reset_index(drop=True)
tournaments.head()

# %%
tournaments['OverviewPage'].unique()

# %%
i = 35
overviews = '"' + tournaments.iloc[i]['OverviewPage'] + '"'
scoreboard_games = get_scoreboard_games(where=f'T.OverviewPage in ({overviews})')
if scoreboard_games is None:
    print(f'{tournaments.iloc[i]["OverviewPage"]} is None')
print(scoreboard_games.shape)
print(scoreboard_games['OverviewPage'].unique())
scoreboard_games.head()

# %%
i = 11
j = 1
overviews = ', '.join(map(lambda x: '"' + x + '"', tournaments.iloc[[i, j]]['OverviewPage']))
# overviews = ', '.join(map(lambda x: '"' + x + '"', tournaments.iloc[i:j+1]['OverviewPage']))
print(overviews)
scoreboard_games = get_scoreboard_games(where=f'T.OverviewPage in ({overviews})')
if scoreboard_games is None:
    print(f'{tournaments.iloc[i]["OverviewPage"]} is None')
# scoreboard_games['OverviewPage'] = scoreboard_games['OverviewPage'].str.split(' ').apply(lambda x: ' '.join(x[:-1]))
scoreboard_games['OverviewPage'] = scoreboard_games['OverviewPage'].str.split('/').apply(lambda x: x[0])
print(scoreboard_games.shape)
print(scoreboard_games['OverviewPage'].unique())
scoreboard_games.head()

# %%
# scoreboard_games['OverviewPage'] = ['LCS/2022 Season/Summer'] * scoreboard_games.shape[0]
# scoreboard_games['OverviewPage'].unique()

# %%
games = change_data(scoreboard_games)
print(scoreboard_games.shape[0] * 2 == games.shape[0])
games.head()

# %%
print(games['OverviewPage'].unique())
games['Team'].unique()

# %%

# %%
teams = get_team_stats(games)
teams.loc[winners[teams['OverviewPage'][0]], 'Winner'] = 1
teams.head()

# %%
# team_stats = pd.DataFrame()
team_stats = pd.concat([team_stats, teams])
team_stats

# %%
team_stats.reset_index().to_csv('./csv/msi_teams.csv', index=False)

# %%
team_stats = pd.read_csv('./csv/msi_teams.csv')
team_stats


# %%

# %%

# %%

# %%

# %%
def update_latest_tournament():
    tournaments = get_tournaments(
        where=f'L.League_Short="WCS" and T.Region="International" and T.Year=2022'
    ).sort_values(by=['DateStart', 'Date'])

    scoreboard_games = pd.DataFrame()
    for overview in tournaments['OverviewPage']:
        temp = get_scoreboard_games(where=f'T.OverviewPage="{overview}"')
        scoreboard_games = pd.concat([scoreboard_games, temp])
    scoreboard_games = scoreboard_games.reset_index(drop=True)
    scoreboard_games['OverviewPage'] = '2022 Season World Championship'

    games = change_data(scoreboard_games)
    print(scoreboard_games.shape[0] * 2 == games.shape[0])
    return games

games = update_latest_tournament()
games.sort_values(by='DateTime UTC').tail()

# %%
import os

games.to_csv(
    os.path.join(
        os.environ['DEV_PATH'],
        'datasets/LoL_esports/2022_worlds_matches.csv'
    ),
    index=False
)

# %%
league = get_leagues(where=f'L.League_Short="WCS"')
league

# %%
tournaments = get_tournaments(where=f'L.League_Short="WCS" and T.Region="International"')
tournaments

# %%
tournaments = tournaments.loc[[19, 20]]
tournaments

# %%
scoreboard_games = pd.DataFrame()
for overview in tournaments['OverviewPage']:
    temp = get_scoreboard_games(where=f'T.OverviewPage="{overview}"')
    scoreboard_games = pd.concat([scoreboard_games, temp])
scoreboard_games = scoreboard_games.reset_index(drop=True)
scoreboard_games['OverviewPage'] = '2022 Season World Championship'
scoreboard_games

# %%
games = change_data(scoreboard_games)
print(scoreboard_games.shape[0] * 2 == games.shape[0])
games

# %%
games.sort_values(by='DateTime UTC').tail()

# %%
games.to_csv('./csv/2022_worlds_matches.csv', index=False)

# %%

import pandas as pd

from lol_fandom import get_leagues, get_tournaments
from lol_fandom import get_scoreboard_games

pd.set_option('display.max_columns', None)

LEAGUES = [
    'WCS', 'MSI', 'LCK', 'LPL', 'LEC', 'LCS'
]

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


def change_data(games):
    df = pd.DataFrame()
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
            'Gold': [row.Team1Gold, row.Team2Gold],
            'Kills': [row.Team1Kills, row.Team2Kills],
            'Deaths': [row.Team2Kills, row.Team1Kills],
            'KPM': [row.Team1Kills / row._11, row.Team2Kills / row._11],
            'KDPM': [
                (row.Team1Kills - row.Team2Kills) / row._11,
                (row.Team2Kills - row.Team1Kills) / row._11
            ],
            'GPM': [row.Team1Gold / row._11, row.Team2Gold / row._11],
            'GDPM': [
                (row.Team1Gold - row.Team2Gold) / row._11,
                (row.Team2Gold - row.Team1Gold) / row._11
            ]
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



def main():
    leagues = get_leagues(where='L.League_Short in ("NA LCS", "LCS")')
    tournaments = pd.DataFrame()
    for league in leagues['League Short']:
        tournament = get_tournaments(
            where=f'L.League_Short="{league}" and T.IsQualifier=0'
        )
        tournaments = pd.concat([tournaments, tournament])

    tournaments = tournaments.sort_values(
        by=['Year', 'DateStart', 'Date']
    ).reset_index(drop=True)

    team_stats = pd.DataFrame()

    # for i in range(tournaments.shape[0]):
    #     overview = '"' + tournaments.iloc[i]['OverviewPage'] + '"'
    #     scoreboard_games = get_scoreboard_games(where=f'T.OverviewPage in ({overview})')
    #     if scoreboard_games is None:
    #         print(f'{tournaments.iloc[i]["OverviewPage"]} is None')
    #         continue
    #     games = change_data(scoreboard_games)
    #     if scoreboard_games.shape[0] * 2 != games.shape[0]:
    #         print(f'{tournaments.iloc[i]["OverviewPage"]} error')
    #         return None
    #     print(i, games['OverviewPage'].unique())
    #     teams = get_team_stats(games)
    #     if teams['OverviewPage'][0] not in WINNERS:
    #         print(f'{tournaments.iloc[i]["OverviewPage"]} nan')
    #         continue
    #     teams.loc[WINNERS[teams['OverviewPage'][0]], 'Winner'] = 1
    #     team_stats = pd.concat([team_stats, teams])
    # print(team_stats)
    # print('\n\n')

    for i in range(0, 33, 2):
        overviews = ', '.join(map(
            lambda x: '"' + x + '"',
            tournaments.iloc[[i, i + 1]]['OverviewPage']
        ))
        scoreboard_games = get_scoreboard_games(
            where=f'T.OverviewPage in ({overviews})'
        )
        if scoreboard_games is None:
            print(f'{tournaments.iloc[i]["OverviewPage"]} is None')
            continue
        scoreboard_games['OverviewPage'] = \
            scoreboard_games['OverviewPage'].str.split(' ').apply(
                lambda x: ' '.join(x[:-1])
            )
        games = change_data(scoreboard_games)
        if scoreboard_games.shape[0] * 2 != games.shape[0]:
            print(f'{tournaments.iloc[i]["OverviewPage"]} error')
            return None
        print(games['OverviewPage'].unique())
        teams = get_team_stats(games)
        if teams['OverviewPage'][0] not in WINNERS:
            print(f'{tournaments.iloc[i]["OverviewPage"]} nan')
            continue
        teams.loc[WINNERS[teams['OverviewPage'][0]], 'Winner'] = 1
        team_stats = pd.concat([team_stats, teams])
    print(team_stats)

    for i in [34, 36, 39, 41]:
        overviews = ', '.join(map(
            lambda x: '"' + x + '"',
            tournaments.iloc[[i, i + 1]]['OverviewPage']
        ))
        scoreboard_games = get_scoreboard_games(
            where=f'T.OverviewPage in ({overviews})'
        )
        if scoreboard_games is None:
            print(f'{tournaments.iloc[i]["OverviewPage"]} is None')
            continue
        scoreboard_games['OverviewPage'] = \
            [' '.join(tournaments.iloc[i]['OverviewPage'].split(' ')[:-1])] * \
            scoreboard_games.shape[0]
        games = change_data(scoreboard_games)
        if scoreboard_games.shape[0] * 2 != games.shape[0]:
            print(f'{tournaments.iloc[i]["OverviewPage"]} error')
            return None
        print(overviews, games['OverviewPage'].unique())
        teams = get_team_stats(games)
        if teams['OverviewPage'][0] not in WINNERS:
            print(f'{tournaments.iloc[i]["OverviewPage"]} nan')
            continue
        teams.loc[WINNERS[teams['OverviewPage'][0]], 'Winner'] = 1
        team_stats = pd.concat([team_stats, teams])
    print(team_stats)

    # team_stats.reset_index().to_csv('./csv/lcs_teams.csv', index=False)


if __name__ == '__main__':
    main()

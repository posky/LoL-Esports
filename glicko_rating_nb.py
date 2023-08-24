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
import math

import pandas as pd
import numpy as np
import mwclient

from google_sheet import Sheet

from lol_fandom import SITE
from lol_fandom import get_leagues, get_tournaments
from lol_fandom import get_scoreboard_games, get_scoreboard_players
from lol_fandom import from_response

pd.set_option('display.max_columns', None)
with open('./sheet_id.txt', 'r') as f:
    SHEET_ID = f.read()

# %%
sheet = Sheet(SHEET_ID)
sheet.connect_sheet()


# %%
class Team:
    q = math.log(10) / 400
    leagues_r = {
        'LCK': 1000,
        'LPL': 1000,
        'LEC': 950,
        'LCS': 900,
        'PCS': 850,
        'VCS': 850,
        'LJL': 800,
        'CBLOL': 800,
        'LLA': 800,
        'LCO': 800,
        'TCL': 800,
        'LCL': 800,
    }

    def __init__(self, name, league):
        self.name = name
        self.league = league
        self.win = 0
        self.loss = 0
        self.r = 1000
        self.r = self.leagues_r[league]
        self.RD = 350

    def get_g(RDi):
        return 1 / math.sqrt(1 + (3 * Team.q ** 2 * RDi ** 2) / math.pi ** 2)

    def get_e(r0, ri, g):
        return 1 / (1 + 10 ** ((g * (r0 - ri)) / -400))

    def get_d(g, e):
        return 1 / (Team.q ** 2 * g ** 2 * e * (1 - e))

    def init_rd(self):
        self.RD = 350
    
    def update_point(team1, team2, result):
        # team1 win - result = 1 team1 loss - result = 0
        assert isinstance(team1, Team)
        assert isinstance(team2, Team)

        team1_r = team1.r
        team2_r = team2.r
        team1_RD = team1.RD
        team2_RD = team2.RD

        team1._update_point(team2_r, team2_RD, result)
        team2._update_point(team1_r, team1_RD, 1 - result)

    def _update_point(self, ri, RDi, s):
        if s == 1:
            self.win += 1
        else:
            self.loss += 1

        g_RD = Team.get_g(RDi)
        e = Team.get_e(self.r, ri, g_RD)
        d_2 = Team.get_d(g_RD, e)
        self.r = self.r + Team.q / (1 / self.RD ** 2 + 1 / d_2) * g_RD * (s - e)

        self.RD = math.sqrt((1 / self.RD ** 2 + 1 / d_2) ** -1)

    def get_win_prob(self, opponent):
        return Team.get_e(self.r, opponent.r, Team.get_g(opponent.RD))

    def to_dict(self):
        data = {
            'League': self.league,
            'Win': self.win,
            'Loss': self.loss,
            'WinRate': self.win / (self.win + self.loss) if self.win != 0 else 0,
            'r': self.r,
            'RD': self.RD
        }

        return data



# %%
def proceed_rating(teams, games):
    for row in games.itertuples():
        team1, team2 = row.Team1, row.Team2
        result = 1 if row.WinTeam == team1 else 0
        Team.update_point(teams[team1], teams[team2], result)

def get_rating(teams):
    ratings = pd.DataFrame(
        data=map(lambda x: x.to_dict(), teams.values()),
        index=teams.keys()
    )
    ratings = ratings.sort_values(by='r', ascending=False)
    return ratings

def get_team_name(same_team_names, name):
    while name in same_team_names:
        name = same_team_names[name]
    return name


# %%
leagues = get_leagues()
leagues

# %%
target_leagues = [
    'LCK', 'LPL', 'LEC', 'LCS', 'MSI', 'WCS',
    'PCS', 'VCS', 'LJL', 'CBLOL', 'LLA', 'LCO', 'TCL', 'LCL',
]

# %%
response = SITE.api(
    'cargoquery',
    limit='max',
    tables='Leagues=L, Tournaments=T',
    join_on='L.League=T.League',
    fields='T.Name, T.OverviewPage, T.DateStart, T.Date, T.League, ' +
        'T.Region, T.EventType, T.StandardName, T.Split, T.SplitNumber, ' +
        'T.TournamentLevel, T.IsQualifier, T.IsPlayoffs, T.IsOfficial, T.Year',
    where='L.League_Short="LCO" and T.Year=2020'
)
response

# %%
tournaments = pd.DataFrame()
for league in target_leagues:
# for league in ['LCL']:
    t = get_tournaments(f'L.League_Short="{league}" and T.Year in (2022)')
    tournaments = pd.concat([tournaments, t])
tournaments = tournaments.sort_values(by=['Year', 'DateStart', 'Date']).reset_index(drop=True)
tournaments

# %%
same_team_names = {
    # LCK
    'Afreeca Freecs': 'Kwangdong Freecs',
    # LPL
    'eStar (Chinese Team)': 'Ultra Prime',
    'Rogue Warriors': "Anyone's Legend",
    'Suning': 'Weibo Gaming',
    # PCS
    'Alpha Esports': 'Hurricane Gaming',
    # VCS
    'Percent Esports': 'Burst The Sky Esports',
    'Luxury Esports': 'GMedia Luxury',
    # CBLOL
    'Flamengo Esports': 'Flamengo Los Grandes',
    'Netshoes Miners': 'Miners',
    'Vorax': 'Vorax Liberty',
    'Cruzeiro eSports': 'Netshoes Miners',
    'Vorax Liberty': 'Liberty',
    # LCO
    'Legacy Esports': 'Kanga Esports',
    # TCL
    'SuperMassive Esports': 'SuperMassive Blaze',
}

# %%
teams = {}
for page in tournaments['OverviewPage']:
    print(f'{page} rating ...')
    scoreboard_games = get_scoreboard_games(f'T.OverviewPage="{page}"')
    if scoreboard_games is None:
        print(f'{page} is None\n')
        continue
    print(f'{scoreboard_games.shape[0]} games')
    scoreboard_games = scoreboard_games.sort_values(by='DateTime UTC').reset_index(drop=True)

    team_names = scoreboard_games[['Team1', 'Team2']].apply(pd.unique)
    team_names = list(set(list(team_names['Team1']) + list(team_names['Team2'])))
    league = page.split('/')[0]
    new_teams = {}
    for name in team_names:
        new_name = get_team_name(same_team_names, name)
        if name not in teams:
            if name == new_name:
                teams[name] = Team(name, league)
                new_teams[(name,)] = True
            else:
                if new_name not in teams:
                    teams[new_name] = Team(new_name, league)
                    new_teams[(name, new_name)] = True
                teams[name] = teams[new_name]
    if len(new_teams) > 0:
        print(f'There are {len(new_teams)} new teams')
        print(sorted(list(new_teams.keys()), key=lambda x: x[0].lower()))
    print()
    proceed_rating(teams, scoreboard_games)

rating = get_rating(teams)

# %%
team_names = [
    'Gen.G', 'T1', 'DWG KIA', 'DRX', 
    'JD Gaming', 'Top Esports', 'EDward Gaming', 'Royal Never Give Up',
    'G2 Esports', 'Rogue (European Team)', 'Fnatic', 'MAD Lions',
    'Cloud9', '100 Thieves', 'Evil Geniuses.NA',
    'CTBC Flying Oyster', 'GAM Esports',
    'DetonatioN FocusMe', 'LOUD', 'Beyond Gaming', 'Saigon Buffalo', 'Isurus', 'Istanbul Wildcats', 'Chiefs Esports Club'
]

rating.loc[team_names].sort_values(by='r', ascending=False)

# %%
team_names = [
    'Gen.G', 'T1', 'DWG KIA', 'DRX', 
    'JD Gaming', 'Top Esports', 'EDward Gaming', 'Royal Never Give Up',
    'G2 Esports', 'Rogue (European Team)', 'Fnatic',
    'Cloud9', '100 Thieves', 'Evil Geniuses.NA',
    'CTBC Flying Oyster', 'GAM Esports'
]

rating = rating.loc[team_names].sort_values(by='r', ascending=False)
rating.index = rating.index.set_names('Team')
sheet.update_sheet('glicko_rating', rating)

# %%
rating.loc[team_names].sort_values(by='r', ascending=False)

# %%

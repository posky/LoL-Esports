"""LoLesports Elo rating"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lol_fandom import get_tournaments
from lol_fandom import get_scoreboard_games

from google_sheet import Sheet

class Team:
    """Team information"""
    K = 20
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
        self.point = 1000
        # self.point = self.leagues_r[league]
        self.points = [self.point]

    @classmethod
    def update_point(cls, team1, team2, result):
        """Update rating of team1 and team2

        Args:
            team1 (Team): Team1
            team2 (Team): Team2
            result (int): 1 is Team1 win, 0 is Team2 win
        """
        assert isinstance(team1, Team)
        assert isinstance(team2, Team)

        team1_wr = team1.get_win_prob(team2)
        team2_wr = team2.get_win_prob(team1)
        team1._update_point(team1_wr, result)
        team2._update_point(team2_wr, 1 - result)

    def get_win_prob(self, opponent):
        """Get win probability

        Args:
            opponent (Team): Opponent team

        Returns:
            float: Win probability
        """
        assert isinstance(opponent, Team)

        return 1 / (10 ** ((opponent.point - self.point) / 400) + 1)

    def _update_point(self, winrate, result):
        # result: win 1, loss 0
        assert result == 0 or result == 1

        if result == 1:
            self.win += 1
        else:
            self.loss += 1

        self.point = self.point + self.K * (result - winrate)
        self.points.append(self.point)

    def to_dict(self):
        """Make Team to dictionary

        Returns:
            dict: Dictionary of Team instance
        """
        data = {
            'League': self.league,
            'Win': self.win,
            'Loss': self.loss,
            'WinRate': self.win / (self.win + self.loss) if self.win != 0 else 0,
            'Point': self.point
        }

        return data

    def to_dataframe(self):
        """Make Team points to DataFrame

        Returns:
            DataFrame: Points of Team
        """
        df = pd.DataFrame({
            'name': [self.name] * len(self.points),
            'point': self.points
        })
        return df


def proceed_rating(teams, games):
    """Proceed rating with teams and games

    Args:
        teams (list): List of Team instances
        games (DataFrame): Scoreboard games
    """
    for row in games.itertuples():
        team1, team2 = row.Team1, row.Team2
        result = 1 if row.WinTeam == team1 else 0
        Team.update_point(teams[team1], teams[team2], result)

def get_rating(teams):
    """Get rating of teams

    Args:
        teams (list): List of Team instances

    Returns:
        DataFrame: Rating of teams
    """
    ratings = pd.DataFrame(
        data=map(lambda x: x.to_dict(), teams.values()),
        index=teams.keys()
    )
    ratings = ratings.sort_values(by='Point', ascending=False)
    return ratings

def get_team_name(same_team_names, name):
    """Get latest name of the team

    Args:
        same_team_names (dict): Dictionary of names of same teams
        name (str): Name of the team

    Returns:
        str: Latest name of the team
    """
    while name in same_team_names:
        name = same_team_names[name]
    return name


pd.set_option('display.max_columns', None)
with open('./sheet_id.txt', 'r', encoding='utf8') as f:
    SHEET_ID = f.read()

TARGET_LEAGUES = [
    'LCK', 'LPL', 'LEC', 'LCS', 'MSI', 'WCS',
    'PCS', 'VCS', 'LJL', 'CBLOL', 'LLA', 'LCO', 'TCL', 'LCL',
]
SAME_TEAM_NAMES = {
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


def main():
    """Elo rating system"""
    sheet = Sheet(SHEET_ID)
    sheet.connect_sheet()

    tournaments = pd.DataFrame()
    for league in TARGET_LEAGUES:
        t = get_tournaments(f'L.League_Short="{league}" and T.Year=2022')
        tournaments = pd.concat([tournaments, t])
    tournaments = tournaments.sort_values(
        by=['Year', 'DateStart', 'Date']
    ).reset_index(drop=True)

    teams = {}
    for page in tournaments['OverviewPage']:
        print(f'{page} rating ...')
        scoreboard_games = get_scoreboard_games(f'T.OverviewPage="{page}"')
        if scoreboard_games is None:
            print(f'{page} is None\n')
            continue
        print(f'{scoreboard_games.shape[0]} games')
        scoreboard_games = scoreboard_games.sort_values(
            by='DateTime UTC'
        ).reset_index(drop=True)

        team_names = scoreboard_games[['Team1', 'Team2']].apply(pd.unique)
        team_names = list(set(list(team_names['Team1']) + list(team_names['Team2'])))
        league = page.split('/')[0]
        new_teams = {}
        for name in team_names:
            new_name = get_team_name(SAME_TEAM_NAMES, name)
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


    team_names = [
        'Gen.G', 'T1', 'DWG KIA', 'DRX',
        'JD Gaming', 'Top Esports', 'EDward Gaming', 'Royal Never Give Up',
        'G2 Esports', 'Rogue (European Team)', 'Fnatic',
        'Cloud9', '100 Thieves', 'Evil Geniuses.NA',
        'CTBC Flying Oyster', 'GAM Esports'
    ]

    rating.index = rating.index.set_names('Team')
    sheet.update_sheet('elo_rating', rating.loc[team_names].sort_values(
        by='Point', ascending=False
    ))


    _, ax = plt.subplots(1, 1, figsize=(20, 20))
    df = pd.DataFrame()
    for name in team_names:
        team = teams[name]
        temp = team.to_dataframe()
        df = pd.concat([df, temp])
    sns.lineplot(data=df, x=df.index, y='point', hue='name', ax=ax)
    plt.show()



if __name__ == '__main__':
    main()

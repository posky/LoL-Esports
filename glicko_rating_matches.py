import math
from itertools import combinations, permutations

import pandas as pd

from lol_fandom import get_tournaments
from lol_fandom import get_scoreboard_games


pd.set_option('display.max_columns', None)


class Team:
    """Team information"""
    q = math.log(10) / 400

    def __init__(self, name, league):
        self.name = name
        self.league = league
        self.win = 0
        self.loss = 0
        self.streak = 0
        self.r = 1000
        self.RD = 350
        self.last_game_date = None

    def update_team_name(self, name):
        self.name = name

    def update_league(self, league):
        self.league = league

    def update_last_game_date(self, game_date):
        self.last_game_date = game_date

    def update_streak(self, result):
        result = 1 if result == 1 else -1
        if (self.streak > 0) == (result > 0):
            self.streak += result
        else:
            self.streak = result

    @classmethod
    def get_g(cls, RDi):
        """Compute g(RDi)

        Args:
            RDi (float): Ratings Deviation (RD)

        Returns:
            float: g(RDi)
        """
        return 1 / math.sqrt(1 + (3 * cls.q ** 2 * RDi ** 2) / math.pi ** 2)

    @classmethod
    def get_e(cls, r0, ri, g):
        """Compute E(s|r0, ri, RDi)

        Args:
            r0 (float): previous rating
            ri (float): rating of opponent
            g (float): g(RDi)

        Returns:
            float: E(s | r0, ri, RDi)
        """
        return 1 / (1 + 10 ** ((g * (r0 - ri)) / -400))

    @classmethod
    def get_d(cls, g, e):
        """Compute d^2

        Args:
            g (float): g(RDi)
            e (float): E(s | r0, ri, RDi)

        Returns:
            float: d^2
        """
        return 1 / (cls.q ** 2 * g ** 2 * e * (1 - e))

    def init_rd(self):
        """Initialize RD"""
        self.RD = 350

    @classmethod
    def update_point(cls, team1, team2, result):
        """Update ratings of team1 and team2

        Args:
            team1 (Team): Team1
            team2 (Team): Team2
            result (int): 1 is Team1 win, 0 is Team2 win
        """
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
        """Get win probability

        Args:
            opponent (Team): Opponent team

        Returns:
            float: Win probability
        """
        return Team.get_e(self.r, opponent.r, Team.get_g(opponent.RD))

    def to_dict(self):
        """Make Team class to dictionary

        Returns:
            dict: Dictionary of Team instance
        """
        data = {
            'Team': self.name,
            'League': self.league,
            'Win': self.win,
            'Loss': self.loss,
            'WinRate': self.win / (self.win + self.loss) if self.win != 0 else 0,
            'Streak': self.streak,
            'r': self.r,
            'RD': self.RD,
            'last_game_date': self.last_game_date,
        }

        return data

def get_team_id(teams_id, name):
    return teams_id.loc[teams_id['team'] == name, 'team_id'].iloc[0]

def proceed_rating(teams_id, teams, games):
    """Proceed rating with teams and games

    Args:
        teams_id (DataFrame): ID of teams
        teams (list): List of Team instances
        games (DataFrame): Scoreboard games
    """
    for row in games.itertuples():
        team1, team2 = row.Team1, row.Team2
        game_date = row._6
        result = 1 if row.WinTeam == team1 else 0
        id1, id2 = get_team_id(teams_id, team1), get_team_id(teams_id, team2)
        Team.update_point(teams[id1], teams[id2], result)
        teams[id1].update_last_game_date(game_date)
        teams[id2].update_last_game_date(game_date)
        teams[id1].update_streak(result)
        teams[id2].update_streak(1 - result)

def get_rating(teams):
    """Get rating of teams

    Args:
        teams (list): List of Team instances

    Returns:
        DataFrame: Rating of teams
    """
    ratings = pd.DataFrame(data=map(lambda x: x.to_dict(), teams.values()))
    ratings = ratings.sort_values(by='r', ascending=False).reset_index(drop=True)
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

def is_proper_league(league):
    if 'WCS' == league or 'MSI' == league:
        return False
    return True


def main():
    teams_id = pd.read_csv('./csv/teams_id.csv')

    teams = {}
    for year in range(2011, 2024):
        tournaments = pd.read_csv(f'./csv/tournaments/{year}_tournaments.csv')
        scoreboard_games = pd.read_csv(f'./csv/scoreboard_games/{year}_scoreboard_games.csv')
        for page in tournaments['OverviewPage']:
            print(f'{page} rating ...')
            sg = scoreboard_games.loc[scoreboard_games['OverviewPage'] == page]
            print(f'\t{sg.shape[0]} matches')

            league = sg['League'].iloc[0]
            team_names = sg[['Team1', 'Team2']].unstack().unique()
            team_check = True
            for name in team_names:
                if name not in teams_id['team'].values:
                    print(f'{name} not in teams')
                    team_check = False
                    break
                id = get_team_id(teams_id, name)
                if id not in teams:
                    teams[id] = Team(name, league)
                else:
                    teams[id].update_team_name(name)
                    if is_proper_league(league):
                        teams[id].update_league(league)
            if not team_check:
                break

            for name in team_names:
                id = get_team_id(teams_id, name)
                teams[id].init_rd()

            proceed_rating(teams_id, teams, sg)

        if not team_check:
            break
    if team_check:
        rating = get_rating(teams)
        rating.to_csv('./csv/glicko_rating.csv', index=False)


if __name__ == '__main__':
    main()
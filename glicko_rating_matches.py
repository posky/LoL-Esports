import os
import math
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


pd.set_option("display.max_columns", None)
DEFAULT_C = 195.618


class Glicko:
    Q = math.log(10) / 400

    def __init__(self):
        pass

    def rate(self, rating1, rating2, result):
        new_rating1, new_ratings_deviation1, error1 = self._rate(
            rating1, rating2, result
        )
        new_rating2, new_ratings_deviation2, error2 = self._rate(
            rating2, rating1, 1 - result
        )

        rating1.update(new_rating1, new_ratings_deviation1, error1)
        rating2.update(new_rating2, new_ratings_deviation2, error2)

    def _rate(self, rating1, rating2, result):
        g = self.get_g(rating2.ratings_deviation)
        expectation = self.get_expectation(g, rating1.rating, rating2.rating)
        d_square_inverse = self.get_d_square_inverse(g, expectation)

        rating = rating1.rating + self.Q / (
            rating1.ratings_deviation**-2 + d_square_inverse
        ) * g * (result - expectation)
        ratings_deviation = (rating1.ratings_deviation**-2 + d_square_inverse) ** -0.5

        error = abs(result - expectation)

        return rating, ratings_deviation, error

    def get_g(self, ratings_deviation):
        return (1 + (3 * self.Q**2 * ratings_deviation**2) / math.pi**2) ** -0.5

    def get_expectation(self, g, my_rating, op_rating):
        return 1 / (1 + 10 ** (g * (my_rating - op_rating) / -400))

    def get_d_square_inverse(self, g, expectation):
        return self.Q**2 * g**2 * expectation * (1 - expectation)

    def get_win_probability(self, rating1, rating2):
        g = self.get_g(rating2.ratings_deviation)
        expectation = self.get_expectation(g, rating1.rating, rating2.rating)
        return expectation


class Rating:
    C = DEFAULT_C

    def __init__(self, rating=1000, ratings_deviation=350, games=0, error=0):
        self.rating = rating
        self.ratings_deviation = ratings_deviation
        self.games = games
        self.error = error

    def update(self, rating, ratings_deviation, error):
        self.rating = rating
        self.ratings_deviation = ratings_deviation
        self.games += 1
        self.error += error

    def init_ratings_deviation(self):
        self.ratings_deviation = min(
            (self.ratings_deviation**2 + self.C**2 * self.games) ** 0.5, 350
        )
        self.games = 0


class Team:
    """Team information"""

    glicko = Glicko()

    def __init__(
        self,
        name,
        league,
        win=0,
        loss=0,
        streak=0,
        rating=1000,
        ratings_deviation=350,
        games=0,
        error=0.0,
        last_game_date=None,
    ):
        self.name = name
        self.league = league
        self.win = win
        self.loss = loss
        self.streak = streak
        self.rating = Rating(rating, ratings_deviation, games, error)
        self.last_game_date = last_game_date

    def update_team_name(self, name):
        self.name = name

    def update_league(self, league):
        self.league = league

    def update_last_game_date(self, game_date):
        self.last_game_date = game_date

    def update_result(self, result):
        if result == 1:
            self.win += 1
        else:
            self.loss += 1

        self.update_streak(result)

    def update_streak(self, result):
        result = 1 if result == 1 else -1
        if (self.streak > 0) == (result > 0):
            self.streak += result
        else:
            self.streak = result

    @classmethod
    def update_match(cls, team1, team2, result, game_date):
        assert isinstance(team1, Team) and isinstance(team2, Team)

        cls.glicko.rate(team1.rating, team2.rating, result)

        team1.update_result(result)
        team2.update_result(1 - result)
        team1.update_last_game_date(game_date)
        team2.update_last_game_date(game_date)

    def init_ratings_deviation(self):
        self.rating.init_ratings_deviation()

    def get_win_probability(self, other):
        assert isinstance(other, Team)

        return self.glicko.get_win_probability(self.rating, other.rating)

    def to_tuple(self):
        return (
            self.name,
            self.league,
            self.win,
            self.loss,
            self.win / (self.win + self.loss) if self.win > 0 else 0,
            self.streak,
            self.rating.rating,
            self.rating.ratings_deviation,
            self.rating.games,
            self.rating.error,
            self.last_game_date,
        )


def get_team_id(teams_id, name):
    return teams_id.loc[teams_id["team"] == name, "team_id"].iloc[0]


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
        Team.update_match(teams[id1], teams[id2], result, game_date)


def get_rating(teams):
    """Get rating of teams

    Args:
        teams (list): List of Team instances

    Returns:
        DataFrame: Rating of teams
    """
    # ratings = pd.DataFrame(data=map(lambda x: x.to_dict(), teams.values()))
    data = np.array(
        list(map(lambda x: x.to_tuple(), teams.values())),
        dtype=[
            ("team", "object"),
            ("league", "object"),
            ("win", "int"),
            ("loss", "int"),
            ("winrate", "float"),
            ("streak", "int"),
            ("rating", "float"),
            ("ratings_deviation", "float"),
            ("games", "int"),
            ("error", "float"),
            ("last_game_date", "datetime64[ns]"),
        ],
    )
    ratings = pd.DataFrame.from_records(data)
    ratings = ratings.sort_values(by="rating", ascending=False).reset_index(drop=True)
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
    if league in ["WCS", "MSI"]:
        return False
    return True


def parse_teams(teams_id, rating):
    teams = {}
    for row in rating.itertuples():
        team_id = get_team_id(teams_id, row.team)
        team = Team(
            row.team,
            row.league,
            row.win,
            row.loss,
            row.streak,
            row.rating,
            row.ratings_deviation,
            row.games,
            row.error,
            row.last_game_date,
        )
        teams[team_id] = team
    return teams


def main():
    teams_id = pd.read_csv("./csv/teams_id.csv")

    for year in tqdm(range(2023, 2024)):
        if os.path.isfile(f"./csv/glicko_rating/glicko_rating_{year - 1}.csv"):
            logging.info("Parse %d rating ...", year - 1)
            rating = pd.read_csv(
                f"./csv/glicko_rating/glicko_rating_{year - 1}.csv",
                parse_dates=["last_game_date"],
            )
            teams = parse_teams(teams_id, rating)
            logging.info("Complete")
        else:
            teams = {}

        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv"
        )
        for page in tqdm(tournaments["OverviewPage"]):
            logging.info("%s rating ...", page)
            sg = scoreboard_games.loc[scoreboard_games["OverviewPage"] == page]
            logging.info("\t%d matches", sg.shape[0])

            league = sg["League"].iloc[0]
            team_names = sg[["Team1", "Team2"]].unstack().unique()
            team_check = True
            for name in team_names:
                if name not in teams_id["team"].values:
                    logging.error("%s not in teams", name)
                    team_check = False
                    break
                team_id = get_team_id(teams_id, name)
                if team_id not in teams:
                    teams[team_id] = Team(name, league)
                else:
                    teams[team_id].update_team_name(name)
                    if is_proper_league(league):
                        teams[team_id].update_league(league)
            if not team_check:
                break

            for name in team_names:
                team_id = get_team_id(teams_id, name)
                teams[team_id].init_ratings_deviation()

            proceed_rating(teams_id, teams, sg)
            ratings = get_rating(teams)
            ratings.to_csv(f"./csv/glicko_rating/glicko_rating_{year}.csv", index=False)

        if not team_check:
            break


if __name__ == "__main__":
    main()

import os
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


class Elo:
    def __init__(self):
        pass

    def rate(self, rating1, rating2, result):
        new_rating1, error1 = self._rate(rating1, rating2, result)
        new_rating2, error2 = self._rate(rating2, rating1, 1 - result)

        rating1.update(new_rating1, error1)
        rating2.update(new_rating2, error2)

    def _rate(self, rating1, rating2, result):
        expectation = rating1.get_expectation(rating2)
        k_factor = rating1.get_k_factor()
        rating = rating1.rating + k_factor * (result - expectation)
        error = abs(result - expectation)

        return rating, error

    def get_win_probability(self, rating1, rating2):
        return rating1.get_expectation(rating2)


class Rating:
    # LOW_K = 32  # below 2100
    # MID_K = 24  # between 2100 and 2400
    # HIGH_K = 16  # above 2400

    LOW_K = 140  # below 2100
    MID_K = 38  # between 2100 and 2400
    HIGH_K = 25  # above 2400

    def __init__(self, rating=1500, error=0.0):
        self.rating = rating
        self.error = error

    def get_expectation(self, other):
        return 1 / (1 + 10 ** ((other.rating - self.rating) / 400))

    def get_k_factor(self):
        if self.rating < 2100:
            return self.LOW_K
        if self.rating <= 2400:
            return self.MID_K
        return self.HIGH_K

    def update(self, rating, error):
        self.rating = rating
        self.error += error


class Team:
    elo = Elo()

    def __init__(
        self,
        name,
        league,
        win=0,
        loss=0,
        streak=0,
        rating=1500,
        error=0.0,
        last_game_date=None,
    ):
        self.name = name
        self.league = league
        self.win = win
        self.loss = loss
        self.streak = streak
        self.rating = Rating(rating, error)
        self.last_game_date = last_game_date

    @property
    def winrate(self):
        return self.win / (self.win + self.loss) if self.win > 0 else 0

    def update_name(self, name):
        self.name = name

    def update_league(self, league):
        self.league = league

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

    def update_last_game_date(self, game_date):
        self.last_game_date = game_date

    @classmethod
    def update_match(cls, team1, team2, result, game_date):
        cls.elo.rate(team1.rating, team2.rating, result)

        team1.update_result(result)
        team2.update_result(1 - result)
        team1.update_last_game_date(game_date)
        team2.update_last_game_date(game_date)

    def get_win_probability(self, other):
        assert isinstance(other, Team)

        return self.elo.get_win_probability(self.rating, other.rating)

    def to_tuple(self):
        return (
            self.name,
            self.league,
            self.win,
            self.loss,
            self.winrate,
            self.streak,
            self.rating.rating,
            self.rating.error,
            self.last_game_date,
        )


def get_team_id(teams_id, team_name):
    return teams_id.loc[teams_id["team"] == team_name, "team_id"].iloc[0]


def parse_teams(teams_id, ratings):
    teams = {}
    for row in ratings.itertuples():
        teams[get_team_id(teams_id, row.team)] = Team(
            row.team,
            row.league,
            row.win,
            row.loss,
            row.streak,
            row.rating,
            row.error,
            row.last_game_date,
        )
    return teams


def is_proper_league(league):
    if league in ["WCS", "MSI"]:
        return False
    return True


def setup_teams(teams_id, teams, team_names, league):
    for team_name in team_names:
        if team_name not in teams_id["team"].values:
            logging.error('%s not in teams', team_name)
            return False
        team_id = get_team_id(teams_id, team_name)
        if team_id not in teams:
            teams[team_id] = Team(team_name, league)
        else:
            teams[team_id].update_name(team_name)
            if is_proper_league(league):
                teams[team_id].update_league(league)
    return True


def rate(teams_id, teams, games):
    for row in games.itertuples():
        team1, team2 = row.Team1, row.Team2
        game_date = row._6
        result = 1 if row.WinTeam == team1 else 0
        team1_id, team2_id = get_team_id(teams_id, team1), get_team_id(teams_id, team2)
        Team.update_match(teams[team1_id], teams[team2_id], result, game_date)


def get_ratings(teams):
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
            ("error", "float"),
            ("last_game_date", "datetime64[ns]"),
        ],
    )
    ratings = pd.DataFrame.from_records(data)
    ratings = ratings.sort_values(by="rating", ascending=False, ignore_index=True)
    return ratings


def main(verbose=0):
    if verbose == 1:
        logging.basicConfig(level=logging.INFO)

    teams_id = pd.read_csv("./csv/teams_id.csv")

    for year in tqdm(range(2011, 2024)):
        if os.path.isfile(f"./csv/elo_rating/{year - 1}_elo_rating.csv"):
            logging.info('Parse %d ratings ...', year - 1)
            ratings = pd.read_csv(
                f"./csv/elo_rating/{year - 1}_elo_rating.csv",
                parse_dates=["last_game_date"],
            )
            teams = parse_teams(teams_id, ratings)
            logging.info('Complete!')
        else:
            teams = {}

        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv"
        )
        for page in tournaments["OverviewPage"]:
            logging.info('%s rating ...', page)
            sg = scoreboard_games.loc[scoreboard_games["OverviewPage"] == page]
            logging.info('%d matches', sg.shape[0])

            league = sg["League"].iloc[0]
            team_names = sg[["Team1", "Team2"]].unstack().unique()
            team_check = setup_teams(teams_id, teams, team_names, league)
            if not team_check:
                break

            rate(teams_id, teams, sg)
            ratings = get_ratings(teams)
            ratings.to_csv(f"./csv/elo_rating/{year}_elo_rating.csv", index=False)

        if not team_check:
            break


if __name__ == "__main__":
    main(verbose=0)

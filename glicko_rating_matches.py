"""Glicko Rating System about LoLesports
"""
import os
import sys
import math
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm


DEFAULT_POINT = 1500
DEFAULT_RD = 350
DEFAULT_SIGMA = 0.6
DEFAULT_TAU = 0.2
CONVERT_VALUE = 173.7178


class GlickoSystem:
    """Glicko System"""

    EPSILON = 0.000001

    def __init__(self, tau=DEFAULT_TAU):
        """GlickoSystem init.

        Args:
            tau (float, optional): Reasonable choices are between 0.3 and 1.2.
            Smaller values of tau prevent the volatility
            measures from changing by large amounts, which in turn prevent
            enormous changes in ratings based on very improbable results.
            Defaults to 0.3.
        """
        self.tau = tau

    def calculate_g(self, pi):
        """Calculate g(pi).

        Args:
            pi (float): Converted rating deviation.

        Returns:
            float: Calculated g(pi).
        """
        return 1 / math.sqrt(1 + 3 * pi**2 / math.pi**2)

    def calculate_E(self, mu=None, op_mu=None, op_pi=None, g=None):
        """Calculate E(mu, op_mu, op_pi).

        Args:
            mu (float, optional): Converted my rating point. Defaults to None.
            op_mu (float, optional): Converted opponent rating point. Defaults to None.
            op_pi (float, optional): Converted opponent rating deviation. Defaults to None.
            g (float, optional): Calculated g(op_pi). Defaults to None.

        Returns:
            float: Calculated E(mu, op_mu, op_pi).
        """
        assert mu is not None and op_mu is not None
        if g is None:
            assert op_pi is not None
            g = self.calculate_g(op_pi)
        return 1 / (1 + math.exp(-1 * g * (mu - op_mu)))

    def calculate_v(self, mu=None, op_mu=None, op_pi=None, g=None, E=None):
        """Calculate v. v is the estimated variance of the team's rating
        based only on game outcome.

        Args:
            mu (float, optional): Converted my rating point. Defaults to None.
            op_mu (float, optional): Converted opponent rating point. Defaults to None.
            op_pi (float, optional): Converted opponent rating deviation. Defaults to None.
            g (float, optional): Calculated g(op_pi). Defaults to None.
            E (float, optional): Calculated E(mu, op_mu, op_pi). Defaults to None.

        Returns:
            float: Calculated v.
        """
        if g is None:
            assert op_pi is not None
            g = self.calculate_g(op_pi)
        if E is None:
            assert mu is not None and op_mu is not None and op_pi is not None
            E = self.calculate_E(mu, op_mu, op_pi)
        return 1 / g**2 * E * (1 - E)

    def calculate_delta(
        self, outcome, mu=None, op_mu=None, op_pi=None, g=None, E=None, v=None
    ):
        """Calculate delta. delta is the estimated improvement in rating
        by comparing the pre-period rating to the performance rating
        based only on game outcome.

        Args:
            outcome (int): Outcome of match. 1 is win, 0 is loss.
            mu (float, optional): Converted my rating point. Defaults to None.
            op_mu (float, optional): Converted opponent rating point. Defaults to None.
            op_pi (float, optional): Converted opponent rating deviation. Defaults to None.
            g (float, optional): Calculated g(op_pi). Defaults to None.
            E (float, optional): Calculated E(mu, op_mu, op_pi). Defaults to None.
            v (float, optional): Calculated v. Defaults to None.

        Returns:
            float: Calculated delta.
        """
        if g is None:
            assert op_pi is not None
            g = self.calculate_g(op_pi)
        if E is None:
            assert mu is not None and op_mu is not None and op_pi is not None
            E = self.calculate_E(mu, op_mu, op_pi)
        if v is None:
            assert g is not None and E is not None
            v = self.calculate_v(g=g, E=E)
        return v * g * (outcome - E)

    def calculate_f(self, x, pi, sigma, v, delta):
        """Calculate f(x).

        Args:
            x (float): value of x.
            pi (float): Converted rating deviation.
            sigma (float): Volatility.
            v (float): Quantity v.
            delta (float): Quantity delta.

        Returns:
            float: calculated f(x).
        """
        e_x = math.exp(x)
        a = math.log(sigma**2)
        left = e_x * (delta**2 - pi**2 - v - e_x) / (2 * (pi**2 + v + e_x) ** 2)
        right = (x - a) / self.tau**2
        return left - right

    def calculate_new_sigma(self, team, v, delta):
        """Calculate new sigma.

        Args:
            team (Team): Team object.
            v (float): Quantity v.
            delta (float): Quantity delta.

        Returns:
            float: Calculated sigma.
        """
        pi = team.pi
        sigma = team.sigma
        a = math.log(sigma**2)

        A = a
        if delta**2 > pi**2 + v:
            B = math.log(delta**2 - pi**2 - v)
        else:
            k = 1
            while self.calculate_f(a - k * self.tau, pi, sigma, v, delta) < 0:
                k += 1
            B = a - k * self.tau

        f_A = self.calculate_f(A, pi, sigma, v, delta)
        f_B = self.calculate_f(B, pi, sigma, v, delta)
        while abs(B - A) > self.EPSILON:
            C = A + (A - B) * f_A / (f_B - f_A)
            f_C = self.calculate_f(C, pi, sigma, v, delta)
            if f_C * f_B <= 0:
                A, f_A = B, f_B
            else:
                f_A = f_A / 2
            B, f_B = C, f_C
        return math.exp(A / 2)

    def update_rating(self, team, op_mu, op_pi, outcome):
        """Update rating.

        Args:
            team (Team): Team object.
            op_mu (float): Converted opponent rating point.
            op_pi (float): Converted opponent rating deviation.
            outcome (int): Outcome. 1 is win, 0 is loss.
        """
        mu = team.mu
        pi = team.pi
        g = self.calculate_g(op_pi)
        E = self.calculate_E(mu, op_mu, g=g)
        v = self.calculate_v(g=g, E=E)
        delta = self.calculate_delta(outcome, g=g, E=E, v=v)

        new_sigma = self.calculate_new_sigma(team, v, delta)
        new_pi = math.sqrt(pi**2 + new_sigma**2)
        new_pi = 1 / math.sqrt(1 / new_pi**2 + 1 / v)
        new_mu = mu + new_pi**2 * g * (outcome - E)

        team.update(new_mu, new_pi, new_sigma)
        error = outcome - E
        team.update_error(error)

    def rate(self, team1, team2, outcome):
        """Rate.

        Args:
            team1 (Team): Team object.
            team2 (Team): Team object.
            outcome (int): Outcome. 1 is win, 0 is loss.
        """
        mu1 = team1.mu
        pi1 = team1.pi
        mu2 = team2.mu
        pi2 = team2.pi
        self.update_rating(team1, mu2, pi2, outcome)
        self.update_rating(team2, mu1, pi1, 1 - outcome)


class Team:
    """Team class."""

    def __init__(
        self,
        name,
        league,
        team_id,
        win=0,
        loss=0,
        streak=0,
        point=DEFAULT_POINT,
        rd=DEFAULT_RD,
        sigma=DEFAULT_SIGMA,
        error=0,
        error_square=0,
        last_game_date=None,
    ):
        """Team init.

        Args:
            name (str): Team name.
            league (str): League of team.
            win (int, optional): Number of win. Defaults to 0.
            loss (int, optional): Number of loss. Defaults to 0.
            streak (int, optional): Number of streak. Defaults to 0.
            point (_type_, optional): Number of point. Defaults to DEFAULT_POINT.
            rd (int, optional): Rating deviation. Defaults to 350.
            sigma (float, optional): Volatile. Defaults to 0.6.
            last_game_date (datetime, optional): Date of last game. Defaults to None.
        """
        self.name = name
        self.league = league
        self.team_id = team_id
        self.win = win
        self.loss = loss
        self.streak = streak
        self.point = point
        self.rd = rd
        self.sigma = sigma
        self.error = error
        self.error_square = error_square
        self.last_game_date = last_game_date

    @property
    def mu(self):
        """Convert point to mu

        Returns:
            float: Converted mu
        """
        return (self.point - DEFAULT_POINT) / CONVERT_VALUE

    @property
    def pi(self):
        """Convert rating deviation to pi

        Returns:
            float: Converted pi
        """
        return self.rd / CONVERT_VALUE

    @property
    def games(self):
        """Number of games.

        Returns:
            int: Number of games.
        """
        return self.win + self.loss

    @property
    def winrate(self):
        """Calculate winrate.

        Returns:
            float: Winrate.
        """
        return self.win / self.games if self.win > 0 else 0

    def update(self, mu, pi, sigma):
        """Update point, rd, sigma using mu, pi, sigma

        Args:
            mu (float): Converted point
            pi (_type_): Converted rating deviation
            sigma (_type_): Quantity volatile
        """
        self.point = CONVERT_VALUE * mu + DEFAULT_POINT
        self.rd = CONVERT_VALUE * pi
        self.sigma = sigma

    def update_error(self, error):
        """Update error.

        Args:
            error (float): Error.
        """
        self.error += abs(error)
        self.error_square += error**2

    def update_team_name(self, name):
        """Update team name.

        Args:
            name (str): Name of team.
        """
        self.name = name

    def update_league(self, league):
        """Update league.

        Args:
            league (str): League of team.
        """
        self.league = league

    def update_streak(self, outcome):
        """Update streak

        Args:
            outcome (int): Outcome of match. 1 is win, 0 is loss.
        """
        outcome = 1 if outcome == 1 else -1
        if self.streak * outcome > 0:
            self.streak += outcome
        else:
            self.streak = outcome

    def update_match(self, outcome, game_date):
        """Update match.

        Args:
            outcome (int): Outcome of match. 1 is win, 0 is loss.
            game_date (datetime): Date of match.
        """
        self.last_game_date = game_date
        if outcome == 1:
            self.win += 1
        else:
            self.loss += 1

    def init_rd(self):
        """Init rating deviation."""
        self.rd = DEFAULT_RD

    def init_sigma(self):
        """Init sigma."""
        self.sigma = DEFAULT_SIGMA

    def get_win_probability(self, glicko, op_team):
        """Calculate win probability against op_team.

        Args:
            glicko (GlickoSystem): GlickoSystem object.
            op_team (Team): Team object. Opponent team.

        Returns:
            float: Win probability.
        """
        return glicko.calculate_E(self.mu, op_team.mu, op_team.pi)

    def to_tuple(self):
        """Convert to tuple.

        Returns:
            tuple: Tuple(team_id, name, league, games, win, loss, winrate, streak,
            point, rd, sigma, error, error_square, last_game_date)
        """
        return (
            self.team_id,
            self.name,
            self.league,
            self.games,
            self.win,
            self.loss,
            self.winrate,
            self.streak,
            self.point,
            self.rd,
            self.sigma,
            self.error,
            self.error_square,
            self.last_game_date,
        )

    def __str__(self):
        """Convert to str.

        Returns:
            str: Information of Team object.
        """
        return f"point: {self.point}, rd: {self.rd}, sigma: {self.sigma}"


def get_team_id(teams_id, name):
    """Get id of team from teams_id

    Args:
        teams_id (DataFrame): DataFrame of teams_id.
        name (str): Name of team.

    Returns:
        int: Id of team.
    """
    return teams_id.loc[teams_id["team"] == name, "team_id"].iloc[0]


def is_proper_league(league):
    """Check league is proper league.

    Args:
        league (str): League.

    Returns:
        bool: True if league is proper league, else False.
    """
    if league in ["WCS", "MSI"]:
        return False
    return True


def proceed_rating(glicko, teams_id, teams, sg):
    """Proceed rating.

    Args:
        glicko (GlickoSystem): GlickoSystem object.
        teams_id (DataFrame): DataFrame of teams_id.
        teams (dict): Dictionary of Team object.
        sg (DataFrame): DataFrame of scoreboard games.
    """
    for row in sg.itertuples():
        team1, team2 = row.Team1, row.Team2
        game_date = row._6
        outcome = 1 if row.WinTeam == team1 else 0
        team1_id, team2_id = get_team_id(teams_id, team1), get_team_id(teams_id, team2)
        glicko.rate(teams[team1_id], teams[team2_id], outcome)
        teams[team1_id].update_match(outcome, game_date)
        teams[team2_id].update_match(1 - outcome, game_date)


def get_rating(teams):
    """Get rating.

    Args:
        teams (dict): Dictionary of teams.

    Returns:
        DataFrame: DataFrame of ratings.
    """
    data = np.array(
        list(map(lambda x: x.to_tuple(), teams.values())),
        dtype=[
            ("team_id", "int"),
            ("team", "object"),
            ("league", "object"),
            ("games", "int"),
            ("win", "int"),
            ("loss", "int"),
            ("winrate", "float"),
            ("streak", "int"),
            ("point", "float"),
            ("rd", "float"),
            ("sigma", "float"),
            ("error", "float"),
            ("error_square", "float"),
            ("last_game_date", "datetime64[ns]"),
        ],
    )
    ratings = pd.DataFrame.from_records(data)
    ratings["change"] = ""
    ratings = ratings.sort_values(by="point", ascending=False).reset_index(drop=True)
    return ratings


def update_changes(origin, ratings):
    if len(origin) == 0:
        origin = pd.DataFrame(columns=ratings.columns)

    for row in ratings.itertuples():
        df = origin.loc[origin["team_id"] == row.team_id]
        if len(df) > 0:
            standing_change = df.index[0] - row.Index
            point_change = round(row.point - df["point"].iloc[0])

            if standing_change == 0:
                standing_change = "-"
            else:
                standing_change = str(standing_change)
                if standing_change[0] != "-":
                    standing_change = "+" + standing_change
            if point_change == 0:
                point_change = "-"
            else:
                point_change = str(point_change)
                if point_change[0] != "-":
                    point_change = "+" + point_change
            ratings.loc[row.Index, "change"] = f"{standing_change} ({point_change})"
        else:
            ratings.loc[row.Index, "change"] = "new"

    return ratings


def parse_teams(file_path):
    last_ratings = pd.read_csv(file_path, parse_dates=["last_game_date"])

    teams = {}
    for row in last_ratings.itertuples():
        team_id = int(row.team_id)
        name = row.team
        league = row.league
        win = int(row.win)
        loss = int(row.loss)
        streak = int(row.streak)
        point = float(row.point)
        rd = float(row.rd)
        sigma = float(row.sigma)
        error = float(row.error)
        error_square = float(row.error_square)
        last_game_date = row.last_game_date
        teams[team_id] = Team(
            name,
            league,
            team_id,
            win,
            loss,
            streak,
            point,
            rd,
            sigma,
            error,
            error_square,
            last_game_date,
        )

    return teams


def rate_year_matches(teams_id, glicko, year: int):
    last_year_file_path = f"./csv/glicko_rating/glicko_rating_{year - 1}.csv"
    if year > 2011:
        if os.path.isfile(last_year_file_path):
            teams = parse_teams(last_year_file_path)
        else:
            teams = rate_year_matches(teams_id, glicko, year - 1)
    else:
        teams = {}

    tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
    scoreboard_games = pd.read_csv(
        f"./csv/scoreboard_games/{year}_scoreboard_games.csv",
        parse_dates=["DateTime UTC"],
    )
    file_path = f"./csv/glicko_rating/glicko_rating_{year}.csv"
    if os.path.isfile(file_path):
        origin = pd.read_csv(file_path)
    elif os.path.isfile(last_year_file_path):
        origin = pd.read_csv(last_year_file_path)
    else:
        origin = pd.DataFrame()

    print(f"{year} year")
    for page in tqdm(tournaments["OverviewPage"]):
        sg = scoreboard_games.loc[scoreboard_games["OverviewPage"] == page]
        league = sg["League"].iloc[0]
        team_names = sg[["Team1", "Team2"]].unstack().unique()

        for name in team_names:
            if name not in teams_id["team"].values:
                logging.error("%d year; %s tournament", year, page)
                logging.error("%s not in teams id", name)
                sys.exit(1)
            team_id = get_team_id(teams_id, name)
            if team_id not in teams:
                teams[team_id] = Team(name, league, team_id)
            else:
                teams[team_id].update_team_name(name)
                if is_proper_league(league):
                    teams[team_id].update_league(league)

        for name in team_names:
            team_id = get_team_id(teams_id, name)
            teams[team_id].init_rd()
            teams[team_id].init_sigma()

        proceed_rating(glicko, teams_id, teams, sg)

    ratings = get_rating(teams)
    ratings = update_changes(origin, ratings)
    ratings.to_csv(file_path, index=False)

    return teams


def main():
    """Rating using glicko-2 rating"""
    glicko = GlickoSystem()

    teams_id = pd.read_csv("./csv/teams_id.csv")

    year = 2023
    _ = rate_year_matches(teams_id, glicko, year)


if __name__ == "__main__":
    main()

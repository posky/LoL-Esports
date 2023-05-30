from itertools import product
from functools import reduce
from pprint import pprint

import pandas as pd
import numpy as np
from tqdm import tqdm

from lol_fandom import get_tournaments, get_match_schedule
from google_sheet import Sheet

with open("./sheet_id.txt", "r") as f:
    SHEET_ID = f.read()


class Head2Head:
    def __init__(self, win=0, loss=0, set_win=0, set_loss=0):
        self.win = win
        self.loss = loss
        self.set_win = set_win
        self.set_loss = set_loss

    def __str__(self):
        return f"({self.win}, {self.loss})"

    def __add__(self, other):
        return Head2Head(
            self.win + other.win,
            self.loss + other.loss,
            self.set_win + other.set_win,
            self.set_loss + other.set_loss,
        )


class Match:
    BEST_OF = {3: [(2, 0), (2, 1), (1, 2), (0, 2)]}

    def __init__(
        self, team1, team2, winner, team1_score, team2_score, is_tiebreaker, best_of
    ):
        self.team1 = team1
        self.team2 = team2
        self.winner = winner
        self.team1_score = team1_score
        self.team2_score = team2_score
        self.is_tiebreaker = is_tiebreaker
        self.best_of = best_of

        min_score = min(self.team1_score, self.team2_score)
        max_score = max(self.team1_score, self.team2_score)
        if not (min_score < max_score and best_of // 2 + 1 == max_score):
            self.team1_score = 0
            self.team2_score = 0
            self.winner = 0

    def set_match(self, idx):
        self.team1_score = self.BEST_OF[self.best_of][idx][0]
        self.team2_score = self.BEST_OF[self.best_of][idx][1]
        if self.team1_score > self.team2_score:
            self.winner = 1
        else:
            self.winner = 2

    def init_match(self):
        self.team1_score = 0
        self.team2_score = 0
        self.winner = 0

    def __str__(self):
        return f"{self.team1} {self.team1_score} vs {self.team2_score} {self.team2}"


class Team:
    def __init__(self, name):
        self.name = name
        self.win = 0
        self.loss = 0
        self.set_win = 0
        self.set_loss = 0
        self.streak = 0
        self.standing = 0
        self.matches = []
        self.proceeded_matches = []
        self.head_to_head = {}

    @property
    def games(self):
        return self.win + self.loss

    @property
    def set_games(self):
        return self.set_win + self.set_loss

    @property
    def point(self):
        return self.set_win - self.set_loss

    @property
    def winrate(self):
        return self.win / self.games if self.win > 0 else 0

    @property
    def set_winrate(self):
        return self.set_win / self.set_games if self.set_win > 0 else 0

    def init_head_to_head(self, team_names):
        for name in team_names:
            if name != self.name:
                self.head_to_head[name] = Head2Head()

    def add_match(self, match):
        self.matches.append(match)

    def _update_match(self, match):
        if self.name == match.team1:
            if match.winner == 1:
                self.win += 1
            else:
                self.loss += 1
            self.set_win += match.team1_score
            self.set_loss += match.team2_score
        else:
            if match.winner == 1:
                self.loss += 1
            else:
                self.win += 1
            self.set_win += match.team2_score
            self.set_loss += match.team1_score

    def _rollback_match(self, match):
        if self.name == match.team1:
            if match.winner == 1:
                self.win -= 1
            else:
                self.loss -= 1
            self.set_win -= match.team1_score
            self.set_loss -= match.team2_score
        else:
            if match.winner == 1:
                self.loss -= 1
            else:
                self.win -= 1
            self.set_win -= match.team2_score
            self.set_loss -= match.team1_score

    def update_head_to_head(self, match):
        if match.team1 == self.name:
            if match.winner == 1:
                self.head_to_head[match.team2].win += 1
            else:
                self.head_to_head[match.team2].loss += 1
            self.head_to_head[match.team2].set_win += match.team1_score
            self.head_to_head[match.team2].set_loss += match.team2_score
        else:
            if match.winner == 2:
                self.head_to_head[match.team1].win += 1
            else:
                self.head_to_head[match.team1].loss += 1
            self.head_to_head[match.team1].set_win += match.team2_score
            self.head_to_head[match.team1].set_loss += match.team1_score

    def rollback_head_to_head(self, match):
        if match.team1 == self.name:
            if match.winner == 1:
                self.head_to_head[match.team2].win -= 1
            else:
                self.head_to_head[match.team2].loss -= 1
            self.head_to_head[match.team2].set_win -= match.team1_score
            self.head_to_head[match.team2].set_loss -= match.team2_score
        else:
            if match.winner == 2:
                self.head_to_head[match.team1].win -= 1
            else:
                self.head_to_head[match.team1].loss -= 1
            self.head_to_head[match.team1].set_win -= match.team2_score
            self.head_to_head[match.team1].set_loss -= match.team1_score

    def update_match(self, match):
        self.proceeded_matches.append(match)
        self._update_match(match)
        self.update_head_to_head(match)

    def update_rest_match(self, match):
        self._update_match(match)
        self.update_head_to_head(match)

    def rollback_rest_match(self, match):
        self._rollback_match(match)
        self.rollback_head_to_head(match)

    def __str__(self):
        return f"{self.name}: {self.games} | {self.win} - {self.loss} | {self.set_win} - {self.set_loss} | {self.point}"

    def __lt__(self, other):
        if self.win != other.win:
            return self.win > other.win
        if self.loss != other.loss:
            return self.loss < other.loss
        if self.point != other.point:
            return self.point > other.point
        return False

    def __eq__(self, other):
        if self.win != other.win:
            return False
        if self.loss != other.loss:
            return False
        if self.set_win != other.set_win:
            return False
        if self.set_loss != other.set_loss:
            return False
        return True


class League:
    def __init__(self, name):
        self.name = name
        self.teams = {}
        self.matches = []
        self.proceeded_matches = []
        self.rest_matches = []

    def add_team(self, name):
        self.teams[name] = Team(name)

    def add_match(self, match):
        assert isinstance(match, Match)

        self.matches.append(match)
        self.teams[match.team1].add_match(match)
        self.teams[match.team2].add_match(match)

    def update_match(self, match):
        self.proceeded_matches.append(match)
        self.teams[match.team1].update_match(match)
        self.teams[match.team2].update_match(match)

    def update_rest_match(self, match):
        self.teams[match.team1].update_rest_match(match)
        self.teams[match.team2].update_rest_match(match)

    def rollback_rest_match(self, match):
        self.teams[match.team1].rollback_rest_match(match)
        self.teams[match.team2].rollback_rest_match(match)

    def init_head_to_head(self):
        for team in self.teams.values():
            team.init_head_to_head(self.teams.keys())

    def proceed_matches(self):
        for match in self.matches:
            if match.winner != 0:
                self.update_match(match)
            else:
                self.rest_matches.append(match)

    def rank(self):
        standings = sorted(self.teams.values())
        for i, team in enumerate(standings, start=1):
            team.standing = i
        return standings

    def tiebreak(self, teams):
        if len(teams) == 2:
            head_to_head = teams[0].head_to_head[teams[1].name]
            if head_to_head.win > head_to_head.loss:
                teams[1].standing = teams[0].standing + 1
            elif head_to_head.win < head_to_head.loss:
                teams[0].standing = teams[1].standing + 1
        elif len(teams) == 3:
            head_to_head_lst = []
            for i, team in enumerate(teams):
                op_teams = teams[:i] + teams[i + 1 :]
                head_to_head = (
                    team.head_to_head[op_teams[0].name]
                    + team.head_to_head[op_teams[1].name]
                )
                head_to_head_lst.append(head_to_head)
            lst = sorted(
                zip(teams, head_to_head_lst), key=lambda x: x[1].win, reverse=True
            )
            if lst[0][1].win == 3 and lst[1][1].win == 3 and lst[2][1].win == 0:
                lst[2][0].standing = lst[0][0].standing + 2
            elif lst[0][1].win == 4 and lst[1][1].win == 1 and lst[2][1].win == 1:
                lst[1][0].standing = lst[0][0].standing + 1
                lst[2][0].standing = lst[0][0].standing + 1
            elif lst[0][1].win == 4 and lst[1][1].win == 2 and lst[2][1].win == 0:
                lst[1][0].standing = lst[0][0].standing + 1
                lst[2][0].standing = lst[0][0].standing + 2

    def simulate_rest_matches(self):
        teams_standings = {}
        for team in self.teams.values():
            teams_standings[team.name] = [
                0 for _ in range(len(self.teams.values()) + 1)
            ]

        best_of = self.matches[0].best_of
        num_cases = len(self.matches[0].BEST_OF[best_of])
        num = num_cases ** len(self.rest_matches)
        with tqdm(total=num) as pbar:
            for idx in product("0123", repeat=len(self.rest_matches)):
                for i, match in zip(idx, self.rest_matches):
                    match.set_match(int(i))
                    self.update_rest_match(match)

                standings = self.rank()
                lst = [standings[0]]
                for i, team in enumerate(standings[1:], start=1):
                    if standings[i - 1] == team:
                        lst.append(team)
                        team.standing = lst[0].standing
                    elif len(lst) > 2:
                        self.tiebreak(lst)
                        lst = [team]
                    else:
                        lst = [team]
                if len(lst) > 2:
                    self.tiebreak(lst)

                standings = sorted(standings, key=lambda x: (x.standing, x.name))
                for team in standings:
                    teams_standings[team.name][team.standing] += 1

                for match in self.rest_matches:
                    self.rollback_rest_match(match)
                    match.init_match()
                pbar.update(1)

        return teams_standings


class LeagueLPL(League):
    def tiebreak(self, teams):
        if len(teams) == 2:
            head_to_head = teams[0].head_to_head[teams[1].name]
            if head_to_head.win > head_to_head.loss:
                teams[1].standing = teams[0].standing + 1
            else:
                teams[0].standing = teams[1].standing + 1


def update_matches_to_csv(page_name):
    match_schedules = get_match_schedule(where=f'MS.OverviewPage="{page_name}"')
    match_schedules.sort_values(by=["DateTime UTC"], ignore_index=True, inplace=True)
    columns = [
        "Team1",
        "Team2",
        "Winner",
        "Team1Score",
        "Team2Score",
        "IsTiebreaker",
        "BestOf",
    ]
    matches = match_schedules[columns].fillna(0)
    columns = ["Winner", "Team1Score", "Team2Score", "IsTiebreaker", "BestOf"]
    matches[columns] = matches[columns].astype("int")
    matches.to_csv("./csv/match_schedule/target_matches_schedule.csv", index=False)


def select_page():
    tournaments = get_tournaments(
        where=f'T.Year=2023 and T.TournamentLevel="Primary" and T.IsPlayoffs=0'
    )
    tournaments = tournaments.loc[
        tournaments["League"].isin(
            [
                "LoL Champions Korea",
                "League of Legends Championship Series",
                "LoL EMEA Championship",
                "Tencent LoL Pro League",
            ]
        )
    ].sort_values(["League", "SplitNumber"], ignore_index=True)

    pages = tournaments["OverviewPage"].values
    for i, page in enumerate(pages):
        print(f"{i}: {page}")
    number = int(input("Input page number: "))

    assert 0 <= number < len(pages)

    return pages[number]


def main():
    page = select_page()
    update_matches_to_csv(page)
    matches = pd.read_csv("./csv/match_schedule/target_matches_schedule.csv")

    for i in range(80, 90):
        matches.loc[i, ["Winner", "Team1Score", "Team2Score"]] = [0, 0, 0]

    league = League(page.split("/")[0])
    team_names = matches[["Team1", "Team2"]].unstack().unique()
    for name in team_names:
        league.add_team(name)

    league.init_head_to_head()

    for row in matches.itertuples():
        match = Match(
            row.Team1,
            row.Team2,
            row.Winner,
            row.Team1Score,
            row.Team2Score,
            row.IsTiebreaker,
            row.BestOf,
        )
        league.add_match(match)

    league.proceed_matches()

    standings = league.rank()
    for team in standings:
        print(team)

    teams_standings = league.simulate_rest_matches()
    teams_standings = sorted(
        teams_standings.items(), key=lambda x: tuple(x[1]), reverse=True
    )

    sheet = Sheet(SHEET_ID)
    sheet.connect_sheet()

    values = []
    value = ["Team"] + [i for i in range(1, len(teams_standings) + 1)]
    values.append(value)
    for team, standing in teams_standings:
        value = [team] + standing[1:]
        values.append(value)

    sheet.write_sheet(
        f"simulations!R1C1:C{len(teams_standings) + 1}", {"values": values}
    )


if __name__ == "__main__":
    main()

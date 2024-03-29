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
from itertools import product

import pandas as pd
import numpy as np
from tqdm import tqdm

from lol_fandom import get_tournaments, get_match_schedule

pd.set_option("display.max_columns", None)

# %%
tournaments = get_tournaments(
    where='T.TournamentLevel="Primary" and T.IsOfficial=1 and T.Year=2023 and T.Region="Korea"'
)
tournaments

# %%
# page_name = "LCK/2023 Season/Spring Season"
page_name = "LCK/2023 Season/Summer Season"
page_name

# %%
match_schedules = get_match_schedule(where=f'MS.OverviewPage="{page_name}"')
match_schedules.sort_values(by=["DateTime UTC"], ignore_index=True, inplace=True)
match_schedules

# %%
matches = match_schedules[
    ["Team1", "Team2", "Winner", "Team1Score", "Team2Score", "IsTiebreaker", "BestOf"]
].fillna(0)
matches

# %%
matches[['Winner', 'Team1Score', 'Team2Score', 'IsTiebreaker', 'BestOf']] = matches[['Winner', 'Team1Score', 'Team2Score', 'IsTiebreaker', 'BestOf']].astype('int')
matches

# %%
matches.dtypes

# %%
# for i in range(80, 90):
#     matches.loc[i, ['Winner', 'Team1Score', 'Team2Score']] = [0, 0, 0]

# %%
matches


# %%
# matches.to_csv(f"./csv/{page_name.replace('/', '_')}_matches.csv", index=False)

# %%
class Head2Head:
    def __init__(self):
        self.win = 0
        self.loss = 0
        self.set_win = 0
        self.set_loss = 0

    def __str__(self):
        return f"({self.win}, {self.loss})"


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

            print(f"{teams[0].name} vs {teams[1].name}: {head_to_head}")

        elif len(teams) == 3:
            head_to_head_lst = [Head2Head() for _ in range(3)]
            for i, (team, head_to_head) in enumerate(zip(teams, head_to_head_lst)):
                op_teams = teams[:i] + teams[i + 1 :]
                head_to_head.win = (
                    team.head_to_head[op_teams[0].name].win
                    + team.head_to_head[op_teams[1].name].win
                )
                head_to_head.loss = (
                    team.head_to_head[op_teams[0].name].loss
                    + team.head_to_head[op_teams[1].name].loss
                )
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

            for team, head_to_head in lst:
                print(f"{team.standing}: {team} | {head_to_head}")
            print()

    def simulate_rest_matches(self):
        num = 4 ** len(self.rest_matches)
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

                standings = sorted(standings, key=lambda x: x.standing)

                for match in self.rest_matches:
                    self.rollback_rest_match(match)
                    match.init_match()
                pbar.update(1)


# %%
league = League('LCK')
team_names = matches[['Team1', 'Team2']].unstack().unique()
for name in team_names:
    league.add_team(name)

# %%
league.teams

# %%
league.init_head_to_head()

# %%
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

# %%
league.matches, len(league.matches)

# %%
league.proceed_matches()

# %%
for team in league.teams.values():
    print(team)

# %%
standings = league.rank()
for team in standings:
    print(team)

# %%
league.simulate_rest_matches()

# %%

# %%

# %%
import numpy as np
import pandas as pd

from lol_fandom import get_tournaments

# %%
tournaments = get_tournaments(where=f'T.Year=2023 and T.TournamentLevel="Primary" and T.IsPlayoffs=0')
tournaments

# %%
tournaments['League'].unique()

# %%
tournaments.loc[tournaments['League'].isin(['LoL Champions Korea', 'League of Legends Championship Series', 'LoL EMEA Championship', 'Tencent LoL Pro League'])].sort_values(['League', 'SplitNumber'])

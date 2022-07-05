MATCH_BO3 = [(2, 0,), (2, 1,), (1, 2,), (0, 2,)]

# Match class
class Match:
    def __init__(self, team1, team2, winner, team1_score, team2_score, num):
        self.team1 = team1
        self.team2 = team2
        self.winner = winner
        self.team1_score = team1_score
        self.team2_score = team2_score
        self.num = num

    def set_seq(self, seq):
        team1_score, team2_score = MATCH_BO3[seq]
        self.team1_score = team1_score
        self.team2_score = team2_score
        self.winner = seq // 2 + 1

    def init_seq(self):
        self.winner = -1
        self.team1_score = 0
        self.team2_score = 0

# Team class
class Team:
    def __init__(self, name, league):
        self.name = name
        self.league = league
        self.win = 0
        self.loss = 0
        self.set_win = 0
        self.set_loss = 0
        self.matches = []

    @property
    def point(self):
        return self.set_win - self.set_loss

    @property
    def streak(self):
        #
        return 0

    def _match_result(self, match):
        if match.winner == 1:
            if match.team1 == self.name:
                self.win += 1
                self.set_win += match.team1_score
                self.set_loss += match.team2_score
            else:
                self.loss += 1
                self.set_win += match.team2_score
                self.set_loss += match.team1_score
        elif match.winner == 2:
            if match.team1 == self.name:
                self.loss += 1
                self.set_win += match.team1_score
                self.set_loss += match.team2_score
            else:
                self.win += 1
                self.set_win += match.team2_score
                self.set_loss += match.team1_score

    def add_match(self, match):
        self.matches.append(match)
        self._match_result(match)

    def set_match(self, match):
        self._match_result(match)

    def undo_match(self, match):
        if match.winner == 1:
            if match.team1 == self.name:
                self.win -= 1
                self.set_win -= match.team1_score
                self.set_loss -= match.team2_score
            else:
                self.loss -= 1
                self.set_win -= match.team2_score
                self.set_loss -= match.team1_score
        elif match.winner == 2:
            if match.team1 == self.name:
                self.loss -= 1
                self.set_win -= match.team1_score
                self.set_loss -= match.team2_score
            else:
                self.win -= 1
                self.set_win -= match.team2_score
                self.set_loss -= match.team1_score

    def __lt__(self, other):
        if self.win != other.win:
            return self.win < other.win
        if self.loss != other.loss:
            return self.loss > other.loss
        return self.point < other.point

    def __eq__(self, other):
        if self.win == other.win and self.loss == other.loss and self.point == other.point:
            return True
        return False



# League class
class League:
    def __init__(self, name):
        self.name = name
        self.teams = {}
        self.matches = []

    def add_team(self, team_name):
        if team_name not in self.teams:
            self.teams[team_name] = Team(team_name, self)

    def add_teams(self, team_names):
        for name in team_names:
            self.add_team(name)

    def add_match(self, match_info):
        match = Match(
            match_info['team1'],
            match_info['team2'],
            match_info['winner'],
            match_info['team1_score'],
            match_info['team2_score'],
            match_info['num']
        )
        self.matches.append(match)
        self.teams[match.team1].add_match(match)
        self.teams[match.team2].add_match(match)

    def add_matches(self, matches):
        for match in matches:
            self.add_match(match)

    def set_match(self, match, seq):
        match.set_seq(seq)
        self.teams[match.team1].set_match(match)
        self.teams[match.team2].set_match(match)

    def undo_match(self, match):
        if match.winner != -1:
            self.teams[match.team1].undo_match(match)
            self.teams[match.team2].undo_match(match)
            match.init_seq()

    def rank(self):
        standings = sorted(self.teams.values(), reverse=True)
        return standings

    def simulate(self):
        standings = {}
        for team in self.teams:
            standings[team] = [0 for _ in range(len(self.teams) + 1)]

        rest_matches = []
        for match in self.matches:
            if match.winner == -1:
                rest_matches.append(match)

        full_case = pow(4, len(rest_matches))
        cur_case = 0

        stack = [(0, -1)] if len(rest_matches) > 0 else []  # (number of rest_matches, seq)
        while len(stack) > 0:
            i, seq = stack.pop()
            match = rest_matches[i]
            self.undo_match(match)

            seq += 1
            print(cur_case / full_case * 100, end='\r')
            while seq < 4:
                match = rest_matches[i]
                self.set_match(match, seq)
                stack.append((i, seq,))
                if i == len(rest_matches) - 1:
                    cur_case += 1
                    standing = self.rank()
                    pre_pos = 1
                    pre_team = None
                    for pos, team in enumerate(standing, start=1):
                        if pos > 1 and pre_team == team:
                            pos = pre_pos
                        standings[team.name][pos] += 1
                        pre_pos = pos
                        pre_team = team
                    break
                i += 1
                seq = 0

        return standings

def main():
    pass


if __name__ == '__main__':
    main()
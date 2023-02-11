import math

import pandas as pd

from lol_fandom import get_leagues

pd.set_option('display.max_columns', None)


class Elo:
    K = 20

    def __init__(self):
        self.point = 1500
        self.win = 0
        self.loss = 0
        self.streak = 0
        self.last_game_date = None

    def update_win_loss(self, result):
        if result == 1:
            self.win += 1
        else:
            self.loss += 1

    def update_streak(self, result):
        result = 1 if result == 1 else -1
        if (self.streak > 0) == (result > 0):
            self.streak += result
        else:
            self.streak = result

    def update_last_game_date(self, last_game_date):
        self.last_game_date = last_game_date

    def get_win_prob(self, other):
        assert isinstance(other, Elo)

        return 1 / (pow(10, (other.point - self.point) / 400) + 1)

    def update_point(self, other, result):
        assert isinstance(other, Elo)

        self.update_win_loss(result)
        self.update_streak(result)
        self.point = self.point + self.K * (result - self.get_win_prob(other))


class Team(Elo):
    def __init__(self, name, league=None):
        super().__init__()
        self.name = name
        self.league = league
        self.roster = {}

    def update_team_name(self, name):
        self.name = name

    def update_league(self, league):
        self.league = league

    def update_roster(self, players, roster):
        changed = []
        new_roster = {}
        for id in roster:
            if id not in self.roster.keys():
                changed.append(id)
            new_roster[id] = players[id]
        self.roster = new_roster
        for player in self.roster.values():
            player.update_team(self)

    def update_streak(self, result):
        result = 1 if result == 1 else -1
        if (self.streak > 0) == (result > 0):
            self.streak += result
        else:
            self.streak = result

    def update_last_game_date(self, last_game_date):
        self.last_game_date = last_game_date

    def get_rating(self, players):
        assert len(players) > 0

        points = 0
        for id in players:
            points += self.roster[id].point
        return points / len(players)

    def get_expectation(self, my_rating, op_rating):
        return 1 / (1 + pow(10, (op_rating - my_rating) / 400))

    def get_change(self, result, my_rating, op_rating):
        return result - self.get_expectation(my_rating, op_rating)

    @classmethod
    def update_point(cls, team1, team2, result, players1_id, players2_id):
        team1_rating = team1.get_rating(players1_id)
        team2_rating = team2.get_rating(players2_id)

        team1_change = team1.get_change(result, team1_rating, team2_rating)
        team2_change = team2.get_change(1 - result, team2_rating, team1_rating)

        for id, player in team1.roster.items():
            played = False
            if id in players1_id:
                played = True
            player.update_point(played, result, team2_rating, team1_change)
        for id, player in team2.roster.items():
            played = False
            if id in players2_id:
                played = True
            player.update_point(played, 1 - result, team1_rating, team2_change)
        super().update_point(team1, team2, result)
        super().update_point(team2, team1, 1 - result)

    def init_player_point(self):
        new_players = []
        points = 0
        old_player_count = 0
        for player in self.roster.values():
            if player.point is None:
                new_players.append(player)
            else:
                points += player.point
                old_player_count += 1
        point = points / old_player_count if old_player_count > 0 else 0
        if len(new_players) > 0:
            if len(new_players) > old_player_count:
                point = self.point
            for player in new_players:
                player.point = point

    def to_dict(self):
        data = {
            'Team': self.name,
            'League': self.league,
            'Win': self.win,
            'Loss': self.loss,
            'WinRate': self.win / (self.win + self.loss) if self.win != 0 else 0,
            'Streak': self.streak,
            'Point': self.point,
            'last_game_date': self.last_game_date,
        }

        return data


class Player(Elo):
    def __init__(self, name):
        self.name = name
        self.team = None
        self.point = None
        self.k = 40
        self.q = 1
        self.win = 0
        self.loss = 0
        self.streak = 0
        self.last_game_date = None

    def update_player_name(self, name):
        self.name = name

    def update_team(self, team):
        self.team = team
        self.k = 40
        self.q = 1

    def update_point(self, played, result, opponent_rating, team_change):
        if played:
            self.update_win_loss(result)
            self.update_streak(result)

            self.point = self.point + self.k * ((self.q * self.get_change(result, opponent_rating)) + ((1 - self.q) * team_change))
            self.k = self.k - 0.25 if self.k - 0.25 > 24 else 24
            self.q = self.q - 0.025 if self.q - 0.025 > 0.5 else 0.5
        else:
            self.k = self.k + 0.5 if self.k < 40 else 40
            self.q = self.q + 0.025 if self.q + 0.025 < 1 else 1

    def get_expectation(self, opponent_rating):
        return 1 / (1 + pow(10, (opponent_rating - self.point) / 400))

    def get_change(self, result, opponent_rating):
        return result - self.get_expectation(opponent_rating)

    def to_dict(self):
        data = {
            'Player': self.name,
            'League': self.team.league,
            'Team': self.team.name,
            'Win': self.win,
            'Loss': self.loss,
            'WinRate': self.win / (self.win + self.loss) if self.win != 0 else 0,
            'Streak': self.streak,
            'Point': self.point,
            'last_game_date': self.last_game_date,
        }

        return data



def get_team_id(teams_id, name):
    if isinstance(name, str):
        return teams_id.loc[teams_id['team'] == name, 'team_id'].iloc[0]
    lst = []
    for t in name:
        lst.append(teams_id.loc[teams_id['team'] == t, 'team_id'].iloc[0])
    return list(set(lst))

def get_player_id(players_id, name):
    if isinstance(name, str):
        return players_id.loc[players_id['player'] == name, 'player_id'].iloc[0]
    lst = []
    for p in name:
        lst.append(players_id.loc[players_id['player'] == p, 'player_id'].iloc[0])
    return list(set(lst))

def is_proper_league(leagues, league):
    return leagues.loc[leagues['League Short'] == league, 'Region'].iloc[0] != 'International'

def get_roster(roster):
    return list(map(lambda x: x.strip(), roster.split(';;')))

def split_string(string, delimiter=';;'):
    if isinstance(string, str):
        return list(map(lambda x: x.strip(), string.split(delimiter)))
    return []

def get_player_from_roster(players_id, str_roster, str_roles):
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    team_roster = split_string(str_roster)
    player_roles = split_string(str_roles)
    player_lst = []
    for i in range(len(team_roster)):
        role = split_string(player_roles[i], ',')
        is_player = False
        for position in role:
            if position.lower() in roles:
                is_player = True
                break
        if is_player:
            player_lst.append(get_player_id(players_id, team_roster[i]))
    return list(set(player_lst))

def get_all_rosters(tournament_rosters):
    roles = ['top', 'jungle', 'mid', 'bot', 'support']
    players = []
    for row in tournament_rosters.itertuples():
        team_roster = split_string(row.RosterLinks)
        player_roles = split_string(row.Roles)
        player_lst = []
        for i in range(len(team_roster)):
            role = split_string(player_roles[i], ',')
            is_player = False
            for position in role:
                if position.lower() in roles:
                    is_player = True
                    break
            if is_player:
                player_lst.append(team_roster[i])
        players += player_lst
    return list(set(players))

def proceed_rating(
    teams_id, players_id, teams, players,
    scoreboard_games, scoreboard_players, tournament_rosters
    ):
    roles = ['top', 'jungle', 'mid', 'bot', 'support']

    teams_id_lst = list(set(
        get_team_id(
            teams_id, list(scoreboard_games[['Team1', 'Team2']].unstack().unique())
        ) +
        get_team_id(teams_id, tournament_rosters['Team'].values)
    ))
    league_teams = {team_id: [] for team_id in teams_id_lst}

    # team roster update
    for row in tournament_rosters.itertuples():
        team_id = get_team_id(teams_id, row.Team)
        player_lst = get_player_from_roster(players_id, row.RosterLinks, row.Roles)
        league_teams[team_id] += player_lst
    for team_name in scoreboard_games[['Team1', 'Team2']].unstack().unique():
        team_id = get_team_id(teams_id, team_name)
        player_lst = get_player_id(
            players_id,
            scoreboard_players.loc[scoreboard_players['Team'] == team_name, 'Link'].unique()
        )
        league_teams[team_id] += player_lst
    for team_id, player_lst in league_teams.items():
        team = teams[team_id]
        player_lst = list(set(player_lst))
        team.update_roster(players, player_lst)
        team.init_player_point()

    for row in scoreboard_games.itertuples():
        team1_name, team2_name = row.Team1, row.Team2
        game_date = row._6
        game_id = row.GameId
        result = 1 if row.WinTeam == team1_name else 0
        team1 = teams[get_team_id(teams_id, team1_name)]
        team2 = teams[get_team_id(teams_id, team2_name)]

        sp = scoreboard_players.loc[scoreboard_players['GameId'] == game_id]
        players1 = sp.loc[sp['Team'] == team1_name, 'Link'].values
        players2 = sp.loc[sp['Team'] == team2_name, 'Link'].values
        players1_id = get_player_id(players_id, players1)
        players2_id = get_player_id(players_id, players2)

        Team.update_point(team1, team2, result, players1_id, players2_id)
        team1.update_last_game_date(game_date)
        team2.update_last_game_date(game_date)
        for player_id in players1_id + players2_id:
            players[player_id].update_last_game_date(game_date)

def get_rating(lst):
    ratings = pd.DataFrame(data=map(lambda x: x.to_dict(), lst))
    ratings = ratings.sort_values(by='Point', ascending=False).reset_index(drop=True)
    return ratings



def main():
    leagues = get_leagues(where=f'L.Level="Primary" and L.IsOfficial="Yes"')

    teams_id = pd.read_csv('./csv/teams_id.csv')
    players_id = pd.read_csv('./csv/players_id.csv')

    teams = {}
    players = {}
    for year in range(2011, 2024):
        tournaments = pd.read_csv(f'./csv/tournaments/{year}_tournaments.csv')
        scoreboard_games = pd.read_csv(
            f'./csv/scoreboard_games/{year}_scoreboard_games.csv'
        )
        scoreboard_players = pd.read_csv(
            f'./csv/scoreboard_players/{year}_scoreboard_players.csv'
        )
        tournament_rosters = pd.read_csv(
            f'./csv/tournament_rosters/{year}_tournament_rosters.csv'
        )
        for page in tournaments['OverviewPage']:
            print(f'{page} rating ...')
            sg = scoreboard_games.loc[scoreboard_games['OverviewPage'] == page]
            sp = scoreboard_players.loc[scoreboard_players['OverviewPage'] == page]
            tr = tournament_rosters.loc[tournament_rosters['OverviewPage'] == page]
            print(
                f'\t{sg.shape[0]} scoreboard matches | ' +
                f'{sp.shape[0]} scoreboard players | {tr.shape[0]} teams roster'
            )

            league = sg['League'].iloc[0]
            team_names = list(set(
                list(sg[['Team1', 'Team2']].unstack().unique()) +
                list(tr['Team'].unique())
            ))
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
                    if teams[id].league is None or is_proper_league(leagues, league):
                        teams[id].update_league(league)
            if not team_check:
                break
            player_names = list(set(
                list(sp['Link'].unique()) + get_all_rosters(tr)
            ))
            player_check = True
            for name in player_names:
                if name not in players_id['player'].values:
                    print(f'{name} not in players')
                    player_check = False
                    break
                id = get_player_id(players_id, name)
                if id not in players:
                    players[id] = Player(name)
                else:
                    players[id].update_player_name(name)
            if not player_check:
                break

            proceed_rating(teams_id, players_id, teams, players, sg, sp, tr)

        if not team_check or not player_check:
            break
    if team_check and player_check:
        rating_teams = get_rating(teams.values())
        rating_players = get_rating(players.values())

        rating_teams.to_csv('./csv/team_elo_rating.csv', index=False)
        rating_players.to_csv('./csv/player_elo_rating.csv', index=False)



if __name__ == '__main__':
    main()
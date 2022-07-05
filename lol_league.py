from optparse import Values
from pprint import pprint

from use_mwclient import get_league_schedules
from google_sheet import Sheet
from lol_esports import League


SPREADSHEET_ID = '1HrZwRZEoGeydW9HgNHuRyie3Z9tGyAlQ_hBoiyS0vVY'
SCHEDULE_RANGE = 'LCK Schedule!A2:E'
STANDING_RANGE = 'LCK Standings!A2:H'
SIMULATION_INIT_RANGE = 'LCK Simulation!A1:A'


def init_sheet(sheet, init_range, num_col):
    rows = sheet.read_rows_sheet(init_range)
    num_rows = len(rows)

    value_input_option = 'USER_ENTERED'
    values = [['' for _ in range(num_col)] for _ in range(num_rows)]
    value_range_body = {
        'values': values
    }

    sheet.write_sheet(
        sheet_range = init_range,
        value_input_option = value_input_option,
        value_range_body = value_range_body
    )

def init_league_schedule(sheet):
    init_sheet(sheet, SCHEDULE_RANGE, 5)

def update_league_schedule(sheet, schedules):
    value_input_option = 'USER_ENTERED'
    values = schedules
    value_range_body = {
        'values': values
    }

    sheet.write_sheet(
        sheet_range = SCHEDULE_RANGE,
        value_input_option = value_input_option,
        value_range_body = value_range_body
    )

def read_league_schedule(sheet):
    rows = sheet.read_rows_sheet(SCHEDULE_RANGE)
    teams = []
    matches = []
    for n, row in enumerate(rows, start=1):
        teams += [row[0], row[1]]
        if len(row) < 3:
            row += ['-1', '0', '0']
        match = {
            'team1': row[0],
            'team2': row[1],
            'winner': int(row[2]),
            'team1_score': int(row[3]),
            'team2_score': int(row[4]),
            'num': n
        }
        matches.append(match)

    teams = list(set(teams))
    return (teams, matches,)

def write_league_standings(sheet, league):
    init_sheet(sheet, STANDING_RANGE, 8)

    standings = league.rank()

    value_input_option = 'USER_ENTERED'
    values = []
    for pos, team in enumerate(standings, start=1):
        value = [
            pos,
            team.name,
            '{0}-{1}'.format(team.win, team.loss),
            team.win / (team.win + team.loss) if team.win != 0 else 0,
            '{0}-{1}'.format(team.set_win, team.set_loss),
            team.set_win / (team.set_win + team.set_loss) if team.set_win != 0 else 0,
            team.point,
            team.streak
        ]
        if pos > 1 and standings[pos - 2] == team:
            value[0] = values[pos - 2][0]
        values.append(value)
    value_range_body = {
        'values': values
    }

    sheet.write_sheet(
        sheet_range = STANDING_RANGE,
        value_input_option = value_input_option,
        value_range_body = value_range_body
    )

def init_league_simulation(sheet):
    rows = sheet.read_rows_sheet(SIMULATION_INIT_RANGE)
    num_rows = len(rows)

    value_input_option = 'USER_ENTERED'
    values = [['' for _ in range(num_rows)] for _ in range(num_rows)]
    value_range_body = {
        'values': values
    }

    sheet.write_sheet(
        sheet_range = SIMULATION_INIT_RANGE,
        value_input_option = value_input_option,
        value_range_body = value_range_body
    )


def main():
    sheet = Sheet(SPREADSHEET_ID)
    sheet.connect_sheet()

    # Update league match schedules
    match_schedules = get_league_schedules()
    init_league_schedule(sheet)
    update_league_schedule(sheet, match_schedules)


    # read league matches
    teams, matches = read_league_schedule(sheet)
    league = League('LCK')
    league.add_teams(teams)
    league.add_matches(matches)

    # write current league standings
    write_league_standings(sheet, league)

    # simulate rest matches
    standings = league.simulate()

    init_league_simulation(sheet)

    for team, standing in standings.items():
        print('{0}: {1} / {2}'.format(team, sum(standing), standing))


if __name__ == '__main__':
    main()
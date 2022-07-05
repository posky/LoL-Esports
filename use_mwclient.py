from pprint import pprint

import mwclient


SITE = mwclient.Site('lol.fandom.com', path='/')


def league_schedules_to_list(schedules):
    # Team1, Team2, Winner, Team1Score, Team2Score
    match_schedules = list(map(
        lambda x: [
            x['title']['Team1'],
            x['title']['Team2'],
            x['title']['Winner'],
            x['title']['Team1Score'],
            x['title']['Team2Score']
        ], schedules))
    return match_schedules

def get_league_schedules():
    response = SITE.api('cargoquery',
        limit = 'max',
        tables = 'Leagues=L, Tournaments=T',
        join_on = 'L.League=T.League',
        fields = 'T.OverviewPage, T.Split, T.SplitNumber, T.Year',
        where = 'L.League_Short="LCK" and T.Year=2022 and T.SplitNumber=2 and T.IsPlayoffs=false'
    )

    tournament_name = response['cargoquery'][0]['title']['OverviewPage']

    response = SITE.api('cargoquery',
        limit = 'max',
        tables = 'Tournaments=T, MatchSchedule=MS',
        join_on = 'T.OverviewPage=MS.OverviewPage',
        fields = 'MS.Team1, MS.Team2, MS.Winner, MS.OverviewPage, MS.DateTime_UTC, MS.Team1Score, MS.Team2Score, MS.BestOf',
        where = 'T.OverviewPage="{}"'.format(tournament_name)
    )

    match_schedules = sorted(response['cargoquery'], key=lambda x: x['title']['DateTime UTC'])
    match_schedules = league_schedules_to_list(match_schedules)

    return match_schedules


def main():
    match_schedules = get_league_schedules()


if __name__ == '__main__':
    main()
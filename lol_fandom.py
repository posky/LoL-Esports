import time

import pandas as pd
import numpy as np
import mwclient

SITE = mwclient.Site('lol.fandom.com', path='/')
DEFAULT_DELAY = 1
last_query = time.time()


def set_default_delay(delay):
    global DEFAULT_DELAY

    DEFAULT_DELAY = delay

def delay_between_query():
    global last_query

    delay = DEFAULT_DELAY - (time.time() - last_query)
    if delay > 0:
        time.sleep(delay)
    last_query = time.time()

def from_response(response):
    return pd.DataFrame([l['title'] for l in response['cargoquery']])

def get_leagues(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Leagues=L',
        fields='L.League, L.League_Short, L.Region, L.Level, L.IsOfficial',
        where=where
    )
    return from_response(response)

def get_tournaments(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Leagues=L, Tournaments=T',
        join_on='L.League=T.League',
        fields='T.Name, T.OverviewPage, T.DateStart, T.Date, T.League, ' +
            'T.Region, T.EventType, T.StandardName, T.Split, T.SplitNumber, ' +
            'T.TournamentLevel, T.IsQualifier, T.IsPlayoffs, T.IsOfficial, T.Year',
        where=where
    )
    df = from_response(response)
    if len(df) > 0:
        df['DateStart'] = pd.to_datetime(df['DateStart'])
        df['Date'] = pd.to_datetime(df['Date'])
    return df

def get_scoreboard_games(where='', casting=True):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Tournaments=T, ScoreboardGames=SG',
        join_on='T.OverviewPage=SG.OverviewPage',
        fields='SG.OverviewPage, SG.Team1, SG.Team2, SG.WinTeam, SG.LossTeam, ' +
            'SG.DateTime_UTC, SG.Team1Score, SG.Team2Score, SG.Winner, ' +
            'SG.Gamelength, SG.Gamelength_Number, SG.Team1Bans, SG.Team2Bans, ' +
            'SG.Team1Picks, SG.Team2Picks, SG.Team1Players, SG.Team2Players, ' +
            'SG.Team1Dragons, SG.Team2Dragons, SG.Team1Barons, SG.Team2Barons, ' +
            'SG.Team1Towers, SG.Team2Towers, SG.Team1Gold, SG.Team2Gold, ' +
            'SG.Team1Kills, SG.Team2Kills, SG.Team1RiftHeralds, SG.Team2RiftHeralds, ' +
            'SG.Team1Inhibitors, SG.Team2Inhibitors, SG.Patch, SG.GameId, ' +
            'SG.MatchId, SG.RiotGameId',
        where=where
    )
    int_types = [
        'Team1Score', 'Team2Score', 'Winner'
    ]
    float_types = ['Gamelength Number', 'Team1Dragons', 'Team2Dragons',
        'Team1Barons', 'Team2Barons', 'Team1Towers', 'Team2Towers',
        'Team1Gold', 'Team2Gold', 'Team1Kills', 'Team2Kills',
        'Team1RiftHeralds', 'Team2RiftHeralds',
        'Team1Inhibitors', 'Team2Inhibitors'
    ]
    datetime_type = 'DateTime UTC'

    df = from_response(response)
    if df['OverviewPage'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    if casting:
        df[datetime_type] = pd.to_datetime(df[datetime_type])
        df[int_types] = df[int_types].astype('int')
        df[float_types] = df[float_types].astype('float')
    return df

def get_scoreboard_players(where='', casting=True):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Tournaments=T, ScoreboardPlayers=SP',
        join_on='T.OverviewPage=SP.OverviewPage',
        fields='SP.OverviewPage, SP.Name, SP.Link, SP.Champion, SP.Kills, SP.Deaths, ' +
            'SP.Assists, SP.SummonerSpells, SP.Gold, SP.CS, SP.DamageToChampions, ' +
            'SP.VisionScore, SP.Items, SP.Trinket, SP.KeystoneMastery, ' +
            'SP.KeystoneRune, SP.PrimaryTree, SP.SecondaryTree, SP.Runes, ' +
            'SP.TeamKills, SP.TeamGold, SP.Team, SP.TeamVs, SP.Time, SP.PlayerWin, ' +
            'SP.DateTime_UTC, SP.DST, SP.Tournament, SP.Role, SP.Role_Number, ' +
            'SP.IngameRole, SP.Side, SP.UniqueLine, SP.UniqueLineVs, SP.UniqueRole, ' +
            'SP.UniqueRoleVs, SP.GameId, SP.MatchId, SP.GameTeamId, SP.GameRoleId, ' +
            'SP.GameRoleIdVs, SP.StatsPage',
        where=where
    )

    int_types = [
        'Kills', 'Deaths', 'Assists'
    ]
    float_types = [
        'Gold', 'CS', 'DamageToChampions', 'VisionScore',
        'TeamKills', 'TeamGold', 'Role Number', 'Side'
    ]
    datetime_type = 'DateTime UTC'

    df = from_response(response)
    if len(df) == 0 or df['OverviewPage'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    if casting:
        df[datetime_type] = pd.to_datetime(df[datetime_type])
        df[int_types] = df[int_types].astype('int')
        df[float_types] = df[float_types].astype('float')
    return df

def get_tournament_rosters(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Tournaments=T, TournamentRosters=TR',
        join_on='T.OverviewPage=TR.OverviewPage',
        fields='TR.Team, TR.OverviewPage, TR.Region, TR.RosterLinks, ' +
            'TR.Roles, TR.Flags, TR.Footnotes, TR.IsUsed, TR.Tournament, ' +
            'TR.Short, TR.IsComplete, TR.PageAndTeam, TR.UniqueLine',
        where=where
    )

    df = from_response(response)
    if len(df) == 0 or df['OverviewPage'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

def get_player_redirects(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='PlayerRedirects=PR',
        fields='PR.AllName, PR.OverviewPage, PR.ID',
        where=where,
    )

    df = from_response(response)
    if len(df) == 0 or df['OverviewPage'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

def get_teams(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Teams=T',
        fields='T.Name, T.OverviewPage, T.Short, T.Location, T.TeamLocation, ' +
            'T.Region, T.OrganizationPage, T.Image, T.Twitter, T.Youtube, ' +
            'T.Facebook, T.Instagram, T.Discord, T.Snapchat, T.Vk, ' +
            'T.Subreddit, T.Website, T.RosterPhoto, T.IsDisbanded, ' +
            'T.RenamedTo, T.IsLowercase',
        where=where,
    )

    df = from_response(response)
    if len(df) == 0 or df['OverviewPage'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

def get_tournament_results(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='TournamentResults=TR',
        fields='TR.Event, TR.Tier, TR.Date, TR.RosterPage, TR.Place, ' +
            'TR.ForceNewPlace, TR.Place_Number, TR.Qualified, TR.Prize, ' +
            'TR.Prize_USD, TR.Prize_Euro, TR.PrizeUnit, TR.Prize_Markup, ' +
            'TR.PrizeOther, TR.Phase, TR.Team, TR.IsAchievement, ' +
            'TR.LastResult, TR.LastTeam, TR.LastOpponent_Markup, ' +
            'TR.GroupName, TR.LastOutcome, TR.PageAndTeam, '
            'TR.OverviewPage, TR.UniqueLine',
        where=where,
    )

    df = from_response(response)
    if len(df) == 0 or df['Team'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

def get_team_redirects(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='TeamRedirects=TR',
        fields='TR.AllName, TR.OtherName, TR.UniqueLine',
        where=where,
    )

    df = from_response(response)
    if len(df) == 0 or df['AllName'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

def get_match_schedule(where=''):
    delay_between_query()

    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='MatchSchedule=MS',
        fields='MS.Team1, MS.Team2, MS.Team1Final, MS.Team2Final, MS.Winner, ' +
            'MS.Team1Points, MS.Team2Points, MS.Team1PointsTB, ' +
            'MS.Team2PointsTB, MS.Team1Score, MS.Team2Score, ' +
            'MS.Team1Poster, MS.Team2Poster, MS.Team1Advantage, ' +
            'MS.Team2Advantage, MS.FF, MS.IsNullified, MS.Player1, ' +
            'MS.Player2, MS.MatchDay, MS.DateTime_UTC, MS.HasTime, ' +
            'MS.DST, MS.IsFlexibleStart, MS.IsReschedulable, ' +
            'MS.OverrideAllowPredictions, MS.OverrideDisallowPredictions, ' +
            'MS.IsTiebreaker, MS.OverviewPage, MS.ShownName, MS.ShownRound, ' +
            'MS.BestOf, MS.Round, MS.Phase, MS.N_MatchInPage, MS.Tab, ' +
            'MS.N_MatchInTab, MS.N_TabInPage, MS.N_Page, MS.Patch, ' +
            'MS.PatchPage, MS.Hotfix, MS.DisabledChampions, ' +
            'MS.PatchFootnote, MS.InitialN_MatchInTab, MS.InitialPageAndTab, ' +
            'MS.GroupName, MS.Stream, MS.StreamDisplay, MS.Venue, ' +
            'MS.CastersPBP, MS.CastersColor, MS.Casters, MS.MVP, ' +
            'MS.MVPPoints, MS.VodInterview, MS.VodHighlights, ' +
            'MS.InterviewWith, MS.Recap, MS.Reddit, MS.QQ, MS.Wanplus, ' +
            'MS.WanplusId, MS.PageAndTeam1, MS.PageAndTeam2, ' +
            'MS.Team1Footnote, MS.Team2Footnote, MS.Footnote, ' +
            'MS.UniqueMatch, MS.MatchId',
        where=where,
    )

    df = from_response(response)
    if len(df) == 0 or df['Team1'].iloc[0] is None:
        return None

    df.replace([None], np.nan, inplace=True)
    return df

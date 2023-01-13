import pandas as pd
import numpy as np
import mwclient

SITE = mwclient.Site('lol.fandom.com', path='/')

def from_response(response):
    return pd.DataFrame([l['title'] for l in response['cargoquery']])

def get_leagues(where=''):
    response = SITE.api(
        'cargoquery',
        limit='max',
        tables='Leagues=L',
        fields='L.League, L.League_Short, L.Region, L.Level, L.IsOfficial',
        where=where
    )
    return from_response(response)

def get_tournaments(where=''):
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

    df.replace([None], np.nan)
    if casting:
        df[datetime_type] = pd.to_datetime(df[datetime_type])
        df[int_types] = df[int_types].astype('int')
        df[float_types] = df[float_types].astype('float')
    return df

def get_scoreboard_players(where=''):
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
        'Kills', 'Deaths', 'Assists', 'Gold', 'CS', 'DamageToChampions',
        'VisionScore', 'TeamKills', 'TeamGold', 'Role Number', 'Side',
    ]
    datetime_type = 'DateTime UTC'

    df = from_response(response)
    df[int_types] = df[int_types].astype('int')
    df[datetime_type] = pd.to_datetime(df[datetime_type])
    return df

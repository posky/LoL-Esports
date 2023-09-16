"""lol.fandom.com API."""
import time

import mwclient
import numpy as np
import pandas as pd

SITE = mwclient.Site("lol.fandom.com", path="/")

DEFAULT_DELAY = 1

last_query = time.time()


def set_default_delay(delay: int) -> None:
    """Set the default delay for the program.

    Args:
        delay (int): The new default delay value.


    Returns:
        None
    """
    global DEFAULT_DELAY

    DEFAULT_DELAY = delay


def delay_between_query() -> None:
    """Delays the execution of queries.

    This function calculates the delay between consecutive queries based on the current

    time and the time of the last query. If the calculated delay is greater than zero,

    the function sleeps for that amount of time. After the delay, the function updates

    the last_query variable with the current time.


    Args:
        None


    Returns:
        None
    """
    global last_query

    delay = DEFAULT_DELAY - (time.time() - last_query)

    if delay > 0:
        time.sleep(delay)

    last_query = time.time()


def from_response(response: dict) -> pd.DataFrame:
    """Generates a Pandas DataFrame from a given response dictionary.

    Args:
        response (dict): The response dictionary containing the data.


    Returns:
        pd.DataFrame: The generated DataFrame containing the titles from the response.
    """
    return pd.DataFrame([row["title"] for row in response["cargoquery"]])


def get_leagues(where: str = "") -> pd.DataFrame:
    """Retrieves a DataFrame of leagues based on the specified conditions.

    Args:
        where (str, optional): A condition for filtering the leagues. Defaults to "".

    Returns:
        pd.DataFrame: A DataFrame containing the leagues data.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Leagues=L",
        fields="L.League, L.League_Short, L.Region, L.Level, L.IsOfficial",
        where=where,
    )

    return from_response(response)


def get_tournaments(where: str = "") -> pd.DataFrame:
    """Retrieves tournaments based on the specified condition.

    Args:
        where (str): A SQL-like condition to filter the tournaments. (default: "")

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved tournaments.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Leagues=L, Tournaments=T",
        join_on="L.League=T.League",
        fields=(
            "T.Name, T.OverviewPage, T.DateStart, T.Date, T.League, T.Region,"
            " T.EventType, T.StandardName, T.Split, T.SplitNumber, T.TournamentLevel,"
            " T.IsQualifier, T.IsPlayoffs, T.IsOfficial, T.Year"
        ),
        where=where,
    )

    tournaments = from_response(response)

    if len(tournaments) > 0:
        tournaments["DateStart"] = pd.to_datetime(tournaments["DateStart"])

        tournaments["Date"] = pd.to_datetime(tournaments["Date"])
    return tournaments


def get_scoreboard_games(where: str = "", *, casting: bool = True) -> pd.DataFrame:
    """Retrieves the scoreboard games from the API based on the specified parameters.

    Args:
        where (str): The additional query parameter to filter the results (default: "")
        casting (bool): Whether to perform casting of data types (default: True)

    Returns:
        pd.DataFrame: The dataframe containing the retrieved scoreboard games data.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Tournaments=T, ScoreboardGames=SG",
        join_on="T.OverviewPage=SG.OverviewPage",
        fields=(
            "SG.OverviewPage, SG.Team1, SG.Team2, SG.WinTeam, SG.LossTeam,"
            " SG.DateTime_UTC, SG.Team1Score, SG.Team2Score, SG.Winner, SG.Gamelength,"
            " SG.Gamelength_Number, SG.Team1Bans, SG.Team2Bans, SG.Team1Picks,"
            " SG.Team2Picks, SG.Team1Players, SG.Team2Players, SG.Team1Dragons,"
            " SG.Team2Dragons, SG.Team1Barons, SG.Team2Barons, SG.Team1Towers,"
            " SG.Team2Towers, SG.Team1Gold, SG.Team2Gold, SG.Team1Kills, SG.Team2Kills,"
            " SG.Team1RiftHeralds, SG.Team2RiftHeralds, SG.Team1Inhibitors,"
            " SG.Team2Inhibitors, SG.Patch, SG.GameId,SG.MatchId, SG.RiotGameId"
        ),
        where=where,
    )

    int_types = ["Team1Score", "Team2Score", "Winner"]

    float_types = [
        "Gamelength Number",
        "Team1Dragons",
        "Team2Dragons",
        "Team1Barons",
        "Team2Barons",
        "Team1Towers",
        "Team2Towers",
        "Team1Gold",
        "Team2Gold",
        "Team1Kills",
        "Team2Kills",
        "Team1RiftHeralds",
        "Team2RiftHeralds",
        "Team1Inhibitors",
        "Team2Inhibitors",
    ]

    datetime_type = "DateTime UTC"

    scoreboard_games = from_response(response)

    if scoreboard_games["OverviewPage"].iloc[0] is None:
        return None

    scoreboard_games = scoreboard_games.replace([None], np.nan)

    if casting:
        scoreboard_games[datetime_type] = pd.to_datetime(
            scoreboard_games[datetime_type],
        )

        scoreboard_games[int_types] = scoreboard_games[int_types].astype("int")

        scoreboard_games[float_types] = scoreboard_games[float_types].astype("float")
    return scoreboard_games


def get_scoreboard_players(where: str = "", *, casting: bool = True) -> pd.DataFrame:
    """Retrieves the scoreboard players based on the given criteria.

    Args:
        where (str): The criteria to filter the scoreboard players. Defaults to "".
        casting (bool): Determines whether to cast the data types of certain columns.
            Defaults to True.

    Returns:
        pd.DataFrame: The dataframe containing the scoreboard players matching
            the criteria. Returns None if no players are found.
    """
    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Tournaments=T, ScoreboardPlayers=SP",
        join_on="T.OverviewPage=SP.OverviewPage",
        fields=(
            "SP.OverviewPage, SP.Name, SP.Link, SP.Champion, SP.Kills, SP.Deaths,"
            " SP.Assists, SP.SummonerSpells, SP.Gold, SP.CS, SP.DamageToChampions,"
            " SP.VisionScore, SP.Items, SP.Trinket, SP.KeystoneMastery,"
            " SP.KeystoneRune, SP.PrimaryTree, SP.SecondaryTree, SP.Runes,"
            " SP.TeamKills, SP.TeamGold, SP.Team, SP.TeamVs, SP.Time, SP.PlayerWin,"
            " SP.DateTime_UTC, SP.DST, SP.Tournament, SP.Role, SP.Role_Number,"
            " SP.IngameRole, SP.Side, SP.UniqueLine, SP.UniqueLineVs, SP.UniqueRole,"
            " SP.UniqueRoleVs, SP.GameId, SP.MatchId, SP.GameTeamId, SP.GameRoleId,"
            " SP.GameRoleIdVs, SP.StatsPage"
        ),
        where=where,
    )

    int_types = ["Kills", "Deaths", "Assists"]

    float_types = [
        "Gold",
        "CS",
        "DamageToChampions",
        "VisionScore",
        "TeamKills",
        "TeamGold",
        "Role Number",
        "Side",
    ]

    datetime_type = "DateTime UTC"

    scoreboard_players = from_response(response)

    if (
        len(scoreboard_players) == 0
        or scoreboard_players["OverviewPage"].iloc[0] is None
    ):
        return None

    scoreboard_players = scoreboard_players.replace([None], np.nan)

    if casting:
        scoreboard_players[datetime_type] = pd.to_datetime(
            scoreboard_players[datetime_type],
        )

        scoreboard_players[int_types] = scoreboard_players[int_types].astype("int")

        scoreboard_players[float_types] = scoreboard_players[float_types].astype(
            "float",
        )
    return scoreboard_players


def get_tournament_rosters(where: str = "") -> pd.DataFrame:
    """Retrieves tournament rosters based on the given condition.

    Args:
        where (str): A condition to filter the rosters. Defaults to an empty string.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the tournament rosters.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Tournaments=T, TournamentRosters=TR",
        join_on="T.OverviewPage=TR.OverviewPage",
        fields=(
            "TR.Team, TR.OverviewPage, TR.Region, TR.RosterLinks, TR.Roles, TR.Flags,"
            " TR.Footnotes, TR.IsUsed, TR.Tournament, TR.Short, TR.IsComplete,"
            " TR.PageAndTeam, TR.UniqueLine"
        ),
        where=where,
    )

    tournament_rosters = from_response(response)

    if (
        len(tournament_rosters) == 0
        or tournament_rosters["OverviewPage"].iloc[0] is None
    ):
        return None

    return tournament_rosters.replace([None], np.nan)


def get_player_redirects(where: str = "") -> pd.DataFrame:
    """Get player redirects from the API.

    Args:
        where (str, optional): The condition to filter the redirects. Defaults to "".

    Returns:
        pd.DataFrame: A DataFrame containing the player redirects with columns: AllName,
        OverviewPage, ID. If no redirects are found or the OverviewPage is None,
        returns None.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="PlayerRedirects=PR",
        fields="PR.AllName, PR.OverviewPage, PR.ID",
        where=where,
    )

    player_redirects = from_response(response)

    if len(player_redirects) == 0 or player_redirects["OverviewPage"].iloc[0] is None:
        return None

    return player_redirects.replace([None], np.nan)


def get_teams(where: str = "") -> pd.DataFrame:
    """Retrieves teams from the API based on specified criteria.

    Args:
        where (str, optional): A string specifying the criteria for filtering teams.
        Defaults to "".

    Returns:
        pd.DataFrame: A pandas DataFrame containing the retrieved teams information.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="Teams=T",
        fields=(
            "T.Name, T.OverviewPage, T.Short, T.Location, T.TeamLocation, T.Region,"
            " T.OrganizationPage, T.Image, T.Twitter, T.Youtube, T.Facebook,"
            " T.Instagram, T.Discord, T.Snapchat, T.Vk, T.Subreddit, T.Website,"
            " T.RosterPhoto, T.IsDisbanded, T.RenamedTo, T.IsLowercase"
        ),
        where=where,
    )

    teams = from_response(response)

    if len(teams) == 0 or teams["OverviewPage"].iloc[0] is None:
        return None

    return teams.replace([None], np.nan)


def get_tournament_results(where: str = "") -> pd.DataFrame:
    """Retrieves tournament results based on the specified condition.

    Args:
        where (str, optional): A condition to filter the results. Defaults to "".

    Returns:
        pd.DataFrame: A DataFrame containing the tournament results, or None if no
            results are found.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="TournamentResults=TR",
        fields=(
            "TR.Event, TR.Tier, TR.Date, TR.RosterPage, TR.Place, TR.ForceNewPlace,"
            " TR.Place_Number, TR.Qualified, TR.Prize, TR.Prize_USD, TR.Prize_Euro,"
            " TR.PrizeUnit, TR.Prize_Markup, TR.PrizeOther, TR.Phase, TR.Team,"
            " TR.IsAchievement, TR.LastResult, TR.LastTeam, TR.LastOpponent_Markup,"
            " TR.GroupName, TR.LastOutcome, TR.PageAndTeam, TR.OverviewPage,"
            " TR.UniqueLine"
        ),
        where=where,
    )

    tournament_results = from_response(response)

    if len(tournament_results) == 0 or tournament_results["Team"].iloc[0] is None:
        return None

    return tournament_results.replace([None], np.nan)


def get_team_redirects(where: str = "") -> pd.DataFrame:
    """Retrieves the team redirects from the database based on the given filter criteria.

    Args:
        where (str, optional): The filter criteria to apply to the query. Defaults to "".

    Returns:
        pd.DataFrame or None: The DataFrame containing the team redirects if found,
            else None.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="TeamRedirects=TR",
        fields="TR.AllName, TR.OtherName, TR.UniqueLine",
        where=where,
    )

    team_redirects = from_response(response)

    if len(team_redirects) == 0 or team_redirects["AllName"].iloc[0] is None:
        return None

    return team_redirects.replace([None], np.nan)


def get_match_schedule(where: str = "") -> pd.DataFrame:
    """Retrieves the match schedule from the API.

    Args:
        where (str, optional): A filter condition to apply to the query. Defaults to "".

    Returns:
        pd.DataFrame: The match schedule data as a pandas DataFrame.
            Returns None if no data is available.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="MatchSchedule=MS",
        fields=(
            "MS.Team1, MS.Team2, MS.Team1Final, MS.Team2Final, MS.Winner,"
            " MS.Team1Points, MS.Team2Points, MS.Team1PointsTB, MS.Team2PointsTB,"
            " MS.Team1Score, MS.Team2Score, MS.Team1Poster, MS.Team2Poster,"
            " MS.Team1Advantage, MS.Team2Advantage, MS.FF, MS.IsNullified, MS.Player1,"
            " MS.Player2, MS.MatchDay, MS.DateTime_UTC, MS.HasTime, MS.DST,"
            " MS.IsFlexibleStart, MS.IsReschedulable, MS.OverrideAllowPredictions,"
            " MS.OverrideDisallowPredictions, MS.IsTiebreaker, MS.OverviewPage,"
            " MS.ShownName, MS.ShownRound, MS.BestOf, MS.Round, MS.Phase,"
            " MS.N_MatchInPage, MS.Tab, MS.N_MatchInTab, MS.N_TabInPage, MS.N_Page,"
            " MS.Patch, MS.PatchPage, MS.Hotfix, MS.DisabledChampions,"
            " MS.PatchFootnote, MS.InitialN_MatchInTab, MS.InitialPageAndTab,"
            " MS.GroupName, MS.Stream, MS.StreamDisplay, MS.Venue, MS.CastersPBP,"
            " MS.CastersColor, MS.Casters, MS.MVP, MS.MVPPoints, MS.VodInterview,"
            " MS.VodHighlights, MS.InterviewWith, MS.Recap, MS.Reddit, MS.QQ,"
            " MS.Wanplus, MS.WanplusId, MS.PageAndTeam1, MS.PageAndTeam2,"
            " MS.Team1Footnote, MS.Team2Footnote, MS.Footnote, MS.UniqueMatch,"
            " MS.MatchId"
        ),
        where=where,
    )

    match_schedule = from_response(response)

    if len(match_schedule) == 0 or match_schedule["Team1"].iloc[0] is None:
        return None

    return match_schedule.replace([None], np.nan)

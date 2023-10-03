"""lol.fandom.com API."""
import time

import mwclient
import polars as pl

pl.Config(fmt_str_lengths=100)

SITE = mwclient.Site("lol.fandom.com", path="/")


class Config:
    """Config class."""

    DELAY = 1
    last_query = time.time()


def set_query_delay(delay: int) -> None:
    """Set the delay for executing queries.

    Args:
        delay (int): The delay time in milliseconds.

    Returns:
        None
    """
    Config.DELAY = delay


def delay_between_query() -> None:
    """Delays the execution of queries to ensure a minimum time interval between them.

    This function calculates the remaining time until the next query can be executed
    based on the configured delay interval and the time elapsed since the last query.
    If the remaining time is greater than zero, the function pauses the execution for
    that duration using the `time.sleep()` function. After the delay,
    the `last_query` timestamp in the `Config` object is updated.

    Args:
        None

    Returns:
        None
    """
    delay = Config.DELAY - (time.time() - Config.last_query)

    if delay > 0:
        time.sleep(delay)

    Config.last_query = time.time()


def from_response(response: dict) -> pl.DataFrame:
    """Generate a polars DataFrame from the given response dictionary.

    Args:
        response (dict): The dictionary containing the response data.

    Returns:
        pl.DataFrame: The generated polars DataFrame.
    """
    if len(response["cargoquery"]) > 0:
        columns = response["cargoquery"][0]["title"].keys()
        return pl.DataFrame(
            [row["title"] for row in response["cargoquery"]],
            schema={key: pl.Utf8 for key in columns},
        )
    return pl.DataFrame()


def get_leagues(where: str = "") -> pl.DataFrame:
    """Retrieves a DataFrame of leagues from the API.

    Args:
        where (str, optional): A condition to filter the leagues. Defaults to "".

    Returns:
        DataFrame: A DataFrame containing the leagues.
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


def get_tournaments(where: str = "") -> pl.DataFrame:
    """Retrieves tournaments based on the provided filter criteria.

    Args:
        where (str, optional): A filter to apply on the tournaments. Defaults to "".

    Returns:
        pl.DataFrame: A DataFrame containing the retrieved tournaments.
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

    if tournaments.shape[0] > 0:
        datetime_cols = ["DateStart", "Date"]
        tournaments.with_columns(
            pl.col(datetime_cols).str.strptime(pl.Datetime, "%Y-%m-%d"),
        )
    return tournaments


def get_scoreboard_games(where: str = "") -> pl.DataFrame:
    """Retrieves the scoreboard games from the API.

    Args:
        where (str, optional): A string specifying the filter criteria for
        the scoreboard games. Defaults to "".

    Returns:
        pl.DataFrame: A polars DataFrame containing the retrieved scoreboard games.
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

    scoreboard_games = from_response(response)

    if scoreboard_games.shape[0] > 0:
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
        datetime_type = ["DateTime UTC"]

        scoreboard_games = scoreboard_games.with_columns(
            pl.col(int_types).cast(pl.Int32),
        )
        scoreboard_games = scoreboard_games.with_columns(
            pl.col(float_types).cast(pl.Float32),
        )
        scoreboard_games = scoreboard_games.with_columns(
            pl.col(datetime_type).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        )
    return scoreboard_games


def get_scoreboard_players(where: str = "") -> pl.DataFrame:
    """Returns a dataframe containing scoreboard player data.

    Parameters:
        where (str): A string representing the condition to filter the data.
            (default: "")

    Returns:
        pl.DataFrame: A pandas-like dataframe containing the scoreboard player data.

    """
    delay_between_query()

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

    scoreboard_players = from_response(response)

    if scoreboard_players.shape[0] > 0:
        int_types = [
            "Kills",
            "Deaths",
            "Assists",
            "Gold",
            "CS",
            "DamageToChampions",
            "VisionScore",
            "TeamKills",
            "TeamGold",
            "Role Number",
            "Side",
        ]
        datetime_type = ["DateTime UTC"]

        scoreboard_players = scoreboard_players.with_columns(
            [pl.col(int_types).cast(pl.Int32)],
        )
        scoreboard_players = scoreboard_players.with_columns(
            pl.col(datetime_type).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        )
    return scoreboard_players


def get_tournament_rosters(where: str = "") -> pl.DataFrame:
    """Fetches tournament rosters based on the provided filter.

    Args:
        where (str, optional): A filter to apply to the tournament rosters.
        Defaults to "".

    Returns:
        pl.DataFrame: A DataFrame containing the tournament rosters.
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

    return from_response(response)


def get_player_redirects(where: str = "") -> pl.DataFrame:
    """Retrieves player redirects from the API based on the specified condition.

    Args:
        where (str, optional): The condition to filter the redirects. Defaults to "".

    Returns:
        pl.DataFrame: The player redirects data.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="PlayerRedirects=PR",
        fields="PR.AllName, PR.OverviewPage, PR.ID",
        where=where,
    )

    return from_response(response)


def get_teams(where: str = "") -> pl.DataFrame:
    """Retrieves teams from the API based on the given query parameters.

    Args:
        where (str, optional): The query parameter to filter the teams. Defaults to "".

    Returns:
        pl.DataFrame: A DataFrame containing the retrieved teams.
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

    return from_response(response)


def get_tournament_results(where: str = "") -> pl.DataFrame:
    """Retrieves tournament results based on the specified query.

    Args:
        where (str, optional): A string specifying the query filter. Defaults to "".

    Returns:
        pl.DataFrame: A pandas DataFrame containing the tournament results.
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

    return from_response(response)


def get_team_redirects(where: str = "") -> pl.DataFrame:
    """Retrieves team redirects from the API based on the specified filter.

    Args:
        where (str): A filter to apply to the query. Defaults to an empty string.

    Returns:
        pl.DataFrame: A polars DataFrame containing the retrieved team redirects.
    """
    delay_between_query()

    response = SITE.api(
        "cargoquery",
        limit="max",
        tables="TeamRedirects=TR",
        fields="TR.AllName, TR.OtherName, TR.UniqueLine",
        where=where,
    )

    return from_response(response)


def get_match_schedule(where: str = "") -> pl.DataFrame:
    """Retrieves the match schedule data from the API.

    Args:
        where (str, optional): The condition to filter the match schedule data.
        Defaults to "".

    Returns:
        pl.DataFrame: The match schedule data as a Pandas DataFrame.
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

    return from_response(response)

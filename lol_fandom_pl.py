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
    return pl.DataFrame([row["title"] for row in response["cargoquery"]])


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

    return tournaments.with_columns(
        pl.col("DateStart").str.strptime(pl.Datetime, "%Y-%m-%d"),
        pl.col("Date").str.strptime(pl.Datetime, "%Y-%m-%d"),
    )


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

    scoreboard_games = scoreboard_games.with_columns([pl.col(int_types).cast(pl.Int32)])
    scoreboard_games = scoreboard_games.with_columns(
        pl.col(float_types).cast(pl.Float32),
    )
    return scoreboard_games.with_columns(
        pl.col(datetime_type).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    )


if __name__ == "__main__":
    pass

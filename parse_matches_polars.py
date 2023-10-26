"""Parse lolesports matches."""
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from tqdm import tqdm

from lol_fandom_pl import Leaguepedia


def write_parquet(df: pl.DataFrame, path: str) -> None:
    """Writes a pandas DataFrame to a Parquet file.

    Args:
        df (pl.DataFrame): The DataFrame to be written.
        path (str): The path to the Parquet file.

    Returns:
        None
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(file_path)


def parse_tournaments(start: int = 2011, end: int | None = None) -> None:
    """Parses tournaments.

    Args:
        start (int): The start year of the tournaments to parse. Defaults to 2011.
        end (int | None): The end year of the tournaments to parse.
            Defaults to None, which means the current year.

    Returns:
        None
    """
    end = end or datetime.datetime.now(tz=ZoneInfo("Asia/Seoul")).year
    leaguepedia = Leaguepedia()
    leagues = leaguepedia.get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    for year in tqdm(range(start, end + 1), desc="Years"):
        tournaments = pl.DataFrame()
        for (league,) in tqdm(
            leagues.select(pl.col("League")).rows(),
            desc="Leagues",
        ):
            tournament = leaguepedia.get_tournaments(
                tables=["leagues"],
                join_on="L.League=T.League",
                where=f'L.League="{league}" and T.Year={year}',
            )
            if not tournament.is_empty():
                tournaments = pl.concat([tournaments, tournament])
        write_parquet(tournaments, f"./parquet/tournaments/{year}_tournaments.parquet")


def parse_scoreboard_games(start: int = 2011, end: int | None = None) -> None:
    """Parses the scoreboard games for a given range of years.

    Args:
        start (int, optional): The starting year to parse scoreboard games from.
            Defaults to 2011.
        end (int | None, optional): The ending year to parse scoreboard games until.
            If None, the current year is used. Defaults to None.

    Returns:
        None
    """
    end = end or datetime.datetime.now(tz=ZoneInfo("Asia/Seoul")).year
    leaguepedia = Leaguepedia()
    for year in tqdm(range(start, end + 1), desc="Years"):
        tournaments = pl.read_parquet(
            f"./parquet/tournaments/{year}_tournaments.parquet",
        )
        scoreboard_games = pl.DataFrame()
        exclude_pages = []
        for (page,) in tqdm(
            tournaments.select(pl.col("OverviewPage")).rows(),
            desc="Tournaments",
        ):
            sg = leaguepedia.get_scoreboard_games(
                tables=["tournaments"],
                join_on="T.OverviewPage=SG.OverviewPage",
                where=f'T.OverviewPage="{page}"',
            )
            if sg.is_empty():
                exclude_pages.append(page)
            else:
                scoreboard_games = pl.concat([scoreboard_games, sg])
        tournaments = tournaments.filter(~pl.col("OverviewPage").is_in(exclude_pages))
        write_parquet(tournaments, f"./parquet/tournaments/{year}_tournaments.parquet")
        write_parquet(
            scoreboard_games,
            f"./parquet/scoreboard_games/{year}_scoreboard_games.parquet",
        )


def parse_scoreboard_players(start: int = 2011, end: int | None = None) -> None:
    """Parses the scoreboard players for a given range of years.

    Args:
        start (int): The starting year of the range. Default is 2011.
        end (int | None): The ending year of the range. If None, the current year is used.
            Default is None.

    Returns:
        None
    """
    end = end or datetime.datetime.now(tz=ZoneInfo("Asia/Seoul")).year

    leaguepedia = Leaguepedia()
    for year in tqdm(range(start, end + 1), desc="Years", position=0):
        tournaments = pl.read_parquet(
            f"./parquet/tournaments/{year}_tournaments.parquet",
        )
        scoreboard_games = pl.read_parquet(
            f"./parquet/scoreboard_games/{year}_scoreboard_games.parquet",
        )
        scoreboard_players = pl.DataFrame()
        for (page,) in tqdm(
            tournaments.select(pl.col("OverviewPage")).rows(),
            desc="Tournaments",
            position=1,
            leave=False,
        ):
            sg = scoreboard_games.filter(pl.col("OverviewPage") == page)
            teams = (
                sg.melt(id_vars="OverviewPage", value_vars=["Team1", "Team2"])
                .select(pl.col("value"))
                .unique()
            )
            for (team,) in teams.rows():
                sp = leaguepedia.get_scoreboard_players(
                    tables=["tournaments"],
                    join_on="T.OverviewPage=SP.OverviewPage",
                    where=f'T.OverviewPage="{page}" and SP.Team="{team}"',
                )
                if not sp.is_empty():
                    scoreboard_players = pl.concat([scoreboard_players, sp])

            if (
                sg.shape[0] * 10
                != scoreboard_players.filter(pl.col("OverviewPage") == page).shape[0]
            ):
                target = scoreboard_players.filter(pl.col("OverviewPage") == page)
                print(target)
                msg = "scoreboard games: {sg.shape[0]}, players: {df.shape[0]}"
                raise AssertionError(msg)
        scoreboard_players = scoreboard_players.sort(
            by=["OverviewPage", "GameId", "Team", "Role Number"],
        )
        write_parquet(
            scoreboard_players,
            f"./parquet/scoreboard_players/{year}_scoreboard_players.parquet",
        )


def main() -> None:
    parse_tournaments(start=2023)
    parse_scoreboard_games(start=2023)
    parse_scoreboard_players(start=2023)


if __name__ == "__main__":
    main()

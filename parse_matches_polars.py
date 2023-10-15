"""Parse lolesports matches."""
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from tqdm import tqdm

from lol_fandom_pl import Leaguepedia


def parse_tournaments(start: int = 2011, end: int | None = None) -> None:
    """Parses tournaments.

    Args:
        start (int): The start year of the tournaments to parse. Defaults to 2011.
        end (int | None): The end year of the tournaments to parse.
            Defaults to None, which means the current year.

    Returns:
        None
    """
    if end is None:
        end = datetime.datetime.now(tz=ZoneInfo("Asia/Seoul")).year

    leaguepedia = Leaguepedia()
    leagues = leaguepedia.get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    for year in tqdm(range(start, end + 1), desc="Years"):
        tournaments = pl.DataFrame()
        for (league,) in tqdm(leagues.select(pl.col("League")).rows(), desc="Leagues"):
            tournament = leaguepedia.get_tournaments(
                tables=["leagues"],
                join_on="L.League=T.League",
                where=f'L.League="{league}" and T.Year={year}',
            )
            tournaments = (
                pl.concat([tournaments, tournament])
                if not tournament.is_empty()
                else tournaments
            )
        file_path = Path(f"./parquet/tournaments/{year}_tournaments.parquet")
        if not file_path.parent.is_dir():
            file_path.parent.mkdir(parents=True)
        tournaments.write_parquet(file_path)


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
    if end is None:
        end = datetime.datetime.now(tz=ZoneInfo("Asia/Seoul")).year

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
        tournaments.write_parquet(f"./parquet/tournaments/{year}_tournaments.parquet")
        file_path = Path(f"./parquet/scoreboard_games/{year}_scoreboard_games.parquet")
        if not file_path.parent.is_dir():
            file_path.parent.mkdir(parents=True)
        scoreboard_games.write_parquet(file_path)


def main() -> None:
    parse_tournaments(start=2023)
    parse_scoreboard_games(start=2023)


if __name__ == "__main__":
    main()

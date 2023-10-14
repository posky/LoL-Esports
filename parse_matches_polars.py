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


def main() -> None:
    parse_tournaments(start=2023)


if __name__ == "__main__":
    main()

"""Parse lolesports matches."""
import datetime
import logging
from pathlib import Path

import polars as pl
from tqdm import tqdm

from lol_fandom_pl import (
    get_leagues,
    get_scoreboard_games,
    get_scoreboard_players,
    get_tournaments,
)


def get_updated_contents(df: pl.DataFrame, path: Path) -> pl.DataFrame:
    """Return updated contents.

    Args:
        df (pl.DataFrame): The input DataFrame.
        path (Path): The path to the parquet file.

    Returns:
        pl.DataFrame: The updated DataFrame.
    """
    origin = pl.read_parquet(path) if path.is_file() else pl.DataFrame(schema=df.schema)

    return df.filter(
        ~pl.col("GameId").is_in(origin.select(pl.col("GameId")).to_series()),
    )


def parse_tournaments(start: int = 2011, end: int | None = None) -> None:
    """Parses tournaments data from the start year to the current or specified end year.

    Args:
        start (int, optional): The start year of the tournaments to parse.
            Defaults to 2011.
        end (int, optional): The end year of the tournaments to parse.
            Defaults to the current year.

    Returns:
        None: This function does not return anything.
    """
    if end is None:
        end = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year

    leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    for year in tqdm(range(start, end + 1), desc="Tournaments"):
        tournaments = pl.DataFrame()
        for (league,) in tqdm(
            leagues.select(pl.col("League Short")).rows(),
            desc="Leagues",
        ):
            logging.debug("league: %s", league)
            tournament = get_tournaments(
                where=f'L.League_Short="{league}" and T.Year={year}',
            )
            tournaments = (
                pl.concat([tournaments, tournament])
                if tournament.shape[0] > 0
                else tournaments
            )
            logging.debug("tournaments columns: %s", tournaments.columns)
        tournaments = (
            tournaments.sort(by=["Year", "DateStart", "Date"])
            if tournaments.shape[0] > 0
            else tournaments
        )

        file_path = Path(f"./csv/tournaments/{year}_tournaments.parquet")
        if not file_path.parent.is_dir():
            Path.mkdir(file_path.parent, parents=True)
        tournaments.write_parquet(file_path)


def parse_scoreboard_games(start: int = 2011, end: int | None = None) -> None:
    """Parses the scoreboard games for a given range of years.

    Args:
        start (int, optional): The starting year to parse. Defaults to 2011.
        end (int, optional): The ending year to parse. If not provided,
            the current year will be used.

    Returns:
        None
    """
    if end is None:
        end = datetime.datetime.now(tz=datetime.UTC).year

    for year in tqdm(range(start, end + 1), desc="Year"):
        tournaments = pl.read_parquet(f"./csv/tournaments/{year}_tournaments.parquet")
        logging.debug("%d - tournament %s", year, tournaments.shape)
        scoreboard_games = pl.DataFrame()
        for (page,) in tqdm(
            tournaments.select(pl.col("OverviewPage")).rows(),
            desc="Scoreboard Games",
        ):
            logging.debug("page is: %s", page)
            sg = get_scoreboard_games(where=f'T.OverviewPage="{page}"')
            if sg.shape[0] == 0:
                logging.debug("\t%s - drop", page)
                tournaments = tournaments.filter(pl.col("OverviewPage") != page)
                continue
            league = tournaments.select(pl.col("League") == page).to_series()[0]
            sg = sg.with_columns(League=league)
            logging.debug("%s - %d", page, sg.shape[0])
            scoreboard_games = pl.concat([scoreboard_games, sg])
        scoreboard_games = scoreboard_games.sort(by="DateTime UTC")

        file_path = Path(f"./csv/scoreboard_games/{year}_scoreboard_games.parquet")
        updated_df = get_updated_contents(scoreboard_games, file_path)
        logging.info(
            updated_df.select(
                pl.col(
                    [
                        "OverviewPage",
                        "Team1",
                        "Team2",
                        "WinTeam",
                        "DateTime UTC",
                        "GameId",
                    ],
                ),
            ),
        )
        if not file_path.parent.is_dir():
            Path.mkdir(file_path.parent, parents=True)
        scoreboard_games.write_parquet(file_path)
        logging.debug("%d scoreboard_games %s", year, scoreboard_games.shape)
        tournaments.write_parquet(f"./csv/tournaments/{year}_tournaments.parquet")
        logging.debug("%d tournaments %s", year, tournaments.shape)


def parse_scoreboard_players(start: int = 2011, end: int | None = None) -> None:
    """Parses the scoreboard players data for a given range of years.

    Args:
        start (int, optional): The starting year to parse. Defaults to 2011.
        end (int | None, optional): The ending year to parse.
            If None, the current year is used. Defaults to None.

    Returns:
        None
    """
    if end is None:
        end = datetime.datetime.now(tz=datetime.UTC).year

    for year in tqdm(range(start, end + 1), desc="Year"):
        tournaments = pl.read_parquet(f"./csv/tournaments/{year}_tournaments.parquet")
        scoreboard_games = pl.read_parquet(
            f"./csv/scoreboard_games/{year}_scoreboard_games.parquet",
        )
        logging.debug("%d - tournament %s", year, tournaments.shape)
        logging.debug("%d - scoreboard games %s", year, scoreboard_games.shape)
        scoreboard_players = pl.DataFrame()
        for (page,) in tqdm(
            tournaments.select(pl.col("OverviewPage")).rows(),
            desc="Scoreboard Players",
        ):
            logging.debug("\t%s", page)
            teams = (
                scoreboard_games.filter(pl.col("OverviewPage") == page)
                .melt(id_vars="OverviewPage", value_vars=["Team1", "Team2"])
                .select(pl.col("value"))
                .unique()
                .to_series()
                .to_list()
            )
            len_sp = 0
            for team in tqdm(teams, deps="Teams"):
                sp = get_scoreboard_players(
                    where=f'T.OverviewPage="{page}" and SP.Team="{team}"',
                )
                if sp.shape[0] > 0:
                    len_sp += sp.shape[0]
                    scoreboard_players = pl.concat([scoreboard_players, sp])
            len_sg = scoreboard_games.filter(pl.col("OverviewPage") == page).shape[0]
            logging.debug(
                "\t\tscoreboard games - %d | scoreboard players - %d | %s",
                len_sg,
                len_sp,
                len_sg * 10 == len_sp,
            )
        scoreboard_players = scoreboard_players.sort(
            by=["DateTime UTC", "Team", "Role Number"],
        )
        file_path = Path(f"./csv/scoreboard_players/{year}_scoreboard_players.parquet")
        if not file_path.parent.is_dir():
            Path.mkdir(file_path.parent, parents=True)
        scoreboard_players.write_parquet(file_path)
        logging.debug("%d scoreboard players %s", year, scoreboard_players.shape)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parse_tournaments(start=2023)
    parse_scoreboard_games(start=2023)
    parse_scoreboard_players(start=2023)


if __name__ == "__main__":
    main()

"""Parse lolesports matches."""
import datetime
import logging
import webbrowser
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from lol_fandom import (
    get_leagues,
    get_match_schedule,
    get_scoreboard_games,
    get_scoreboard_players,
    get_tournament_rosters,
    get_tournaments,
    set_default_delay,
)

pd.set_option("display.max_columns", None)
set_default_delay(0.2)


def change_to_tuple(lst: list[str]) -> str:
    """Change list of string to tuple.

    Args:
        lst (List[str]): List of string

    Returns:
        str: tuple
    """
    return "(" + ", ".join(f'"{x}"' for x in lst) + ")"


def get_updated_contents(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Update origin dataframe with new data.

    Args:
        df (pd.DataFrame): dataframe of new data
        path (str): path of origin dataframe

    Returns:
        pd.DataFrame: updated dataframe
    """
    if Path(path).is_file():
        origin = pd.read_csv(path)
    else:
        origin = pd.DataFrame(columns=df.columns)
    origin[df.columns] = origin[df.columns].astype(df.dtypes)

    new_df = origin.merge(
        df,
        how="outer",
        on="GameId",
        suffixes=["_origin", ""],
        indicator=True,
    )
    return new_df.query('_merge == "right_only"').drop(columns=["_merge"])


def parse_tournaments(start: int = 2011, end: int | None = None) -> None:
    """Parse tournaments from start year to end year.

    Args:
        start (int, optional): Start year of tournaments. Defaults to 2011.
        end (int, optional): End year of tournaments.
            Defaults to datetime.datetime.now().year.
    """
    if end is None:
        end = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year

    logging.info("=========== Tournaments ===========")
    leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    for year in tqdm(range(start, end + 1)):
        logging.debug("%d tournaments", year)
        tournaments = pd.DataFrame()
        for league in tqdm(leagues["League Short"]):
            t = get_tournaments(where=f'L.League_Short="{league}" and T.Year={year}')
            logging.debug("\t%s - %d", league, t.shape[0])
            tournaments = pd.concat([tournaments, t], ignore_index=True)
        tournaments = tournaments.sort_values(
            by=["Year", "DateStart", "Date"],
        ).reset_index(drop=True)

        file_path = Path(f"./csv/tournaments/{year}_tournaments.csv")
        if not file_path.parent.is_dir():
            Path.mkdir(file_path, parents=True)
        tournaments.to_csv(file_path, index=False)
        logging.debug("%d tournaments - %s", year, tournaments.shape)


def parse_scoreboard_games(start: int = 2011, end: int | None = None) -> None:
    """Parse scoreboard games from start year to end year.

    Args:
        start (int, optional): Start year of tournaments. Defaults to 2011.
        end (int, optional): End year of tournaments.
            Defaults to datetime.datetime.now().year.
    """
    if end is None:
        end = datetime.datetime.now(tz=datetime.UTC).year

    logging.info("=========== Scoreboard Games ===========")
    leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
    for year in tqdm(range(start, end + 1)):
        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        logging.debug("%d - tournament %s", year, tournaments.shape)
        scoreboard_games = pd.DataFrame()
        for page in tqdm(tournaments["OverviewPage"]):
            sg = get_scoreboard_games(where=f'T.OverviewPage="{page}"')
            if sg is None:
                logging.debug("\t%s - drop", page)
                tournaments = tournaments.drop(
                    tournaments.loc[tournaments["OverviewPage"] == page].index,
                )
                continue
            league = tournaments.loc[
                tournaments["OverviewPage"] == page,
                "League",
            ].iloc[0]
            league = leagues.loc[leagues["League"] == league, "League Short"].iloc[0]
            sg["League"] = league
            logging.debug("%s - %d", page, sg.shape[0])
            scoreboard_games = pd.concat([scoreboard_games, sg], ignore_index=True)
        scoreboard_games = scoreboard_games.sort_values(by="DateTime UTC").reset_index(
            drop=True,
        )

        file_path = f"./csv/scoreboard_games/{year}_scoreboard_games.csv"
        updated_df = get_updated_contents(scoreboard_games, file_path)
        print()
        print(
            updated_df[
                ["OverviewPage", "Team1", "Team2", "WinTeam", "DateTime UTC", "GameId"]
            ],
        )
        scoreboard_games.to_csv(file_path, index=False)
        logging.debug("%d scoreboard_games %s", year, scoreboard_games.shape)
        tournaments.to_csv(f"./csv/tournaments/{year}_tournaments.csv", index=False)
        logging.debug("%d tournaments %s", year, tournaments.shape)


def parse_scoreboard_players(start: int = 2011, end: int | None = None) -> None:
    """Parse scoreboard players from start year to end year.

    Args:
        start (int, optional): Start year of tournaments. Defaults to 2011.
        end (int, optional): End year of tournaments.
            Defaults to datetime.datetime.now().year.
    """
    if end is None:
        end = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year

    logging.info("=========== Scoreboard Players ===========")
    for year in tqdm(range(start, end + 1)):
        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv",
        )
        logging.debug("%d - tournament %s", year, tournaments.shape)
        logging.debug("%d - scoreboard games %s", year, scoreboard_games.shape)
        scoreboard_players = pd.DataFrame()
        for page in tqdm(tournaments["OverviewPage"]):
            logging.debug("\t%s", page)
            teams = pd.unique(
                scoreboard_games.loc[
                    scoreboard_games["OverviewPage"] == page,
                    ["Team1", "Team2"],
                ]
                .to_numpy()
                .ravel(),
            )
            len_sp = 0
            for team in teams:
                sp = get_scoreboard_players(
                    where=f'T.OverviewPage="{page}" and SP.Team="{team}"',
                )
                len_sp += sp.shape[0]
                scoreboard_players = pd.concat([scoreboard_players, sp])
            len_sg = scoreboard_games.loc[
                scoreboard_games["OverviewPage"] == page
            ].shape[0]
            logging.debug(
                "\t\tscoreboard games - %d | scoreboard players - %d | %s",
                len_sg,
                len_sp,
                len_sg * 10 == len_sp,
            )
        scoreboard_players = scoreboard_players.sort_values(
            by=["DateTime UTC", "Team", "Role Number"],
        ).reset_index(drop=True)
        file_path = f"./csv/scoreboard_players/{year}_scoreboard_players.csv"
        # updated_df = get_updated_contents(scoreboard_players, file_path)
        # print(updated_df)
        scoreboard_players.to_csv(file_path, index=False)
        logging.debug("%d scoreboard_players %s", year, scoreboard_players.shape)


def parse_tournament_rosters(start: int = 2011, end: int | None = None) -> None:
    """Parse tournament rosters from start year to end year.

    Args:
        start (int, optional): Start year of tournaments. Defaults to 2011.
        end (int, optional): End year of tournaments.
            Defaults to datetime.datetime.now().year.
    """
    if end is None:
        end = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year

    logging.info("=========== Tournament Rosters ===========")
    for year in tqdm(range(start, end + 1)):
        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        logging.debug("%d - tournament %s", year, tournaments.shape)
        tournament_rosters = pd.DataFrame()
        for page in tqdm(tournaments["OverviewPage"]):
            tr = get_tournament_rosters(where=f'T.OverviewPage="{page}"')
            logging.debug("\t%s - %d", page, tr.shape[0])
            tournament_rosters = pd.concat([tournament_rosters, tr], ignore_index=True)
        tournament_rosters.to_csv(
            f"./csv/tournament_rosters/{year}_tournament_rosters.csv",
            index=False,
        )
        logging.debug("%d tournament rosters %s", year, tournament_rosters.shape)


def parse_matches_schedule(start: int = 2011, end: int | None = None) -> None:
    """Parse match schedule from start year to end year.

    Args:
        start (int, optional): Start year of tournaments. Defaults to 2011.
        end (int, optional): End year of tournaments.
            Defaults to datetime.datetime.now().year.
    """
    if end is None:
        end = datetime.datetime.now(
            tz=datetime.timezone(datetime.timedelta(hours=9)),
        ).year

    logging.info("=========== Matches Schedule ===========")
    for year in tqdm(range(start, end + 1)):
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv",
        )
        logging.debug("%d - scoreboard games %s", year, scoreboard_games.shape)
        matches_schedule = pd.DataFrame()
        for page in tqdm(scoreboard_games["OverviewPage"].unique()):
            ids = scoreboard_games.loc[
                scoreboard_games["OverviewPage"] == page,
                "MatchId",
            ].unique()
            ms = get_match_schedule(where=f"MS.MatchId in {change_to_tuple(ids)}")
            if ms is not None:
                logging.debug("\t%s - %d", page, ms.shape[0])
                matches_schedule = pd.concat([matches_schedule, ms], ignore_index=True)
        matches_schedule.to_csv(
            f"./csv/match_schedule/{year}_match_schedule.csv",
            index=False,
        )
        logging.debug("%d match schedule %s", year, matches_schedule.shape)


def check_new_team() -> None:
    """Check new teams."""
    logging.info("=========== Check new teams ===========")

    teams = pd.read_csv("./csv/teams_id.csv")
    names = []

    for year in tqdm(range(2011, 2024)):
        tournaments = pd.read_csv(f"./csv/tournaments/{year}_tournaments.csv")
        scoreboard_games = pd.read_csv(
            f"./csv/scoreboard_games/{year}_scoreboard_games.csv",
        )
        # logging.info("%d - tournament %d", year, tournaments.shape[0])
        # logging.info("%d - scoreboard games %d", year, scoreboard_games.shape[0])
        for page in tqdm(tournaments["OverviewPage"]):
            sg = scoreboard_games.loc[scoreboard_games["OverviewPage"] == page]
            team_names = list(pd.unique(sg[["Team1", "Team2"]].to_numpy().ravel()))
            names = []
            for name in team_names:
                if name not in teams["team"].to_numpy():
                    names.append(name)
            if len(names) > 0:
                names = sorted(names, key=lambda x: x.lower())
                break
        if len(names) > 0:
            url = f'https://lol.fandom.com/wiki/{page.replace(" ", "_")}'
            webbrowser.open(url)
            print(url)
            print(f"{page}\n{names}")
            for link in names:
                url = f'https://lol.fandom.com/wiki/{link.replace(" ", "_")}'
                webbrowser.open(url)
            break
    if len(names) == 0:
        logging.info("Completed")


def main() -> None:
    """Parse lolesports."""
    logging.basicConfig(level=logging.INFO)

    # Parse lolesports data
    parse_tournaments(start=2023)
    parse_scoreboard_games(start=2023)
    parse_scoreboard_players(start=2023)
    # parse_tournament_rosters(start=2023)
    # parse_matches_schedule(start=2023)


if __name__ == "__main__":
    main()

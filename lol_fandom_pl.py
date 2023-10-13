"""lol.fandom.com API."""
from __future__ import annotations

import time
from typing import ClassVar

import mwclient
import polars as pl

pl.Config(fmt_str_lengths=100)

SITE = mwclient.Site("lol.fandom.com", path="/")


class Leaguepedia:
    DELAY = 1
    last_query = time.time()
    TABLES: ClassVar[dict[str, str]] = {
        "leagues": "Leagues=L",
        "tournaments": "Tournaments=T",
        "scoreboard_games": "ScoreboardGames=SG",
        "scoreboard_players": "ScoreboardPlayers=SP",
        "tournament_rosters": "TournamentRosters=TR",
        "player_redirects": "PlayerRedirects=PR",
        "teams": "Teams=TS",
        "tournament_results": "TournamentResults=TR",
    }

    def __init__(self: Leaguepedia) -> None:
        self.api = SITE.api

    @classmethod
    def set_query_delay(cls: Leaguepedia, delay: int) -> None:
        cls.DELAY = delay

    @classmethod
    def delay_between_query(cls: Leaguepedia) -> None:
        delay = cls.DELAY - (time.time() - cls.last_query)

        if delay > 0:
            time.sleep(delay)

        cls.last_query = time.time()

    @classmethod
    def __from_response(cls: Leaguepedia, response: dict) -> pl.DataFrame:
        if len(response["cargoquery"]) > 0:
            columns = response["cargoquery"][0]["title"].keys()
            return pl.DataFrame(
                [row["title"] for row in response["cargoquery"]],
                schema={key: pl.Utf8 for key in columns},
            )
        return pl.DataFrame()

    def get_leagues(self: Leaguepedia, where: str = "") -> pl.DataFrame:
        self.delay_between_query()

        response = self.api(
            "cargoquery",
            limit="max",
            tables=self.TABLES["leagues"],
            fields="L.League, L.League_Short, L.Region, L.Level, L.IsOfficial",
            where=where,
        )

        return self.__from_response(response)

    def get_tournaments(
        self: Leaguepedia,
        *,
        tables: list[str] | None = None,
        join_on: str = "",
        where: str = "",
    ) -> pl.DataFrame:
        if tables is None:
            tables = []
        if any(table not in self.TABLES for table in tables):
            msg = "Invalid table name."
            raise ValueError(msg)
        tables = ", ".join(self.TABLES[table] for table in {*tables, "tournaments"})
        self.delay_between_query()

        response = self.api(
            "cargoquery",
            limit="max",
            tables=tables,
            join_on=join_on,
            fields=(
                "T.Name, T.OverviewPage, T.DateStart, T.Date, T.DateStartFuzzy,"
                " T.League, T.Region, T.Prizepool, T.Currency, T.Country,"
                " T.ClosestTimezone, T.Rulebook, T.EventType, T.Links, T.Sponsors,"
                " T.Organizer, T.Organizers, T.StandardName, T.StandardName_Redirect,"
                " T.BasePage, T.Split, T.SplitNumber, T.SplitMainPage,"
                " T.TournamentLevel, T.IsQualifier, T.IsPlayoffs, T.IsOfficial, T.Year,"
                " T.LeagueIconKey, T.AlternativeNames, T.ScrapeLink, T.Tags,"
                " T.SuppressTopSchedule"
            ),
            where=where,
        )

        tournaments = self.__from_response(response)
        if not tournaments.is_empty():
            int_cols = ["SplitNumber", "Year"]
            datetime_cols = ["DateStart", "Date", "DateStartFuzzy"]
            tournaments = tournaments.with_columns(
                pl.col(int_cols).cast(pl.Int32),
                pl.col(datetime_cols).str.strptime(pl.Datetime, "%Y-%m-%d"),
            )
        return tournaments

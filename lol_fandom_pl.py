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
        """Set the query delay for the Leaguepedia API.

        Args:
            cls (Leaguepedia): The Leaguepedia class.
            delay (int): The delay value to be set.

        Returns:
            None
        """
        cls.DELAY = delay

    @classmethod
    def delay_between_query(cls: Leaguepedia) -> None:
        """Delays the execution of queries by the specified delay time.

        Args:
            cls (Leaguepedia): The Leaguepedia class instance.

        Returns:
            None
        """
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

    def get_leagues(
        self: Leaguepedia,
        *,
        tables: list[str] | None = None,
        join_on: str = "",
        where: str = "",
    ) -> pl.DataFrame:
        """Retrieves leagues from Leaguepedia.

        Args:
            tables (Optional[list[str]]): A list of table names to query.
                Defaults to None.
            join_on (str): The join condition for the tables. Defaults to "".
            where (str): The WHERE condition for the query. Defaults to "".

        Returns:
            DataFrame: A pandas DataFrame containing the retrieved leagues.
        """
        tables = tables or []
        if any(table not in self.TABLES for table in tables):
            msg = "Invalid table name."
            raise ValueError(msg)
        tables = ", ".join(self.TABLES[table] for table in {*tables, "leagues"})
        self.delay_between_query()

        fields = "L.League, L.League_Short, L.Region, L.Level, L.IsOfficial"
        response = self.api(
            "cargoquery",
            limit="max",
            tables=tables,
            join_on=join_on,
            fields=fields,
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
        """Retrieves tournaments data from Leaguepedia.

        Args:
            tables (list[str] | None, optional): A list of tables to include in the query.
                Defaults to None.
            join_on (str, optional): The join condition for the tables. Defaults to "".
            where (str, optional): The where clause for the query. Defaults to "".

        Returns:
            pl.DataFrame: A DataFrame containing the retrieved tournaments data.

        Raises:
            ValueError: If an invalid table name is provided in the 'tables' parameter.
        """
        tables = tables or []
        if any(table not in self.TABLES for table in tables):
            msg = "Invalid table name."
            raise ValueError(msg)
        tables = ", ".join(self.TABLES[table] for table in {*tables, "tournaments"})
        self.delay_between_query()

        fields = (
            "T.Name, T.OverviewPage, T.DateStart, T.Date, T.DateStartFuzzy,"
            " T.League, T.Region, T.Prizepool, T.Currency, T.Country,"
            " T.ClosestTimezone, T.Rulebook, T.EventType, T.Links, T.Sponsors,"
            " T.Organizer, T.Organizers, T.StandardName, T.StandardName_Redirect,"
            " T.BasePage, T.Split, T.SplitNumber, T.SplitMainPage,"
            " T.TournamentLevel, T.IsQualifier, T.IsPlayoffs, T.IsOfficial, T.Year,"
            " T.LeagueIconKey, T.AlternativeNames, T.ScrapeLink, T.Tags,"
            " T.SuppressTopSchedule"
        )
        response = self.api(
            "cargoquery",
            limit="max",
            tables=tables,
            join_on=join_on,
            fields=fields,
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

    def get_scoreboard_games(
        self: Leaguepedia,
        *,
        tables: list[str] | None = None,
        join_on: str = "",
        where: str = "",
    ) -> pl.DataFrame:
        """Retrieves the scoreboard games from the Leaguepedia API.

        Args:
            self (Leaguepedia): The Leaguepedia instance.
            tables (list[str] | None): A list of table names to query. Defaults to None.
            join_on (str): The join condition for the tables. Defaults to an empty string.
            where (str): The where condition for the query. Defaults to an empty string.

        Returns:
            pl.DataFrame: The scoreboard games data as a Pandas DataFrame.

        Raises:
            ValueError: If an invalid table name is provided.
        """
        tables = tables or []
        if any(table not in self.TABLES for table in tables):
            msg = "Invalid table name."
            raise ValueError(msg)
        tables = ", ".join(
            self.TABLES[table] for table in {*tables, "scoreboard_games"}
        )
        self.delay_between_query()

        fields = (
            "SG.OverviewPage, SG.Tournament, SG.Team1, SG.Team2, SG.WinTeam,"
            " SG.LossTeam, SG.DateTime_UTC, SG.DST, SG.Team1Score, SG.Team2Score,"
            " SG.Winner, SG.Gamelength, SG.Gamelength_Number, SG.Team1Bans,"
            " SG.Team2Bans, SG.Team1Picks, SG.Team2Picks, SG.Team1Players,"
            " SG.Team2Players, SG.Team1Dragons, SG.Team2Dragons, SG.Team1Barons,"
            " SG.Team2Barons, SG.Team1Towers, SG.Team2Towers, SG.Team1Gold,"
            " SG.Team2Gold, SG.Team1Kills, SG.Team2Kills, SG.Team1RiftHeralds,"
            " SG.Team2RiftHeralds, SG.Team1Inhibitors, SG.Team2Inhibitors,"
            " SG.Patch, SG.PatchSort, SG.MatchHistory, SG.VOD, SG.N_Page,"
            " SG.N_MatchInTab, SG.N_MatchInPage, SG.N_GameInMatch, SG.Gamename,"
            " SG.UniqueLine, SG.GameId, SG.MatchId, SG.RiotPlatformGameId,"
            " SG.RiotPlatformId, SG.RiotGameId, SG.RiotHash, SG.RiotVersion"
        )
        response = self.api(
            "cargoquery",
            limit="max",
            tables=tables,
            join_on=join_on,
            fields=fields,
            where=where,
        )

        scoreboard_games = self.__from_response(response)
        if not scoreboard_games.is_empty():
            int_cols = [
                "Team1Score",
                "Team2Score",
                "Winner",
                "Team1Dragons",
                "Team2Dragons",
                "Team1Barons",
                "Team2Barons",
                "Team1Towers",
                "Team2Towers",
                "Team1Kills",
                "Team2Kills",
                "Team1RiftHeralds",
                "Team2RiftHeralds",
                "Team1Inhibitors",
                "Team2Inhibitors",
            ]
            float_cols = ["Gamelength Number", "Team1Gold", "Team2Gold"]
            datetime_cols = ["DateTime UTC"]
            scoreboard_games = scoreboard_games.with_columns(
                pl.col(int_cols).cast(pl.Int32),
                pl.col(float_cols).cast(pl.Float32),
                pl.col(datetime_cols).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            )
        return scoreboard_games

    def get_scoreboard_players(
        self: Leaguepedia,
        *,
        tables: list[str] | None = None,
        join_on: str = "",
        where: str = "",
    ) -> pl.DataFrame:
        """Get the scoreboard players from the specified tables.

        Args:
            self (Leaguepedia): The Leaguepedia object.
            tables (list[str] | None): A list of table names to query.
                If None, all tables will be queried.
            join_on (str): The join condition for the tables.
            where (str): The WHERE condition for the query.

        Returns:
            pl.DataFrame: The scoreboard players data.
        """
        if tables is None:
            tables = []
        if any(table not in self.TABLES for table in tables):
            msg = "Invalid table name."
            raise ValueError(msg)
        tables = ", ".join(
            self.TABLES[table] for table in {*tables, "scoreboard_players"}
        )
        self.delay_between_query()

        fields = (
            "SP.OverviewPage, SP.Name, SP.Link, SP.Champion, SP.Kills, SP.Deaths,"
            " SP.Assists, SP.SummonerSpells, SP.Gold, SP.CS, SP.DamageToChampions,"
            " SP.VisionScore, SP.Items, SP.Trinket, SP.KeystoneMastery,"
            " SP.KeystoneRune, SP.PrimaryTree, SP.SecondaryTree, SP.Runes,"
            " SP.TeamKills, SP.TeamGold, SP.Team, SP.TeamVs, SP.Time, SP.PlayerWin,"
            " SP.DateTime_UTC, SP.DST, SP.Tournament, SP.Role, SP.Role_Number,"
            " SP.IngameRole, SP.Side, SP.UniqueLine, SP.UniqueLineVs,"
            " SP.UniqueRole, SP.UniqueRoleVs, SP.GameId, SP.MatchId, SP.GameTeamId,"
            " SP.GameRoleId, SP.GameRoleIdVs, SP.StatsPage"
        )
        response = self.api(
            "cargoquery",
            limit="max",
            tables=tables,
            join_on=join_on,
            fields=fields,
            where=where,
        )

        scoreboard_players = self.__from_response(response)
        if not scoreboard_players.is_empty():
            int_cols = [
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
            datetime_cols = ["Time", "DateTime UTC"]
            scoreboard_players = scoreboard_players.with_columns(
                pl.col(int_cols).cast(pl.Int32),
                pl.col(datetime_cols).str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
            )

        return scoreboard_players

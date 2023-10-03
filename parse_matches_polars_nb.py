# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import logging
import webbrowser

import polars as pl
from tqdm import tqdm

from lol_fandom_pl import get_leagues


# %%
def get_new_id(ids: pl.DataFrame) -> int:
    """Takes a polars DataFrame `ids` and returns a new integer ID.

    Args:
        ids (pl.DataFrame): The input DataFrame containing IDs.

    Returns:
        int: The new ID.
    """
    if "team_id" in ids.columns:
        id_list = sorted(ids.select(pl.col("team_id")).unique().to_series().to_list())
    else:
        id_list = sorted(ids.select(pl.col("player_id")).unique().to_series().to_list())

    if len(id_list) == 0:
        return 1
    prev = id_list[0]
    for cur_id in id_list[1:]:
        if prev + 1 != cur_id:
            return prev + 1
        prev = cur_id
    return prev + 1


def concat_teams(teams: pl.DataFrame, name: str, new_id: int = -1) -> pl.DataFrame:
    """Concatenates a new team to the existing teams DataFrame.

    Args:
        teams (pl.DataFrame): The DataFrame containing the existing teams.
        name (str): The name of the new team to be added.
        new_id (int, optional): The ID of the new team. Defaults to -1.

    Returns:
        pl.DataFrame: The updated DataFrame with the new team added.
    """
    if new_id == -1:
        new_id = get_new_id(teams)
    return pl.concat([teams, pl.DataFrame({"team": [name], "team_id": [new_id]})])


# %%
leagues = get_leagues(where='L.Level="Primary" and L.IsOfficial="Yes"')
print(leagues)

# %% [markdown]
# # Teams
#

# %%
teams = pl.read_csv("./csv/teams_id.csv")
names = []

for year in tqdm(range(2011, 2024), desc="Teams"):
    tournaments = pl.read_parquet(f"./csv/tournaments/{year}_tournaments.parquet")
    scoreboard_games = pl.read_parquet(
        f"./csv/scoreboard_games/{year}_scoreboard_games.parquet",
    )
    tournament_rosters = pl.read_parquet(
        f"./csv/tournament_rosters/{year}_tournament_rosters.parquet",
    )
    logging.info("%d - tournament %d", year, tournaments.shape[0])
    logging.info("%d - scoreboard games %d", year, scoreboard_games.shape[0])
    logging.info("%d - tournament rosters %d", year, tournament_rosters.shape[0])
    for (page,) in tqdm(
        tournaments.select(pl.col("OverviewPage")).rows(),
        desc="Tournaments",
    ):
        logging.info("\t%s", page)
        sg = scoreboard_games.filter(pl.col("OverviewPage") == page)
        tr = tournament_rosters.filter(pl.col("OverviewPage") == page)
        team_names = list(
            set(
                sg.melt(id_vars="OverviewPage", value_vars=["Team1", "Team2"])
                .select(pl.col("value"))
                .to_series()
                .to_list()
                + tr.select(pl.col("Team")).to_series().to_list(),
            ),
        )
        names.clear()
        names.extend(
            name
            for name in team_names
            if teams.filter(pl.col("team") == name).shape[0] == 0
        )
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
    logging.info("Complete")

# %%
target_name = "noah"
with pl.Config(tbl_rows=-1):
    print(teams.filter(pl.col("team").str.to_lowercase().str.contains(target_name)))

# %%
if len(names) > 0:
    teams = concat_teams(teams, names.pop(0), new_id=-1)
names

# %%
teams = teams.sort(by=["team_id", "team"])
print(teams)

# %%
teams = teams.with_columns(
    pl.col("team")
    .str.to_lowercase()
    .str.replace(" ", "")
    .str.replace("-", "")
    .alias("lower"),
)
lst = sorted(teams.select(pl.col(["lower", "team_id"])).rows())

is_complete = True
prev = lst[0]
for cur in lst[1:]:
    if prev[0] == cur[0] and prev[1] != cur[1]:
        is_complete = False
        break
    prev = cur

if not is_complete:
    print("Incomplete")
    candidate = teams.filter(pl.col("lower") == prev[0])
    print(candidate)
else:
    print("Completed")

# %%
if is_complete:
    teams.select(pl.col(["team", "team_id"])).write_csv("./csv/teams_id.csv")
    print("Save complete")

# %%

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
#     display_name: Python 3.10.4 64-bit
#     language: python
#     name: python3
# ---

# %%
from typing import Iterable, Any
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.max_columns", None)


# %%
def get_matches(
    matches_data,
    league: str | Iterable[str] = None,
    split: str | Iterable[str] = None,
    playoffs=None,
    patch: str | Iterable[str] = None,
):
    df = matches_data
    if league is not None:
        if not isinstance(league, str) and isinstance(league, Iterable):
            df = df.loc[df["league"].isin(league)]
        else:
            df = df.loc[df["league"] == league]
    if split is not None:
        if not isinstance(split, str) and isinstance(split, Iterable):
            df = df.loc[df["split"].isin(split)]
        else:
            df = df.loc[df["split"] == split]
    if playoffs is not None:
        df = df.loc[df["playoffs"] == playoffs]
    if patch is not None:
        if not isinstance(patch, str) and isinstance(patch, Iterable):
            df = df.loc[df["patch"].isin(patch)]
        else:
            df = df.loc[df["patch"] == patch]

    return df


def get_rows_columns(df, c=5):
    COLUMNS = c
    ROWS = df.shape[0]

    return ROWS, COLUMNS


def get_fig_subplots(title_list: Iterable[str], ROWS: int, COLUMNS: int):
    titles = reduce(lambda x, y: x + y, [[value] * COLUMNS for value in title_list])

    fig = make_subplots(
        rows=ROWS,
        cols=COLUMNS,
        subplot_titles=titles,
        specs=[[{"type": "polar"} for _ in range(COLUMNS)] for _ in range(ROWS)],
    )

    return fig


def get_theta_list(df, idx: Any, origin_columns: Iterable[str]):
    return [
        f"{col} - {round(value, ndigits=2)} | {int(standing)}"
        for col, value, standing in zip(
            origin_columns,
            df.loc[idx, origin_columns].values,
            df[origin_columns].rank(method="min", ascending=False).loc[idx].values,
        )
    ]


def get_team_stats_columns():
    origin_columns = {
        "combat": ["kd", "firstblood", "team kpm", "dpm", "damagetakenperminute"],
        "objects": [
            "firstdragon",
            "dragons",
            "firstherald",
            "heralds",
            "firstbaron",
            "barons",
            "elders",
        ],
        "towers": [
            "firsttower",
            "towers",
            "firstmidtower",
            "firsttothreetowers",
            "turretplates",
        ],
        "macro": [
            "gamelength",
            "win_gamelength",
            "loss_gamelength",
            "gpm",
            "gdpm",
            "wpm",
            "wcpm",
            "vspm",
        ],
        "early": [
            "golddiffat10",
            "xpdiffat10",
            "csdiffat10",
            "golddiffat15",
            "xpdiffat15",
            "csdiffat15",
        ],
    }
    columns = {}

    for key, values in origin_columns.items():
        columns[key] = ["_" + col for col in values]

    return origin_columns, columns


def get_player_stats_columns():
    origin_columns = {
        "combat": [
            "kills",
            "deaths",
            "assists",
            "kda",
            # "doublekills",
            # "triplekills",
            # "quadrakills",
            # "pentakills",
            "dpm",
            "damageshare",
            "damagetakenperminute",
        ],
        "macro": [
            "firstblood",
            "firstbloodvictim",
            "wpm",
            "wcpm",
            "controlwardsbought",
            "vspm",
            "gpm",
            "gdpm",
            "goldshare",
            "cspm",
            "champions",
        ],
        "early": [
            "golddiffat10",
            "xpdiffat10",
            "csdiffat10",
            "golddiffat15",
            "xpdiffat15",
            "csdiffat15",
        ],
    }
    columns = {}

    for key, values in origin_columns.items():
        columns[key] = ["_" + col for col in values]

    return origin_columns, columns


# %%
matches_data = pd.read_csv(
    "./csv/oracleselixir_match_data/2023_LoL_esports_match_data_from_OraclesElixir.csv",
    dtype={"patch": "object"},
)
matches_data["kda"] = (
    matches_data[["kills", "assists"]].sum(axis=1).divide(matches_data["deaths"])
)
matches_data["kd"] = matches_data["kills"].divide(matches_data["deaths"])
matches_data["gpm"] = matches_data["totalgold"].divide(matches_data["gamelength"] / 60)
matches_data.reset_index(drop=False, inplace=True)
matches_data.shape

# %%
columns = [
    "index",
    "gameid",
    "teamid",
    "participantid",
    "position",
    "gamelength",
    "totalgold",
]
merged = pd.merge(
    matches_data[columns],
    matches_data[columns],
    on=["gameid", "position"],
    how="inner",
)
merged = merged.loc[merged["participantid_x"] != merged["participantid_y"]]
merged.head(20)

# %%
matches_data["gdpm"] = 0
for row in merged.itertuples():
    gdpm = (row.totalgold_x - row.totalgold_y) / (row.gamelength_x / 60)
    matches_data.loc[row.index_x, "gdpm"] = gdpm
    matches_data.loc[row.index_y, "gdpm"] = -gdpm
matches_data["gdpm"]

# %%
columns = ["index", "gameid", "position", "playername", "teamname", "totalgold"]
merged = pd.merge(
    matches_data[columns], matches_data[columns], how="inner", on=["gameid", "teamname"]
)
merged = merged.loc[merged["position_y"] == "team"]
merged["goldshare"] = merged["totalgold_x"].divide(merged["totalgold_y"])
merged.head(10)

# %%
for row in merged.itertuples():
    matches_data.loc[row.index_x, "goldshare"] = row.goldshare

# %%
matches_data.columns

# %%
matches_data.head()

# %%
matches = get_matches(matches_data, "LCK", "Summer")
matches.head()

# %%
teams_data = matches.loc[matches["position"] == "team"]
df = pd.read_csv("./csv/scoreboard_games/2023_scoreboard_games.csv")
df = df.loc[df["OverviewPage"] == "LCK/2023 Season/Summer Season"]
print(teams_data.shape, df.shape, teams_data.shape[0] == df.shape[0] * 2)
teams_data.head()

# %%
teams_data[
    ["gameid", "teamname", "gamelength", "totalgold", "gpm", "earned gpm", "gdpm"]
]

# %%
cond = np.isinf(teams_data["kd"])
teams_data.loc[cond, "kd"] = teams_data.loc[cond, "kills"]
teams_data.loc[cond, "kd"]

# %%
teams_id = teams_data["teamid"].unique()
groupby = teams_data.groupby("teamid")
teams_stats = pd.DataFrame(index=teams_id)
teams_stats["Team"] = groupby.last()["teamname"]

columns = [
    "gamelength",
    "kills",
    "deaths",
    "assists",
    "kd",
    "firstblood",
    "team kpm",
    "ckpm",
    "firstdragon",
    "dragons",
    "opp_dragons",
    "elders",
    "opp_elders",
    "firstherald",
    "heralds",
    "opp_heralds",
    "firstbaron",
    "barons",
    "opp_barons",
    "firsttower",
    "towers",
    "opp_towers",
    "firstmidtower",
    "firsttothreetowers",
    "turretplates",
    "opp_turretplates",
    "inhibitors",
    "opp_inhibitors",
    "dpm",
    "damagetakenperminute",
    "wpm",
    "wcpm",
    "vspm",
    "gpm",
    "gdpm",
    "cspm",
    "golddiffat10",
    "xpdiffat10",
    "csdiffat10",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
]
teams_stats[columns] = groupby[columns].mean()
teams_stats["win_gamelength"] = (
    teams_data.loc[teams_data["result"] == 1].groupby("teamid")["gamelength"].mean()
)
teams_stats["loss_gamelength"] = (
    teams_data.loc[teams_data["result"] == 0].groupby("teamid")["gamelength"].mean()
)
teams_stats[["gamelength", "win_gamelength", "loss_gamelength"]] /= 60


teams_stats.set_index("Team", inplace=True)
teams_stats

# %%
origin_columns, columns = get_team_stats_columns()

for key, values in origin_columns.items():
    scaler = MinMaxScaler()
    df = pd.DataFrame(
        scaler.fit_transform(teams_stats[values]),
        columns=columns[key],
        index=teams_stats.index.values,
    )

    teams_stats = pd.concat([teams_stats, df], axis=1)
teams_stats.sort_index(inplace=True)
teams_stats.head()

# %%
teams_stats

# %%
ROWS, COLUMNS = get_rows_columns(teams_stats, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(teams_stats.index.values, ROWS, COLUMNS)
for i, team_name in enumerate(teams_stats.index):
    for j, (key, cols) in enumerate(columns.items()):
        df = teams_stats.loc[team_name, cols]
        theta_list = get_theta_list(teams_stats, team_name, origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=df, theta=theta_list, fill="toself", name=team_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=400 * ROWS,
    width=550 * COLUMNS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
players_data = matches.loc[~matches["playername"].isna()]
players_data.head()

# %%
cond = np.isinf(players_data["kda"])
players_data.loc[cond, "kda"] = players_data.loc[cond, ["kills", "assists"]].sum(axis=1)
players_data.loc[cond, "kda"]

# %%
groupby = players_data.groupby(["playerid", "position"])
idx = groupby.groups.keys()
idx = pd.MultiIndex.from_tuples(idx, names=["playerid", "position"])
players_stats = pd.DataFrame(index=idx)
players_stats[["Player", "Team"]] = groupby.last()[["playername", "teamname"]]

mean_columns = [
    "kills",
    "deaths",
    "assists",
    "kda",
    "doublekills",
    "triplekills",
    "quadrakills",
    "pentakills",
    "firstblood",
    "firstbloodvictim",
    "dpm",
    "damageshare",
    "damagetakenperminute",
    "wpm",
    "wcpm",
    "controlwardsbought",
    "vspm",
    "gpm",
    "gdpm",
    "goldshare",
    "cspm",
    "golddiffat10",
    "xpdiffat10",
    "csdiffat10",
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
]
num_columns = ["champion"]

players_stats[mean_columns] = groupby[mean_columns].mean()
players_stats["champions"] = groupby["champion"].nunique()
players_stats = (
    players_stats.reset_index().set_index(["Player", "position"]).sort_values("Team")
)
players_stats.head()

# %%
origin_columns, columns = get_player_stats_columns()

positions = ["top", "jng", "mid", "bot", "sup"]

# %%
for key, values in origin_columns.items():
    for pos in positions:
        scaler = MinMaxScaler()
        cond = players_stats.index.get_level_values(1) == pos
        players_stats.loc[cond, columns[key]] = scaler.fit_transform(
            players_stats.loc[cond, values]
        )
players_stats.head()

# %%
position = positions[0]
df = players_stats.loc[players_stats.index.get_level_values(1) == position]

ROWS, COLUMNS = get_rows_columns(df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
position = positions[1]
df = players_stats.loc[players_stats.index.get_level_values(1) == position]

ROWS, COLUMNS = get_rows_columns(df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
position = positions[2]
df = players_stats.loc[players_stats.index.get_level_values(1) == position]

ROWS, COLUMNS = get_rows_columns(df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
position = positions[3]
df = players_stats.loc[players_stats.index.get_level_values(1) == position]

ROWS, COLUMNS = get_rows_columns(df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
position = positions[4]
df = players_stats.loc[players_stats.index.get_level_values(1) == position]

ROWS, COLUMNS = get_rows_columns(df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%

# %%

# %%

# %%
origin_columns, columns = get_team_stats_columns()
target_team = "T1"
target_df = teams_stats.loc[[target_team]]

ROWS, COLUMNS = get_rows_columns(target_df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(target_df.index.values, ROWS, COLUMNS)
for i, team_name in enumerate(target_df.index):
    for j, (key, cols) in enumerate(columns.items()):
        df = target_df.loc[team_name, cols]
        theta_list = get_theta_list(target_df, team_name, origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=df, theta=theta_list, fill="toself", name=team_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=400 * ROWS,
    width=550 * COLUMNS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
players_stats.head()

# %%
df
position = df.loc["Zeus"].index.values[0]
position

# %%
origin_columns, columns = get_player_stats_columns()

position = positions[4]

target_df = players_stats.loc[players_stats["Team"] == target_team].sort_values(
    by=["position"], key=lambda x: [positions.index(pos) for pos in x]
)

ROWS, COLUMNS = get_rows_columns(target_df, len(origin_columns))
min_value, max_value = 0, 1

fig = get_fig_subplots(target_df.index.get_level_values(0), ROWS, COLUMNS)
for i, player_name in enumerate(df.index.get_level_values(0)):
    position = target_df.loc[player_name].index.values[0]
    df = players_stats.loc[players_stats.index.get_level_values(1) == position]
    for j, (key, cols) in enumerate(columns.items()):
        _df = df.loc[(player_name, position), cols]
        theta_list = get_theta_list(df, (player_name, position), origin_columns[key])

        fig.add_trace(
            go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
            row=i + 1,
            col=j + 1,
        )

fig.update_layout(
    height=350 * ROWS, width=550 * COLUMNS, showlegend=False, title=dict(y=1)
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

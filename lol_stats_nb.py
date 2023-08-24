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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lol_fandom import get_tournaments, get_match_schedule


pd.set_option("display.max_columns", None)


# %%
def get_rc(i, cols):
    r = i // cols + 1
    c = i % cols + 1

    return (r, c)


def get_rows_columns(df, c=5):
    COLUMNS = c
    d, m = divmod(df.shape[0], COLUMNS)
    ROWS = d + (1 if m > 0 else m)

    return ROWS, COLUMNS


def get_fig_subplots(df, ROWS, COLUMNS):
    fig = make_subplots(
        rows=ROWS,
        cols=COLUMNS,
        subplot_titles=df.index.values,
        specs=[[{"type": "polar"} for _ in range(COLUMNS)] for _ in range(ROWS)],
    )

    return fig


def get_min_max_value(df, columns):
    min_value = df[columns].unstack().min()
    max_value = df[columns].unstack().max()

    return min_value, max_value


def get_theta_list(df, name, origin_columns):
    return [
        f"{col} - {round(value, ndigits=2)} | {int(standing)}"
        for col, value, standing in zip(
            origin_columns,
            df.loc[name, origin_columns].values,
            df[origin_columns].rank(method="min", ascending=False).loc[name].values,
        )
    ]


# %%
# SCALER = StandardScaler
SCALER = MinMaxScaler
# SCALER = MaxAbsScaler

# %%
teams_stats = pd.read_csv("./csv/stats/teams.csv", index_col="Team")
print(teams_stats.shape)
teams_stats["K%"] = teams_stats["KPM"].divide(teams_stats["CKPM"])
teams_stats.head()

# %%
scaler = SCALER()
origin_columns = ["KD", "CKPM", "KPM", "K%", "GPM", "GDPM"]
columns = ["_" + col for col in origin_columns]
teams_stats[columns] = scaler.fit_transform(teams_stats[origin_columns])
teams_stats.head()

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.set_xlim(teams_stats["GDPM"].min() - 25, teams_stats["GDPM"].max() + 25)
ax.set_ylim(teams_stats["KPM"].min() - 0.01, teams_stats["KPM"].max() + 0.01)
ax = sns.regplot(data=teams_stats, x="GDPM", y="KPM")
y = np.linspace(0, 1.2)
x = 100 / y
sns.lineplot(x=x, y=y)
for idx, i in zip(teams_stats.index, range(teams_stats.shape[0])):
    row = teams_stats.iloc[i]
    plt.annotate(
        idx, xy=(row["GDPM"], row["KPM"]), xytext=(5, 20), textcoords="offset pixels"
    )
    plt.annotate(
        round(row["GDPM"] * row["KPM"], ndigits=1),
        xy=(row["GDPM"], row["KPM"]),
        xytext=(5, 5),
        textcoords="offset pixels",
    )

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.set_xlim(teams_stats["GDPM"].min() - 25, teams_stats["GDPM"].max() + 25)
ax.set_ylim(teams_stats["K%"].min() - 0.01, teams_stats["K%"].max() + 0.01)
ax = sns.regplot(data=teams_stats, x="GDPM", y="K%")
y = np.linspace(0, 1.2)
x = 100 / y
sns.lineplot(x=x, y=y)
for idx, i in zip(teams_stats.index, range(teams_stats.shape[0])):
    row = teams_stats.iloc[i]
    plt.annotate(
        idx, xy=(row["GDPM"], row["K%"]), xytext=(5, 20), textcoords="offset pixels"
    )
    plt.annotate(
        round(row["GDPM"] * row["K%"], ndigits=1),
        xy=(row["GDPM"], row["K%"]),
        xytext=(5, 5),
        textcoords="offset pixels",
    )

# %%
ROWS, COLUMNS = get_rows_columns(teams_stats)
min_value, max_value = get_min_max_value(teams_stats, columns)

fig = get_fig_subplots(teams_stats, ROWS, COLUMNS)
for i, team_name in enumerate(teams_stats.index):
    df = teams_stats.loc[team_name, columns]
    theta_list = get_theta_list(teams_stats, team_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=df, theta=theta_list, fill="toself", name=team_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
players_stats = pd.read_csv("./csv/stats/players_by_position.csv", index_col="Player")
players_stats.shape

# %%
players_stats.columns

# %%
players_stats["DPG"] = players_stats["DPM"].divide(players_stats["GPM"])

# %%
origin_columns = [
    "KDA",
    "KP",
    "KS",
    "DPM",
    "DPG",
    "CSPM",
    "GPM",
    "GS",
    "ChampionsPlayed",
]
columns = ["_" + col for col in origin_columns]
positions = ["Top", "Jungle", "Mid", "Bot", "Support"]

# %%
for position in positions:
    scaler = SCALER()
    df = players_stats.loc[players_stats["Position"] == position]
    players_stats.loc[
        players_stats["Position"] == position, columns
    ] = scaler.fit_transform(df[origin_columns])

# %%
df = players_stats.loc[players_stats["Position"] == positions[0]]
min_value, max_value = get_min_max_value(df, columns)

ROWS, COLUMNS = get_rows_columns(df, 4)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    width=500 * COLUMNS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = players_stats.loc[players_stats["Position"] == positions[1]]
min_value, max_value = get_min_max_value(df, columns)

ROWS, COLUMNS = get_rows_columns(df, 4)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    width=500 * COLUMNS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = players_stats.loc[players_stats["Position"] == positions[2]]
min_value, max_value = get_min_max_value(df, columns)

ROWS, COLUMNS = get_rows_columns(df, 4)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = players_stats.loc[players_stats["Position"] == positions[3]]
min_value, max_value = get_min_max_value(df, columns)

ROWS, COLUMNS = get_rows_columns(df, 4)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = players_stats.loc[players_stats["Position"] == positions[4]]
min_value, max_value = get_min_max_value(df, columns)

ROWS, COLUMNS = get_rows_columns(df, 4)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False),
)
fig.update_traces(marker=dict(), selector=dict(type="scatterpolar"))
fig.show()

# %%

# %%
leagues = {
    "LCK": "LoL Champions Korea",
    "LPL": "Tencent LoL Pro League",
    "LEC": "LoL EMEA Championship",
    "LCS": "League of Legends Championship Series",
    "MSI": "Mid-Season Invitational",
    "WCS": "World Championship",
}

tournaments = get_tournaments(where=f'T.Year=2023 and T.League="{leagues["LCK"]}"')
tournaments

# %%
page = "LCK/2023 Season/Summer Season"
match_schedules = get_match_schedule(where=f'MS.OverviewPage="{page}"').sort_values(
    by=["DateTime UTC"], ignore_index=True
)
match_schedules = match_schedules.loc[match_schedules["Winner"].isna()].reset_index(
    drop=True
)
match_schedules.head()

# %%
team_names = reduce(
    lambda x, y: x + y,
    map(
        lambda x: [x.Team1, x.Team2],
        match_schedules.head(2)[["Team1", "Team2"]].itertuples(),
    ),
)
team_names

# %%
target_teams_stats = teams_stats.loc[team_names]
target_teams_stats

# %%
origin_columns = ["KD", "CKPM", "KPM", "GPM", "GDPM"]
columns = ["_" + col for col in origin_columns]

# %%
ROWS, COLUMNS = get_rows_columns(target_teams_stats, c=2)
min_value, max_value = 0, 1

fig = get_fig_subplots(target_teams_stats, ROWS, COLUMNS)
for i, team_name in enumerate(target_teams_stats.index):
    df = target_teams_stats.loc[team_name, columns]
    theta_list = get_theta_list(target_teams_stats, team_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=df, theta=theta_list, fill="toself", name=team_name),
        row=r,
        col=c,
    )

fig.update_layout(height=400 * ROWS, showlegend=False, title=dict(y=1))
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
origin_columns = [
    "KDA",
    "KP",
    "KS",
    "DPM",
    "DPG",
    "CSPM",
    "GPM",
    "GS",
    "ChampionsPlayed",
]
columns = ["_" + col for col in origin_columns]
positions = ["Top", "Jungle", "Mid", "Bot", "Support"]

# %%
target_players_stats = players_stats.loc[
    players_stats["Team"].isin(team_names)
].sort_values(by=["Team"], key=lambda x: [team_names.index(value) for value in x])
target_players_stats.head()

# %%
df = target_players_stats.loc[target_players_stats["Position"] == positions[0]]
min_value, max_value = 0, 1

ROWS, COLUMNS = get_rows_columns(df, 2)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = target_players_stats.loc[target_players_stats["Position"] == positions[1]]
min_value, max_value = 0, 1

ROWS, COLUMNS = get_rows_columns(df, 2)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = target_players_stats.loc[target_players_stats["Position"] == positions[2]]
min_value, max_value = 0, 1

ROWS, COLUMNS = get_rows_columns(df, 2)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = target_players_stats.loc[target_players_stats["Position"] == positions[3]]
min_value, max_value = 0, 1

ROWS, COLUMNS = get_rows_columns(df, 2)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%
df = target_players_stats.loc[target_players_stats["Position"] == positions[4]]
min_value, max_value = 0, 1

ROWS, COLUMNS = get_rows_columns(df, 2)

fig = get_fig_subplots(df, ROWS, COLUMNS)
for i, player_name in enumerate(df.index):
    _df = df.loc[player_name, columns]
    theta_list = get_theta_list(df, player_name, origin_columns)

    r, c = get_rc(i, COLUMNS)
    fig.add_trace(
        go.Scatterpolar(r=_df, theta=theta_list, fill="toself", name=player_name),
        row=r,
        col=c,
    )

fig.update_layout(
    height=400 * ROWS,
    showlegend=False,
    title=dict(y=1),
)
fig.update_annotations(yshift=20)
fig.update_polars(
    radialaxis=dict(range=[min_value - 0.2, max_value + 0.2], visible=False)
)
fig.show()

# %%

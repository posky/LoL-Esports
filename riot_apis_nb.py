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
from pprint import pprint
from copy import deepcopy

import pandas as pd

from riot_apis import RiotAPI

pd.set_option("display.max_columns", None)

# %%
riot_api = RiotAPI()

# %%
summoners = pd.read_csv("./csv/solo_rank/summoners.csv")
summoners

# %%
participants = pd.read_csv("./csv/solo_rank/participants.csv")
participants

# %%
info = pd.read_csv("./csv/solo_rank/info.csv")
info

# %%
stats = pd.read_csv("./csv/solo_rank/stats.csv", index_col=["player", "teamPosition"])
stats

# %%
if "Zeus" in stats.index.get_level_values("player"):
    zeus_stats = (
        stats.loc["Zeus"]
        .reset_index()
        .sort_values(["teamPosition", "games", "winrate"], ascending=False)
    )
else:
    zeus_stats = pd.DataFrame()
zeus_stats

# %%
if "Oner" in stats.index.get_level_values("player"):
    oner_stats = (
        stats.loc["Oner"]
        .reset_index()
        .sort_values(["teamPosition", "games", "winrate"], ascending=False)
    )
else:
    oner_stats = pd.DataFrame()
oner_stats

# %%
if "Faker" in stats.index.get_level_values("player"):
    faker_stats = (
        stats.loc["Faker"]
        .reset_index()
        .sort_values(["teamPosition", "games", "winrate"], ascending=False)
    )
else:
    faker_stats = pd.DataFrame()
faker_stats

# %%
if "Gumayusi" in stats.index.get_level_values("player"):
    gumayusi_stats = (
        stats.loc["Gumayusi"]
        .reset_index()
        .sort_values(["teamPosition", "games", "winrate"], ascending=False)
    )
else:
    gumayusi_stats = pd.DataFrame()
gumayusi_stats

# %%
if "Keria" in stats.index.get_level_values("player"):
    keria_stats = (
        stats.loc["Keria"]
        .reset_index()
        .sort_values(["teamPosition", "games", "winrate"], ascending=False)
    )
else:
    keria_stats = pd.DataFrame()
keria_stats

# %%

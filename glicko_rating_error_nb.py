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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from glicko_rating_matches import DEFAULT_SIGMA, DEFAULT_TAU


# %%
def get_error(ratings):
    mae = ratings["error"].sum() / ratings["games"].sum()
    mse = ratings["error_square"].sum() / ratings["games"].sum()
    return mae, mse


# %%
print(DEFAULT_SIGMA, DEFAULT_TAU)

# %%
ratings = pd.read_csv(
    "./csv/glicko_rating/glicko_rating_2023.csv", parse_dates=["last_game_date"]
)
ratings

# %%
mae, mse = get_error(ratings)
print(mae, mse)

# %%
columns = ["tau", "sigma", "mae", "mse"]
columns

# %%
error_df = pd.read_csv("./csv/glicko_rating_error.csv")
error_df

# %%
df = pd.DataFrame([[DEFAULT_TAU, DEFAULT_SIGMA, mae, mse]], columns=columns)
df

# %%
error_df = pd.concat([error_df, df], ignore_index=True)
error_df

# %%
error_df.loc[error_df["mae"] == error_df["mae"].min()]

# %%
error_df.loc[error_df["mse"] == error_df["mse"].min()]

# %%
error_df.to_csv("./csv/glicko_rating_error.csv", index=False)

# %%
sns.regplot(error_df, x="tau", y="mae")

# %%
sns.regplot(error_df, x="sigma", y="mae")

# %%

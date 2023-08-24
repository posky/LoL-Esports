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
#     display_name: Python 3.8.5 ('base')
#     language: python
#     name: python3
# ---

# %%
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns', None)

DATA_PATH = os.path.join(os.environ['DEV_PATH'], 'datasets/LoL_esports')
LEAGUES = ['worlds', 'msi', 'lck', 'lpl', 'lec', 'lcs']


# %%
def feature_engineering(df):
    df['KD'] = df['TeamKills'] / df['OpponentKills']
    df.loc[df['KD'] == np.inf, 'KD'] = df.loc[df['KD'] == np.inf, 'TeamKills']
    df['CKPM'] = df[['TeamKills', 'OpponentKills']].sum(axis=1) / df['Gamelength Number']
    df['KPM'] = df['TeamKills'] / df['Gamelength Number']
    df['GPM'] = df['TeamGold'] / df['Gamelength Number']
    df['GDPM'] = (df['TeamGold'] - df['OpponentGold']) / df['Gamelength Number']
    df['GDP'] = (df['TeamGold'] - df['OpponentGold']) / df[['TeamGold', 'OpponentGold']].mean(axis=1)
    df['KDPM'] = (df['TeamKills'] - df['OpponentKills']) / df['Gamelength Number']
    df['Outlier'] = df['KPM'] * df['GDPM']
    df['Outlier2'] = df['KDPM'] * df['GDPM']
    return df

def extract_teams(df):
    grouped = df.groupby(['OverviewPage', 'Team'])
    teams = pd.DataFrame()
    teams['Win'] = grouped['Win'].sum()
    teams['Loss'] = grouped['Loss'].sum()
    teams['Gold'] = grouped['TeamGold'].mean()
    teams['Kills'] = grouped['TeamKills'].mean()
    teams['Deaths'] = grouped['OpponentKills'].mean()
    teams['Gamelength Number'] = grouped['Gamelength Number'].mean()
    teams['Win Gamelength Number'] = \
        grouped[['Win', 'Gamelength Number']].apply(
            lambda x: (x['Win'] * x['Gamelength Number']).sum()
        ) / teams['Win']
    teams['Loss Gamelength Number'] = \
        grouped[['Loss', 'Gamelength Number']].apply(
            lambda x: (x['Loss'] * x['Gamelength Number']).sum()
        ) / teams['Loss']
    teams.loc[teams['Win'] == 0, 'Win Gamelength Number'] = -1
    teams.loc[teams['Loss'] == 0, 'Loss Gamelength Number'] = -1
    teams['Games'] = grouped['Win'].count()
    teams['WinRate'] = teams['Win'] / teams['Games']
    teams['KD'] = grouped['KD'].mean()
    teams['CKPM'] = grouped['CKPM'].mean()
    teams['KPM'] = grouped['KPM'].mean()
    teams['GPM'] = grouped['GPM'].mean()
    teams['GDPM'] = grouped['GDPM'].mean()
    teams['GDP'] = grouped['GDP'].mean()
    teams['KDPM'] = grouped['KDPM'].mean()
    teams['Outlier'] = grouped['Outlier'].mean()
    teams['Outlier2'] = grouped['Outlier2'].mean()

    columns = [
        'Games', 'Win', 'Loss', 'WinRate', 'Gamelength Number',
        'Win Gamelength Number', 'Loss Gamelength Number', 'Gold', 'Kills',
        'Deaths', 'KD', 'CKPM', 'KPM', 'GPM', 'GDPM', 'GDP', 'KDPM',
        'Outlier', 'Outlier2'
    ]
    return teams[columns]



# %%
games = pd.read_csv(os.path.join(DATA_PATH, 'major_matches.csv'))
winners = pd.read_csv(os.path.join(DATA_PATH, 'major_matches_winners.csv'))

games.shape, winners.shape

# %%
games.info()

# %%
games = feature_engineering(games)
games

# %%
teams = extract_teams(games)
teams

# %%
teams['Winner'] = 0
teams.loc[winners.apply(tuple, axis=1), 'Winner'] = 1
teams = teams.reset_index()
teams

# %%
teams = teams.loc[teams[['Gold', 'Kills', 'Deaths']].dropna().index]
teams

# %%
columns = [
    'Games', 'WinRate', 'Gamelength Number', 'Win Gamelength Number',
    'Loss Gamelength Number', 'KD', 'CKPM', 'KPM', 'GPM', 'GDPM',
    'GDP', 'KDPM', 'Outlier', 'Outlier2'
]
X = teams[columns]
y = teams['Winner']

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)

X_trainval.shape, y_trainval.shape, X_test.shape, y_test.shape

# %%
fold = RepeatedStratifiedKFold(random_state=0)

# %%
model = RandomForestClassifier(random_state=0)
scores = cross_val_score(model, X_trainval, y_trainval, scoring='roc_auc', cv=fold, n_jobs=-1)
scores.mean()

# %%
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(random_state=0))])
scores = cross_val_score(pipe, X_trainval, y_trainval, cv=fold, scoring='roc_auc', n_jobs=-1)
scores.mean()

# %%
model = GradientBoostingClassifier(random_state=0)
scores = cross_val_score(model, X_trainval, y_trainval, scoring='roc_auc', cv=fold, n_jobs=-1)
scores.mean()

# %%
param_grid = {
    'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'model__solver': ['lbfgs', 'liblinear', 'sag', 'saga']
}
grid_search = GridSearchCV(pipe, param_grid, cv=fold, return_train_score=True, n_jobs=-1, scoring='roc_auc')

# %%
grid_search.fit(X_trainval, y_trainval)

# %%
print(grid_search.best_params_)
print(grid_search.best_score_)

# %%
results = pd.DataFrame(grid_search.cv_results_)

# %%
scores = np.array(results.mean_test_score).reshape(6, 4)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.heatmap(
    scores,
    xticklabels=['lbfgs', 'liblinear', 'sag', 'saga'],
    yticklabels=[0.001, 0.01, 0.1, 1, 10, 100]
)

# %%
logreg = LogisticRegression(C=10, solver='sag', random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('logreg', logreg)])
pipe.fit(X_trainval, y_trainval)
pipe.score(X_trainval, y_trainval)

# %%
pipe.score(X_test, y_test)

# %%
feature_importance = pd.DataFrame(data=np.abs(pipe['logreg'].coef_), columns=columns)
feature_importance

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(x=feature_importance.values.reshape(-1), y=feature_importance.columns)
plt.title('Feature Importance')

# %%
current_games = pd.read_csv(os.path.join(DATA_PATH, '2022_worlds_matches.csv'))
current_games.shape

# %%
current_games = feature_engineering(current_games)
current_games

# %%
current_teams = extract_teams(current_games)
current_teams

# %%
pred_y = pipe.predict_proba(current_teams[columns])
pred_y

# %%
current_teams['Win_prob'] = pred_y[:, 1]
current_teams = current_teams.sort_values(by='Win_prob', ascending=False).reset_index()
current_teams

# %%
team_names = [
    'T1', 'DRX'
]
# current_teams.index.levels[1]
target_teams = current_teams.loc[current_teams['Team'].isin(team_names)]
target_teams


# %%
def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x)

    return exp_x / sum_exp_x

target_teams['Win_Probability'] = softmax(target_teams['Win_prob'])
target_teams[['Team', 'Win_Probability']]

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
sns.barplot(data=target_teams, x='Team', y='Win_Probability')
plt.xticks(rotation=45)

# %%

import os
import requests
import logging
from datetime import datetime

import pandas as pd
from ratelimit import limits, sleep_and_retry
from furl import furl


DIR_PATH = "./csv/solo_rank"


class RiotAPI:
    HOST_URL = "https://{}.api.riotgames.com"

    def __init__(self, api_key=None, platform="kr", region="asia"):
        logging.info("RiotAPI created.")
        if api_key is None:
            with open("./riot_api_key.txt", "r") as f:
                api_key = f.read().strip()
        self.headers = {"X-Riot-Token": api_key}
        self.platform_host = self.HOST_URL.format(platform)
        self.region_host = self.HOST_URL.format(region)
        self.summoner = self.Summoner(self)
        self.match = self.Match(self)

    @sleep_and_retry
    @limits(calls=100, period=125)
    def get_data(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"API response: {response.status_code}")
        return response.json()

    def get_data_by_platform(self, url_contents):
        url = furl(self.platform_host)
        url /= url_contents.get("path", "")
        url.add(args=url_contents.get("query", {}))
        return self.get_data(url.url)

    def get_data_by_region(self, url_contents):
        url = furl(self.region_host)
        url /= url_contents.get("path", "")
        url.add(args=url_contents.get("query", {}))
        return self.get_data(url.url)

    class Summoner:
        def __init__(self, riot_api):
            self.url = "/lol/summoner/v4/summoners/"
            self.riot_api = riot_api

        def by_account_id(self, account_id):
            endpoint = self.url + f"by-account/{account_id}"
            url_contents = {"path": endpoint}
            return self.riot_api.get_data_by_platform(url_contents)

        def by_name(self, summoner_name):
            endpoint = self.url + f"by-name/{summoner_name}"
            url_contents = {"path": endpoint}
            return self.riot_api.get_data_by_platform(url_contents)

        def by_puuid(self, puuid):
            endpoint = self.url + f"by-puuid/{puuid}"
            url_contents = {"path": endpoint}
            return self.riot_api.get_data_by_platform(url_contents)

    class Match:
        def __init__(self, riot_api):
            self.url = "/lol/match/v5/matches/"
            self.riot_api = riot_api

        def by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}"
            url_contents = {"path": endpoint}
            return self.riot_api.get_data_by_region(url_contents)

        def ids_by_puuid(self, puuid, **kwargs):
            endpoint = self.url + f"by-puuid/{puuid}/ids"
            url_contents = {"path": endpoint, "query": kwargs}
            return self.riot_api.get_data_by_region(url_contents)

        def timeline_by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}/timeline"
            url_contents = {"path": endpoint}
            return self.riot_api.get_data_by_region(url_contents)


def get_dataframe_from_csv(file_name, columns=[]):
    file_path = os.path.join(DIR_PATH, file_name)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)
    return df


def add_summoner(riot_api, name: str, player: str):
    summoner = riot_api.summoner.by_name(name)
    summoner["player"] = player
    update_summoner(summoner, save=True)


def update_summoner(
    summoner: dict,
    summoners_df: pd.DataFrame = None,
    save: bool = False,
):
    if summoners_df is None:
        summoners_df = get_dataframe_from_csv("summoners.csv")
    cond = summoners_df["puuid"] == summoner["puuid"]
    df = summoners_df.loc[cond]
    if df.shape[0] == 0:
        logging.info("Added summoner %s (%s)", summoner["name"], summoner["player"])
        summoners_df = pd.concat([summoners_df, pd.DataFrame([summoner])])
    elif df["revisionDate"].iloc[0] < summoner["revisionDate"]:
        logging.info("Updated summoner %s (%s)", summoner["name"], summoner["player"])
        summoners_df.loc[cond, summoner.keys()] = summoner.values()
    if save:
        save_to_csv(summoners_df, "summoners.csv")
    return summoners_df


def update_summoners(riot_api):
    summoners = get_dataframe_from_csv("summoners.csv")

    for i in range(summoners.shape[0]):
        summoner = riot_api.summoner.by_puuid(summoners["puuid"].iloc[i])
        summoner["player"] = summoners["player"].iloc[i]
        summoners = update_summoner(summoner, summoners)
    save_to_csv(summoners, "summoners.csv")


def update_match_ids(riot_api):
    summoners = get_dataframe_from_csv("summoners.csv", ["puuid"])
    match_ids_df = get_dataframe_from_csv("match_ids.csv", ["matchId"])

    new_match_ids = []
    for row in summoners.itertuples():
        match_ids = riot_api.match.ids_by_puuid(row.puuid, count=100)
        ids = [
            match_id
            for match_id in match_ids
            if match_id not in match_ids_df["matchId"].values
        ]
        logging.info("%s (%s): %d matches", row.name, row.player, len(ids))
        new_match_ids += ids

    new_match_ids = list(set(new_match_ids))
    new_match_ids_df = pd.DataFrame(new_match_ids, columns=["matchId"])
    match_ids_df = pd.concat([match_ids_df, new_match_ids_df])
    save_to_csv(match_ids_df, "match_ids.csv")


def update_match_data(riot_api, match_id):
    match_data = riot_api.match.by_match_id(match_id)
    metadata = pd.DataFrame(match_data["metadata"])
    _info = match_data["info"]
    _info["matchId"] = match_id
    _participants = _info.pop("participants")
    _teams = _info.pop("teams")
    for _t in _teams:
        _t["matchId"] = match_id
    info = pd.DataFrame([_info])
    teams = pd.DataFrame(_teams)

    _challenges = []
    _perks = []
    for _part in _participants:
        if "challenges" in _part:
            _challenges.append(_part.pop("challenges"))
            _challenges[-1]["matchId"] = match_id
            _challenges[-1]["puuid"] = _part["puuid"]
        if "perks" in _part:
            _perks.append(_part.pop("perks"))
            _perks[-1]["matchId"] = match_id
            _perks[-1]["puuid"] = _part["puuid"]
        _part["matchId"] = match_id
    participants = pd.DataFrame(_participants)
    challenges = pd.DataFrame(_challenges)
    perks = pd.DataFrame(_perks)

    logging.info("Updated %s match data", match_id)

    save_to_csv(metadata, "metadata.csv", concat=True)
    save_to_csv(info, "info.csv", concat=True)
    save_to_csv(teams, "teams.csv", concat=True)
    save_to_csv(participants, "participants.csv", concat=True)
    save_to_csv(challenges, "challenges.csv", concat=True)
    save_to_csv(perks, "perks.csv", concat=True)


def update_matches_data(riot_api):
    match_ids_df = get_dataframe_from_csv("match_ids.csv", ["matchId"])
    participants = get_dataframe_from_csv("participants.csv", ["matchId"])

    count = 0
    for match_id in match_ids_df["matchId"].unique():
        if match_id not in participants["matchId"].values:
            update_match_data(riot_api, match_id)
            count += 1
    logging.info("Updated %d matches", count)


def get_latest_matches_by_name(riot_api, name):
    _summoner = riot_api.summoner.by_name(name)
    summoner = pd.DataFrame([_summoner])

    metadata = pd.DataFrame()
    info = pd.DataFrame()
    teams = pd.DataFrame()
    participants = pd.DataFrame()
    challenges = pd.DataFrame()
    perks = pd.DataFrame()

    match_ids = riot_api.match.ids_by_puuid(summoner["puuid"].iloc[0])
    for match_id in match_ids:
        match_data = riot_api.match.by_match_id(match_id)
        metadata = pd.concat([metadata, pd.DataFrame(match_data["metadata"])])
        _info = match_data["info"]
        _info["matchId"] = match_id
        _participants = _info.pop("participants")
        _teams = _info.pop("teams")
        for _t in _teams:
            _t["matchId"] = match_id
        info = pd.concat([info, pd.DataFrame([_info])])
        teams = pd.concat([teams, pd.DataFrame(_teams)])

        _challenges = []
        _perks = []
        for _part in _participants:
            _challenges.append(_part.pop("challenges"))
            _perks.append(_part.pop("perks"))
            _part["matchId"] = match_id

            _challenges[-1]["matchId"] = match_id
            _challenges[-1]["puuid"] = _part["puuid"]
            _perks[-1]["matchId"] = match_id
            _perks[-1]["puuid"] = _part["puuid"]
        participants = pd.concat([participants, pd.DataFrame(_participants)])
        challenges = pd.concat([challenges, pd.DataFrame(_challenges)])
        perks = pd.concat([perks, pd.DataFrame(_perks)])

    metadata.reset_index(drop=True, inplace=True)
    info.reset_index(drop=True, inplace=True)
    teams.reset_index(drop=True, inplace=True)
    participants.reset_index(drop=True, inplace=True)
    challenges.reset_index(drop=True, inplace=True)
    perks.reset_index(drop=True, inplace=True)

    return summoner, metadata, info, teams, participants, challenges, perks


def save_to_csv(df, file_name, concat=False, index=False):
    file_path = os.path.join(DIR_PATH, file_name)
    if concat == True and os.path.isfile(file_path):
        original = pd.read_csv(file_path)
    else:
        original = pd.DataFrame()
    new_df = pd.concat([original, df])
    new_df.to_csv(file_path, index=index)
    logging.info("Saved %s (concat: %s, index: %s)", file_name, concat, index)


def select_options():
    summoners = pd.read_csv("./csv/solo_rank/summoners.csv")
    info = pd.read_csv("./csv/solo_rank/info.csv")
    participants = pd.read_csv("./csv/solo_rank/participants.csv")

    cond = (info["gameMode"] == "CLASSIC") & (info["gameDuration"] >= 4 * 60)
    merged = pd.merge(
        summoners[["puuid", "player"]],
        pd.merge(
            info.loc[cond],
            participants,
            how="inner",
            on="matchId",
        ),
        how="inner",
        on="puuid",
    )

    patch = input("Patch_version: ")
    if patch != "":
        patch_versions = patch.split(" ")
        merged = merged.loc[
            merged["gameVersion"]
            .str.split(".")
            .apply(lambda x: ".".join(x[:2]))
            .isin(patch_versions)
        ]

    logging.info("%d data", merged.shape[0])

    days = input("Days: ")
    if days != "":
        days = int(days)
        date_now = datetime.now()
        merged = merged.loc[
            merged["gameCreation"].apply(
                lambda x: (date_now - datetime.fromtimestamp(x / 1000)).days <= days
            )
        ]

    logging.info("%d data", merged.shape[0])

    return merged


def get_stats(matches_data):
    assert matches_data.shape[0] > 0

    grouped = matches_data.groupby(["player", "teamPosition", "championName"])
    stats = pd.DataFrame(
        columns=["games", "win", "loss", "winrate", "kills", "deaths", "assists", "kda"]
    )

    stats["games"] = grouped["championId"].count()
    stats["win"] = grouped["win"].sum()
    stats["loss"] = stats["games"] - stats["win"]
    stats["winrate"] = stats["win"].divide(stats["games"])
    stats[["kills", "deaths", "assists"]] = grouped[
        ["kills", "deaths", "assists"]
    ].mean()
    stats["kda"] = stats[["kills", "assists"]].sum(axis=1).divide(stats["deaths"])
    stats["gameDuration"] = grouped["gameDuration"].mean() / 60

    logging.info("Statistics completed")

    return stats


def main():
    logging.basicConfig(level=logging.INFO)

    riot_api = RiotAPI()

    update_summoners(riot_api)
    update_match_ids(riot_api)
    update_matches_data(riot_api)

    merged = select_options()
    stats = get_stats(merged)
    save_to_csv(stats, "stats.csv", index=True)


if __name__ == "__main__":
    main()

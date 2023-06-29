import os
import requests
from pprint import pprint
from urllib.parse import urljoin
import logging

import pandas as pd
from ratelimit import limits, sleep_and_retry


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
    @limits(calls=100, period=120)
    def get_data(self, url):
        response = requests.get(url, headers=self.headers)
        if response.status_code != 200:
            raise Exception(response.status_code)
        return response.json()

    def get_data_by_platform(self, endpoint):
        url = urljoin(self.platform_host, endpoint)
        return self.get_data(url)

    def get_data_by_region(self, endpoint):
        url = urljoin(self.region_host, endpoint)
        return self.get_data(url)

    class Summoner:
        def __init__(self, riot_api):
            self.url = "/lol/summoner/v4/summoners/"
            self.riot_api = riot_api

        def by_account_id(self, account_id):
            endpoint = self.url + f"by-account/{account_id}"
            return self.riot_api.get_data_by_platform(endpoint)

        def by_name(self, summoner_name):
            endpoint = self.url + f"by-name/{summoner_name}"
            return self.riot_api.get_data_by_platform(endpoint)

        def by_puuid(self, puuid):
            endpoint = self.url + f"by-puuid/{puuid}"
            return self.riot_api.get_data_by_platform(endpoint)

    class Match:
        def __init__(self, riot_api):
            self.url = "/lol/match/v5/matches/"
            self.riot_api = riot_api

        def by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}"
            return self.riot_api.get_data_by_region(endpoint)

        def ids_by_puuid(self, puuid):
            endpoint = self.url + f"by-puuid/{puuid}/ids"
            return self.riot_api.get_data_by_region(endpoint)

        def timeline_by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}/timeline"
            return self.riot_api.get_data_by_region(endpoint)


def get_dataframe_from_csv(file_name, columns=[]):
    file_path = os.path.join(DIR_PATH, file_name)
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path)
    else:
        df = pd.DataFrame(columns=columns)
    return df


def update_summoner(
    summoner: dict, summoners_df: pd.DataFrame = None, save: bool = False
):
    if summoners_df is None:
        summoners_df = get_dataframe_from_csv("summoners.csv")
    cond = summoners_df["puuid"] == summoner["puuid"]
    df = summoners_df.loc[cond]
    if df.shape[0] == 0:
        summoners_df = pd.concat([summoners_df, pd.DataFrame([summoner])])
    elif df["revisionDate"].iloc[0] < summoner["revisionDate"]:
        logging.info("Updated summoner %s / %s", summoner["name"], summoner["puuid"])
        summoners_df.loc[cond, summoner.keys()] = summoner.values()
    if save:
        save_to_csv(summoners_df, "summoners.csv")
    return summoners_df


def update_summoners(riot_api):
    summoners = get_dataframe_from_csv("summoners.csv")

    for i in range(summoners.shape[0]):
        summoner = riot_api.summoner.by_puuid(summoners["puuid"].iloc[i])
        summoners = update_summoner(summoner, summoners)
    save_to_csv(summoners, "summoners.csv")


def update_match_ids(riot_api):
    summoners = get_dataframe_from_csv("summoners.csv", ["puuid"])
    match_ids_df = get_dataframe_from_csv("match_ids.csv", ["matchId"])

    new_match_ids = []
    for row in summoners.itertuples():
        match_ids = riot_api.match.ids_by_puuid(row.puuid)
        ids = [
            match_id
            for match_id in match_ids
            if match_id not in match_ids_df["matchId"].values
        ]
        logging.info("%s / %s: %d matches", row.name, row.puuid, len(ids))
        new_match_ids += ids

    new_match_ids = list(set(new_match_ids))
    new_match_ids_df = pd.DataFrame(new_match_ids, columns=["matchId"])
    match_ids_df = pd.concat([match_ids_df, new_match_ids_df])
    save_to_csv(match_ids_df, "match_ids.csv")


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


def save_to_csv(df, file_name, concat=False):
    file_path = os.path.join(DIR_PATH, file_name)
    if concat == True and os.path.isfile(file_path):
        original = pd.read_csv(file_path)
    else:
        original = pd.DataFrame()
    new_df = pd.concat([original, df])
    new_df.to_csv(file_path, index=False)


def main():
    logging.basicConfig(level=logging.INFO)

    riot_api = RiotAPI()

    update_summoners(riot_api)
    update_match_ids(riot_api)


if __name__ == "__main__":
    main()

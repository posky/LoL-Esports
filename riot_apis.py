import pprint
import requests
from urllib.parse import urljoin

import pandas as pd


PLATFORM_KR_HOST = "https://kr.api.riotgames.com"
REGION_ASIA_HOST = "https://asia.api.riotgames.com"


class RiotAPI:
    def __init__(self, api_key=None):
        if api_key is None:
            with open("./riot_api_key.txt", "r") as f:
                api_key = f.read().strip()
        self.headers = {"X-Riot-Token": api_key}
        self.summoner = self.Summoner(self)
        self.match = self.Match(self)

    def get_data(self, url):
        response = requests.get(url, headers=self.headers)
        return response

    def get_data_platform(self, endpoint):
        url = urljoin(PLATFORM_KR_HOST, endpoint)
        return self.get_data(url)

    def get_data_region(self, endpoint):
        url = urljoin(REGION_ASIA_HOST, endpoint)
        return self.get_data(url)

    class Summoner:
        def __init__(self, riot_api):
            self.url = "/lol/summoner/v4/summoners/"
            self.riot_api = riot_api

        def by_account(self, account_id):
            endpoint = self.url + f"by-account/{account_id}"
            return self.riot_api.get_data_platform(endpoint)

        def by_summoner_id(self, summoner_id):
            endpoint = self.url + f"{summoner_id}"
            return self.riot_api.get_data_platform(endpoint)

        def by_name(self, name):
            endpoint = self.url + f"by-name/{name}"
            return self.riot_api.get_data_platform(endpoint)

        def by_puuid(self, puuid):
            endpoint = self.url + f"by-puuid/{puuid}"
            return self.riot_api.get_data_platform(endpoint)

    class Match:
        def __init__(self, riot_api):
            self.url = "/lol/match/v5/matches/"
            self.riot_api = riot_api

        def by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}"
            return self.riot_api.get_data_region(endpoint)

        def ids_by_puuid(self, puuid):
            endpoint = self.url + f"by-puuid/{puuid}/ids"
            return self.riot_api.get_data_region(endpoint)

        def timeline_by_match_id(self, match_id):
            endpoint = self.url + f"{match_id}/timeline"
            return self.riot_api.get_data_region(endpoint)


def rename_summoner_columns(summoner):
    summoner["account_id"] = summoner.pop("accountId")
    summoner["profile_icon_id"] = summoner.pop("profileIconId")
    summoner["revision_date"] = summoner.pop("revisionDate")
    summoner["summoner_level"] = summoner.pop("summonerLevel")


def add_summoner(summoner, player_name):
    players_id = pd.read_csv("./csv/players_id.csv")
    summoners = pd.read_csv("./csv/solo_rank/summoners.csv")

    player_id = players_id.loc[players_id["player"] == player_name, "player_id"].iloc[0]
    summoner["player_id"] = player_id
    rename_summoner_columns(summoner)

    df = summoners.loc[summoners["puuid"] == summoner["puuid"]]
    if df.shape[0] == 0:
        summoners = pd.concat([summoners, pd.DataFrame(data=[summoner])])
        summoners.to_csv("./csv/solo_rank/summoners.csv", index=False)
        print(f'Added new summoners ({player_name}, {summoner["name"]})')
    elif df["revision_date"].iloc[0] != summoner["revision_date"]:
        summoners.loc[summoners["puuid"] == summoner["puuid"]] = pd.DataFrame(
            data=[summoner]
        )
        summoners.to_csv("./csv/solo_rank/summoners.csv", index=False)
        print(f'Updated summoner ({player_name}, {summoner["name"]})')
    else:
        print(f'Already exist summoner ({player_name}, {summoner["name"]})')


def update_summoners():
    players_id = pd.read_csv("./csv/players_id.csv")
    summoners = pd.read_csv("./csv/solo_rank/summoners.csv")
    for puuid in summoners["puuid"]:
        row = summoners.loc[summoners["puuid"] == puuid]
        summoner = get_summoner_by_puuid(puuid)
        rename_summoner_columns(summoner)

        if row["revision_date"].iloc[0] != summoner["revision_date"]:
            for key, value in summoner.items():
                row[key] = value
            player_name = players_id.loc[
                players_id["player_id"] == row["player_id"].iloc[0], "player"
            ].iloc[0]
            print(f'Updated summoner ({player_name}, {row["name"]}')


def update_matches_id():
    summoners = pd.read_csv("./csv/solo_rank/summoners.csv")
    matches_id = pd.read_csv("./csv/solo_rank/matches_id.csv")

    for puuid in summoners["puuid"]:
        match_id_lst = get_matches_id_by_puuid(puuid)
        new_matches_id = []
        for match_id in match_id_lst:
            df = matches_id.loc[
                (matches_id["puuid"] == puuid) & (matches_id["match_id"] == match_id)
            ]
            if df.shape[0] == 0:
                new_matches_id.append(match_id)
        df = pd.DataFrame(
            data={"puuid": [puuid] * len(new_matches_id), "match_id": new_matches_id}
        )
        matches_id = pd.concat([matches_id, df], ignore_index=True)
        matches_id.to_csv("./csv/solo_rank/matches_id.csv", index=False)
        print(f"Updated {df.shape[0]} matches ({puuid})")


def match_analysis(match):
    match_id = match["metadata"]["matchId"]
    match_metadata = match["metadata"]
    match_info = match["info"]
    match_participants = match_info["participants"]


def handling_exception(target, content=""):
    if target.status_code != 200:
        raise Exception(content)

    return target.json()


def main():
    riot_api = RiotAPI()
    summoner = riot_api.summoner.by_name("hide on bush")
    summoner = handling_exception(summoner, "summoner")

    match_ids = riot_api.match.ids_by_puuid(summoner["puuid"])
    match_ids = handling_exception(match_ids, "match ids")

    match_id = match_ids[0]
    match = riot_api.match.by_match_id(match_id)
    match = handling_exception(match, "match")
    print(match)


if __name__ == "__main__":
    main()

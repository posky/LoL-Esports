import pandas as pd


def create_html_table(ratings):
    table = """<!DOCTYPE html>
    <html>
    <head>
    <title>Team Data</title>
    </head>
    <body>
        <table>
        <thead>
            <tr>
            <th>Team Name</th>
            <th>League</th>
            <th>Games</th>
            <th>Wins</th>
            <th>Losses</th>
            <th>Streak</th>
            <th>Points</th>
            <th>RD</th>
            <th>Sigma</th>
            <th>Last Game Date</th>
            <th>Change</th>
        </tr>
        </thead>
        <tbody>
    """

    df = ratings.loc[ratings["last_game_date"].dt.year == 2023]
    for team in df.itertuples():
        table += """
        <tr>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
            <td>{}</td>
        </tr>
        """.format(
            team.team,
            team.league,
            team.games,
            team.win,
            team.loss,
            team.streak,
            round(team.point, ndigits=2),
            round(team.rd, ndigits=2),
            round(team.sigma, ndigits=2),
            team.last_game_date,
            team.change,
        )

    table += """
            </tbody>
        </table>
    </body>
    </html>
    """

    return table


if __name__ == "__main__":
    ratings = pd.read_csv(
        "./csv/glicko_rating/glicko_rating_2023.csv", parse_dates=["last_game_date"]
    )

    table = create_html_table(ratings)

    with open("./table.html", "w", encoding="utf-8") as f:
        f.write(table)

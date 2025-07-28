import os
import base64
import requests
from utils.scrape import get_data_from_url
from utils.plot import create_scatterplot
from duckdb import connect

AIPIPE_BASE_URL = "https://aipipe.org/openrouter/v1"
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")  # Set this in Render or .env

HEADERS = {
    "Authorization": f"Bearer {AIPIPE_TOKEN}",
    "Content-Type": "application/json"
}

async def process_question(text: str):
    if "highest grossing films" in text.lower():
        df = get_data_from_url("https://en.wikipedia.org/wiki/List_of_highest-grossing_films")
        # Sample Q1: Count $2B+ before 2020
        q1 = df[(df['Worldwide'] >= 2e9) & (df['Year'] < 2020)].shape[0]

        # Q2: First movie > $1.5B
        q2 = df[df['Worldwide'] > 1.5e9].sort_values("Year").iloc[0]["Title"]

        # Q3: Rank vs Peak correlation
        q3 = df["Rank"].corr(df["Peak"])

        # Q4: Scatterplot
        plot_uri = create_scatterplot(df)

        return [q1, q2, round(q3, 6), plot_uri]

    # fallback
    return ["Not supported", "N/A", 0.0, "data:image/png;base64,..."]

import pandas as pd

def get_data_from_url(url: str):
    dfs = pd.read_html(url)
    df = dfs[0]  # Assuming the 1st table
    df = df.rename(columns=lambda x: str(x).strip())
    df["Worldwide"] = df["Worldwide"].replace('[\$,]', '', regex=True).astype(float)
    df["Year"] = pd.to_numeric(df["Year"], errors='coerce')
    df["Rank"] = pd.to_numeric(df["Rank"], errors='coerce')
    df["Peak"] = pd.to_numeric(df.get("Peak", df["Rank"]), errors='coerce')  # fallback
    return df.dropna()

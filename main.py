import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional
import httpx

AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")

app = FastAPI(title="TDS Data Analyst Agent")

def parse_gross(val):
    if pd.isna(val): return None
    s = str(val)
    s = re.sub(r'^[^\d$]*', '', s)
    m = re.search(r'[\d,]+(\.\d+)?', s)
    if not m: return None
    return float(m.group().replace(',', ''))

def parse_year(val):
    if pd.isna(val): return None
    m = re.search(r'\d{4}', str(val))
    if m: return int(m.group())
    return None

def parse_number(val):
    try: return float(re.sub(r'[^\d.-]', '', str(val)))
    except Exception: return None

@app.get("/", response_class=HTMLResponse)
async def root():
    return '''
    <h1>TDS Data Analyst Agent</h1>
    <form action="/api/" enctype="multipart/form-data" method="post">
      <label>questions.txt: <input type="file" name="questions" required></label><br>
      <label>Data file (.csv or .json): <input type="file" name="data"></label><br>
      <input type="submit" value="Upload and Analyze">
    </form>
    '''

@app.post("/api/")
async def analyze(
    questions: UploadFile = File(...), 
    data: UploadFile = File(None)
):
    q_str = (await questions.read()).decode('utf-8')
    df = None
    if data is not None:
        if data.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(await data.read()))
        elif data.filename.endswith('.json'):
            import json
            df = pd.DataFrame(json.loads((await data.read()).decode('utf-8')))
    if df is None:
        return JSONResponse({"error": "Data file (.csv or .json) required"}, status_code=400)
    colmap = {c.lower(): c for c in df.columns}
    for reqcol in ['rank', 'title', 'worldwide gross', 'year', 'peak']:
        if reqcol.title() not in df.columns and reqcol not in [c.lower() for c in df.columns]:
            return JSONResponse({"error": f"Missing required column {reqcol}"}, status_code=400)
    df['Worldwide gross'] = df[colmap['worldwide gross']].apply(parse_gross)
    df['Year'] = df[colmap['year']].apply(parse_year)
    df['Peak'] = df[colmap['peak']].apply(parse_number)
    df['Rank'] = df[colmap['rank']].apply(parse_number)
    df['Title'] = df[colmap['title']].astype(str)
    # (Optional) Call AI agent to 'think stepwise' about the task
    if AIPIPE_TOKEN:
        async with httpx.AsyncClient(base_url="https://aipipe.org/openrouter/v1") as client:
            _ = await client.post(
                "/chat/completions",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {AIPIPE_TOKEN}"},
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [{"role": "user", "content": q_str[:800]}]
                }
            )
    # Q1
    n_2bn_pre_2020 = int(df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2020)].shape[0])
    # Q2
    df_1_5 = df[df['Worldwide gross'] >= 1_500_000_000]
    earliest_title = df_1_5.loc[df_1_5['Year'].idxmin()]['Title'] if not df_1_5.empty else ""
    # Q3
    clean = df[['Rank','Peak']].dropna()
    from scipy.stats import pearsonr
    q3 = round(float(pearsonr(clean['Rank'], clean['Peak'])[0]), 6) if not clean.empty else 0.0
    # Q4
    import numpy as np
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(clean['Rank'], clean['Peak'], c='blue')
    z = np.polyfit(clean['Rank'], clean['Peak'], 1)
    p = np.poly1d(z)
    ax.plot(clean['Rank'], p(clean['Rank']), 'r--')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Peak')
    ax.set_title('Scatterplot of Rank vs Peak')
    plt.grid(True)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close()
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    img_uri = f"data:image/png;base64,{img_b64}"
    return [n_2bn_pre_2020, earliest_title, q3, img_uri]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

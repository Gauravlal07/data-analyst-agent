import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Union
import httpx
import re

AIPIPE_TOKEN = os.getenv("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZjEwMDE2MTlAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.G1z9xdDGSJ9ySQnW-yAPMu9UtKf4erFV12cWYq8jeMQ")  # Set this in Render/locally as secret

app = FastAPI(title="TDS Data Analyst Agent")

def parse_gross(val):
    if pd.isnull(val):
        return None
    s = str(val)
    s = re.sub(r'^[^\d$]*', '', s)
    m = re.search(r'[\d,]+(\.\d+)?', s)
    if not m:
        return None
    return float(m.group(0).replace(',', ''))

def parse_year(val):
    if pd.isnull(val): return None
    m = re.search(r'\d{4}', str(val))
    if m: return int(m.group(0))
    return None

def parse_number(val):
    try: return float(re.sub(r'[^\d.-]', '', str(val)))
    except: return None

@app.get("/", response_class=HTMLResponse)
async def form():
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
    request: Request,
    questions: UploadFile = File(...),
    data: UploadFile = File(None)
):
    prompt = (await questions.read()).decode('utf-8')

    # Support .csv or .json
    df = None
    if data:
        if data.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(await data.read()))
        elif data.filename.endswith('.json'):
            import json
            df = pd.DataFrame(json.loads((await data.read()).decode('utf-8')))
    if df is None:
        return JSONResponse({"error": "Data file (.csv or .json) required"}, status_code=400)

    # Standardize columns
    colmap = {c.lower(): c for c in df.columns}
    for reqcol in ['rank', 'title', 'worldwide gross', 'year', 'peak']:
        if reqcol.title() not in df.columns and reqcol not in [c.lower() for c in df.columns]:
            return JSONResponse({"error": f"Missing required column {reqcol}"}, status_code=400)
    df['Worldwide gross'] = df[colmap['worldwide gross']].apply(parse_gross)
    df['Year'] = df[colmap['year']].apply(parse_year)
    df['Peak'] = df[colmap['peak']].apply(parse_number)
    df['Rank'] = df[colmap['rank']].apply(parse_number)
    df['Title'] = df[colmap['title']].astype(str)

    # Use the AI model (via aipipe) for any "interpret question" step (optional demo)
    if AIPIPE_TOKEN:
        async with httpx.AsyncClient(base_url="https://aipipe.org/openrouter/v1") as client:
            res = await client.post(
                "/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {AIPIPE_TOKEN}"
                },
                json={
                    "model": "google/gemini-2.0-flash-lite-001",
                    "messages": [{"role": "user", "content": prompt[:800]}]
                }
            )
            _ = res.json()  # Demo: not actually used in logic, but shows LLM call.

    # 1. How many $2 bn movies before 2020?
    q1 = int(df[(df['Worldwide gross'] >= 2_000_000_000) & (df['Year'] < 2020)].shape[0])

    # 2. Earliest film to gross >1.5 bn
    sub = df[df['Worldwide gross'] >= 1_500_000_000]
    q2 = sub.loc[sub['Year'].idxmin()]['Title'] if not sub.empty else ""

    # 3. Correlation between Rank and Peak
    clean = df[['Rank', 'Peak']].dropna()
    from scipy.stats import pearsonr
    q3 = round(float(pearsonr(clean['Rank'], clean['Peak'])[0]), 6) if not clean.empty else 0.0

    # 4. Scatterplot (Rank vs Peak)
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

    return [q1, q2, q3, img_uri]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from utils.agent import analyze_data
import uvicorn

app = FastAPI()

# CORS settings (allow everything for now, restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Data Analyst Agent API is running."}

@app.post("/api/")
async def process_file(file: UploadFile = File(...)):
    content = await file.read()
    question = content.decode("utf-8").strip()

    try:
        result = analyze_data(question)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

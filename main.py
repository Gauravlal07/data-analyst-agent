from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google import genai
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task breakdown function using Gemini
def task_breakdown(task: str) -> str:
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        prompt_path = os.path.join('prompts', "abdul_task_breakdown.txt")
        if not os.path.exists(prompt_path):
            raise FileNotFoundError("Prompt file not found.")

        with open(prompt_path, 'r') as f:
            prompt_text = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=[task, prompt_text],
        )

        result = response.text if hasattr(response, "text") else str(response)
        with open("abdul_breaked_task.txt", "w") as f:
            f.write(result)

        return result
    except Exception as e:
        logger.error(f"Error in task_breakdown: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to the Data Analyst Agent API!"}

@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported.")

    try:
        content = await file.read()
        text = content.decode("utf-8")
        breakdown = task_breakdown(text)

        logger.info(f"Processed file: {file.filename}")
        return {
            "filename": file.filename,
            "original_content": text,
            "breakdown": breakdown
        }
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


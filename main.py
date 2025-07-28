from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
from utils.analyze import process_question

app = FastAPI()

@app.post("/api/")
async def handle_request(file: UploadFile = File(...)):
    content = await file.read()
    try:
        response = await process_question(content.decode())
        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, host="0.0.0.0")

import os
import traceback
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Union
from dotenv import load_dotenv
import json
import time

# --- Load environment variables from .env file in parent directory (backend/.env)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from .rag_pipeline import generate_material
from .export import export_text
from ollama_client import query_deepseek

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app = FastAPI(
    title="AI Material Generator",
    description="Generate educational materials (Question Papers/Worksheets/Lesson Plans) for Grades 1-12 using a RAG (Retrieval-Augmented Generation) pipeline by utilising Deepseek.",
    version="1.1.0"
)

# --- CORS Middleware for local frontend ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## --- DATA MODELS ---
class GenerateRequest(BaseModel):
    grade: str  # "Grade 1", ..., "Grade 12"
    chapter: Union[str, List[str]]  # Now supports both single string and list of chapters
    material_type: str  # "Question Paper" or "Worksheet" or "Lesson Plan"
    difficulty: str  # "Easy", "Medium", "Difficult"
    stream: Optional[str] = None # Only for Grades 11 and 12
    max_marks: Optional[int] = None  # Only required for Question Paper

class GenerateResponse(BaseModel):
    output: str

class ExportRequest(BaseModel):
    text: str
    filetype: str = "pdf"  # "pdf" or "docx"

class ExportResponse(BaseModel):
    file_path: str

class DeepseekRequest(BaseModel):
    materialType: Optional[str] = "worksheet"
    grade: Optional[str] = "X"
    chapter: Optional[str] = "General"
    difficulty: Optional[str] = "medium"

class DeepseekResponse(BaseModel):
    output: str

## --- ENDPOINTS ---

@app.get("/api/grades", response_model=List[str])
def get_grades():
    return [f"Grade {i}" for i in range(1, 13)]

@app.get("/api/material_types", response_model=List[str])
def get_material_types():
    return ["Question Paper", "Worksheet", "Lesson Plan"]

@app.get("/api/difficulty_levels", response_model=List[str])
def get_difficulty_levels():
    return ["Easy", "Medium", "Difficult"]

def build_prompt(data: DeepseekRequest) -> str:
    return (
        f"Generate a {data.materialType or 'worksheet'} for grade {data.grade or 'X'},"
        f' chapter "{data.chapter or "General"}", with {data.difficulty or "medium"} difficulty.'
    )

@app.post("/api/generate", response_model=GenerateResponse)
def generate(generate_req: GenerateRequest):
    # Validate max_marks for Question Paper
    if generate_req.material_type.strip().lower() == "question paper" and not generate_req.max_marks:
        raise HTTPException(status_code=400, detail="max_marks is required for Question Paper.")

    # --- Parse chapters as a list ---
    chapters = generate_req.chapter
    if isinstance(chapters, str):
        # Split on commas, strip whitespace, remove empty entries
        chapters_list = [c.strip() for c in chapters.split(",") if c.strip()]
    elif isinstance(chapters, list):
        chapters_list = [c.strip() for c in chapters if isinstance(c, str) and c.strip()]
    else:
        chapters_list = []

    # Make a copy of the request with chapters as a list
    updated_generate_req = generate_req.copy(update={"chapter": chapters_list})

    try:
        output = generate_material(updated_generate_req)
        return {"output": output}
    except Exception as excep:
        print("Error in /api/generate:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.post("/api/deepseek_generate", response_model=DeepseekResponse)
def deepseek_generate(deepseek_req: DeepseekRequest):
    """Endpoint migrated from Flask for Deepseek prompt-based generation."""
    try:
        prompt = build_prompt(deepseek_req)
        result = query_deepseek(prompt)
        return {"output": result}
    except Exception as excep:
        print("Error in /api/deepseek_generate:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.post("/api/export", response_model=ExportResponse)
def export(export_req: ExportRequest):
    try:
        file_path = export_text(export_req.text, export_req.filetype)
        return {"file_path": file_path}
    except Exception as excep:
        print("Error in /api/export:", excep)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(excep))

@app.get("/api/download")
def download_file(file_path: str):
    try:
        return FileResponse(
            path=file_path, 
            filename=os.path.basename(file_path), 
            media_type='application/octet-stream'
        )
    except Exception as excep:
        raise HTTPException(status_code=404, detail=f"File not found: {excep}")

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

# ----------- STREAMING PROGRESS ENDPOINT -----------

@app.get("/api/generate_stream")
async def generate_stream(
    grade: str = Query(..., description="Grade number, e.g. '10'"),
    chapter: str = Query(..., description="Comma-separated list of chapters"),
    material_type: str = Query(..., description="Material type (Question Paper, Worksheet, Lesson Plan)"),
    difficulty: str = Query(..., description="Difficulty (Easy, Medium, Difficult)"),
    stream: Optional[str] = Query(None, description="Stream for 11/12"),
    max_marks: Optional[int] = Query(None, description="Maximum marks for Question Paper")
):
    """
    Streams progress updates and the final output for the progress bar.
    On the frontend, use EventSource to listen to /api/generate_stream and update the progress bar accordingly.
    """
    async def event_generator():
        # Simulate progress: This is where you'd call your actual AI/generation and yield progress events
        total_steps = 8
        for i in range(total_steps):
            progress = int((i / (total_steps - 1)) * 90)  # up to 90%
            yield f"data: {json.dumps({'progress': progress})}\n\n"
            await asyncio.sleep(0.8)  # Simulate work

        # Now call your actual generate_material as the last step
        from types import SimpleNamespace
        req = SimpleNamespace(
            grade=grade,
            chapter=[c.strip() for c in chapter.split(",") if c.strip()],
            material_type=material_type,
            difficulty=difficulty,
            stream=stream,
            max_marks=max_marks
        )
        try:
            output = generate_material(req)
            yield f"data: {json.dumps({'progress': 100, 'output': output})}\n\n"
        except Exception as ex:
            yield f"data: {json.dumps({'error': str(ex)})}\n\n"

    import asyncio
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/")
def root():
    return {"message": "Backend is running!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

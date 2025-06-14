# AI Educational Material Generator

Generate educational materials (Question papers, Worksheets, Lesson plans) for Grades 1-8 using a RAG (Retrieval Augmented Generation) pipeline with Deepseek.

## Features

- **Generate**: Question Papers, Worksheets, or Lesson Plans for Grades 1-8
- **Customizable**: Choose Chapter, Difficulty (and Max marks only for Ques Papers)
- **Export**: Download generated materials as PDF or DOCX
- **Modern Stack**: FastAPI backend, VITE (React or similar) frontend
- **API integration**: Utilises Deepseek API for content generation

---

## Project Structure

```
LLM-QUESGENERATOR-INTERNSHIP-2025/
│
├── backend/
│   ├── app/
│   │   ├── __pycache__/  # Python bytecode cache 
│   │   ├── __init__.py # Marks 'app' as a Python package
│   │   ├── deepseek_infer.py # Handles Deepseek API calls
│   │   ├── export.py # Logic to export content as PDF/DOCX
│   │   ├── main.py # FastAPI app entry point 
│   │   ├── models.py # Pydantic models/schemas for the API
│   │   ├── pdf_ingest.py # Relevant scripts for ingesting/processing PDFs for RAG
│   │   ├── rag_pipeline.py # Core Retrieval Augmented Generation logic
│   │   ├── utils.py # Helper functions/utilities
│   │
│   ├── data/ # Knowledge base / storage for ingested or processed data
│   ├── vectorstores/ # Vector DB files/data (e.g., FAISS/Chroma Indexes, JSONs; not code, git-ignored)
│   ├── venv/ # Python virtual environment (ignored)
│   ├── .env # Backend environment variables (ignored)
│   ├── ollama_client.py # (if local model is used) Integration for Ollama models
│   ├── README.md # Backend-level documentation (optional)
│   ├── requirements.txt # Python dependencies to be installed
│   ├── routes.py # API route definitions
│
├── frontend/
│   ├── node_modules/ # Frontend dependencies (ignored)
│   ├── public/ # Static assests (favicon, etc.)
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── apis.ts # API call utilities for the frontend
│   │   ├── App.css # Main CSS for app
│   │   ├── App.tsx # Main React component
│   │   ├── index.css # Base CSS
│   │   ├── index.tsx # React entry point
│   │   ├── vite-env.d.ts # Vite TypeScript env definitions
│   ├── .env.example # Example environment config for frontend
│   ├── index.html # Frontend HTML entrypoint
│   ├── package-lock.json # NPM lockfile
│   ├── package.json # NPM dependencies/scripts
│   ├── README.md # Frontend-level documentation (optional)
│   ├── tsconfig.json # TypeScript config
│
├── scripts/
│   ├── example_prompts.md
│   ├── ingest_all_pdfs.py # Script to batch-ingest PDFs for RAG
    ├── start-all.sh # Shell script to launch both frontend and backend
│
├── .gitignore # Files/folders to ignore in git
├── ollama-setup.md # Setup instructions for Ollama (if using)
└── README.md # Project documentation (this file)
```

---

## Getting Started with Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate #This is for Windows; for Mac users: source venv/bin/activate
pip install -r requirements.txt
```

- Create a `.env` file in `backend/` and add:
  ```
  DEEPSEEK_API_KEY=your_deepseek_api_key (sk-xxxx...)
  FRONTEND_URL=http://localhost:5173
  ```
- Run the backend:
  ```bash
  uvicorn app.main:app --reload --log-level debug
  ```

### 3. Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## API Endpoints

- `GET /api/grades` - List grades (Grade 1-8)
- `GET /api/material_types` - List material types
- `GET /api/difficulty_levels` - List difficulty
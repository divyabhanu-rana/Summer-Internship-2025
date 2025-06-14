#!/bin/bash

## STARTING OLLAMA (if not already running)
ollama serve &

## STARTING THE BACKEND
cd ../backend
source venv/bin/activate
uvicorn app.main:app --reload --port 8000 &

## STARTING THE FRONTEND
cd ../frontend
npm run dev &
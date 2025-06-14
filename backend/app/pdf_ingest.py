import os
import json
from typing import List, Dict, Any
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstores")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def extract_text_by_page(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages]

def vectorize_chunks(chunks: List[str], model) -> List[List[float]]:
    return model.encode(chunks, convert_to_numpy=True).tolist()

def process_pdf(pdf_path: str, model) -> List[Dict[str, Any]]:
    file_name = os.path.basename(pdf_path)
    pages = extract_text_by_page(pdf_path)
    embeddings = vectorize_chunks(pages, model)
    records = []
    for i, (text, emb) in enumerate(zip(pages, embeddings)):
        if text.strip():
            records.append({
                "file_name": file_name,
                "page": i + 1,
                "text": text,
                "embedding": emb
            })
    return records

def save_vectorstore(records: List[Dict[str, Any]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def find_pdfs_recursively(root_dir: str) -> List[str]:
    pdf_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, f))
    return pdf_files

def main():
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)
    pdf_files = find_pdfs_recursively(DATA_DIR)
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        records = process_pdf(pdf_path, model)
        if not records:
            print(f"Skipped {pdf_path}: No extractable text.")
            continue
        # Create a relative path for output, e.g. evs/4/grade4_evs_part1_vectors.json
        rel_path = os.path.relpath(pdf_path, DATA_DIR)
        out_filename = rel_path.replace(".pdf", "_vectors.json")
        out_path = os.path.join(VECTORSTORE_DIR, out_filename)
        save_vectorstore(records, out_path)
        print(f"Processed {pdf_path} → {out_path} ({len(records)} pages embedded)")

if __name__ == "__main__":
    main()
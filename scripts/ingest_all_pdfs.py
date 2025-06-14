import os
import sys
from pathlib import Path
from typing import List

try:
     from pdfminer.high_level import extract_text
except ImportError:
     print("pdfminer.six is not installed. Please install it using 'pip install pdfminer.six'.")
     sys.exit(1)

def find_all_pdfs(root_dir: Path) -> List[Path]:
     """Recursively find all PDF files in the given root directory."""
     pdfs = []
     for dirpath, _, filenames in os.walk(root_dir):
          for filename in filenames:
               if filename.lower().endswith('.pdf'):
                    pdfs.append(Path(dirpath) / filename)
     return pdfs

def extract_text_from_pdf(pdf_path: Path) -> str:
     """Extract text from a PDF file."""
     try:
          text = extract_text(str(pdf_path))
          return text
     except Exception as excep:
          print(f"Error extracting text from {pdf_path}: {excep}")
          return ""
     
def main():
     if len(sys.argv) < 3:
          print("Usage: python ingest_all_pdfs.py <pdf_root_folder> <output_folder>")
          sys.exit(1)

     pdf_root = Path(sys.argv[1])
     output_root = Path(sys.argv[2])
     output_root.mkdir(parents=True, exist_ok=True)

     pdf_paths = find_all_pdfs(pdf_root)
     print(f"Found {len(pdf_paths)} PDF files in {pdf_root}.")

     for pdf_path in pdf_paths:
          rel_path = pdf_path.relative_to(pdf_root)
          txt_output_path = output_root / rel_path.with_suffix('.txt')
          txt_output_path.parent.mkdir(parents=True, exist_ok=True)

          if txt_output_path.exists():
               print(f"Skipping {txt_output_path} as it already exists.")
               continue

          print(f"Extracting {pdf_path}...")
          text = extract_text_from_pdf(pdf_path)
          if text.strip():
               with open(txt_output_path, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)
               print(f"Saved extracted text to {txt_output_path}.")
          else:
               print(f"No text extracted from {pdf_path}. Skipping.")

if __name__ == "__main__":
     main()
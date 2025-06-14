import os
import re
from typing import List, Optional

def clean_text(text: str) -> str:
     """
     Cleans text by removing extra whitespaces, unwanted characters and normalizing newlines.
     """

     ## REMOVING LEADING/TRAILING WHITESPACES IN EACH LINE
     text = "\n".join([line.strip() for line in text.splitlines()])

     text = re.sub(r'\n\s*\n', '\n\n', text)

     text = re.sub(r'[^\x09\x0A\x0D\x20 - \x7E]', '', text)

     return text.strip()

def ensure_dir_exists(directory: str) -> None:
     """
     Ensuring that the specific directory exists, creating it if needed.
     """
     os.makedirs(directory, exist_ok = True)

def split_text_into_chunks(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
     """
     Splits long text into chunks of a specified size with an optional overlap.
     Useful for LLM context window mgmt.
     """
     words = text.split()
     chunks = []
     i = 0
     while i < len(words):
          chunk = " ".join(words[i:i+chunk_size])
          chunks.append(chunk)
          i += chunk_size - overlap
     return chunks

def is_valid_filetype(filename: str, allowed_types: Optional[List[str]] = None) -> bool:
     """
     Checks if a filename has a valid file extension.
     """
     if allowed_types is None:
          allowed_types = ['pdf', 'docx', 'txt']
     ext = os.path.splitext(filename)[1].lower().lstrip('.')
     return ext in allowed_types

def get_file_extension(filename: str) -> str:
     """
     Returns the file extension (without dot) in lowercase.
     """
     return os.path.splitext(filename)[1].lower().lstrip('.')

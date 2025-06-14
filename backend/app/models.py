from pydantic import BaseModel, Field
from typing import List, Optional

class GenerateRequest(BaseModel):
     """
     Request model for generating educational materials (e.g., Question Papers, Worksheets, Lesson Plans).
     """
     grade: str = Field(..., description = "Grade level for the material (e.g., 'Grade 1', 'Grade 2', ..., 'Grade 11', 'Grade 12').")
     chapter: str = Field(..., description = "Chapter/topic name for which the material is to be generated.")
     material_type: str = Field(..., description = "Type of material to generate: 'Question Paper', 'Worksheets' and 'Lesson Plans'.")
     difficulty: str = Field(..., description = "Difficulty level of the material: 'Easy', 'Medium', or 'Difficult'.")

class GenerateResponse(BaseModel):
     """
     Response model for the generated educational material.
     """
     output: str = Field(..., description = "The generated content as a string.")

class ExportRequest(BaseModel):
     """
     Requesting the model for exporting generated content to a file.
     """
     text: str = Field(..., description = "The textual content to be exported.")
     filetype: str = Field("pdf", description = "File type to be exported to: 'pdf' or 'docx'.")

class ExportResponse(BaseModel):
     """
     Response model for the file export operation.
     """
     file_path: str = Field(..., description = "Path to the exported file.")

##
GRADE_OPTIONS = [f"Grade {i}" for i in range(1, 13)]
MATERIAL_TYPE_OPTIONS = ["Question Paper", "Worksheet", "Lesson Plan"]
DIFFICULTY_LEVEL_OPTIONS = ["Easy", "Medium", "Difficult"]
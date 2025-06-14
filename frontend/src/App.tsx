import React, { useState } from "react";
import GradeSelector from "./components/GradeSelector";
import ChapterInput from "./components/ChapterInput";
import MaterialTypeSelector from "./components/MaterialTypeSelector";
import DifficultyChooser from "./components/DifficultyChooser";
import OutputDisplay from "./components/OutputDisplay";
import DownloadButtons from "./components/DownloadButtons";

import "./App.css";

const App: React.FC = () => {
  // UI State
  const [grade, setGrade] = useState<string>("1");
  const [chapter, setChapter] = useState<string>("");
  const [materialType, setMaterialType] = useState<string>("Question paper");
  const [difficulty, setDifficulty] = useState<string>("easy");
  const [maxMarks, setMaxMarks] = useState<number | "">(""); // NEW: maxMarks state
  const [output, setOutput] = useState<string>("");
  const [generating, setGenerating] = useState<boolean>(false);

  // Download handlers (now implement actual export/download logic)
  const handleDownloadPDF = async () => {
    const res = await fetch("http://localhost:8000/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: output, filetype: "pdf" }),
    });
    if (!res.ok) {
      alert("Failed to export PDF!");
      return;
    }
    const { file_path } = await res.json();

    window.open(
      `http://localhost:8000/api/download?file_path=${encodeURIComponent(file_path)}`
    );
  };

  const handleDownloadWord = async () => {
    const res = await fetch("http://localhost:8000/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: output, filetype: "docx" }),
    });
    if (!res.ok) {
      alert("Failed to export Word file!");
      return;
    }
    const { file_path } = await res.json();

    window.open(
      `http://localhost:8000/api/download?file_path=${encodeURIComponent(file_path)}`
    );
  };

  // Generate output with LLM backend
  const handleGenerate = async () => {
    setGenerating(true);
    setOutput(""); // clear previous output
    try {
      // Prepare request body
      const requestBody: any = {
        grade,
        chapter,
        material_type: materialType,
        difficulty,
      };
      // Add max_marks only if Question Paper is selected
      if (materialType.trim().toLowerCase() === "question paper") {
        requestBody.max_marks = maxMarks;
      }
      const res = await fetch("http://localhost:8000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      if (!res.ok) {
        const errDetail = await res.json().catch(() => ({}));
        alert(
          errDetail?.detail ||
            "Failed to generate material!"
        );
        setGenerating(false);
        return;
      }
      const { output: generatedQuestions } = await res.json();
      setOutput(
        `Generated ${materialType} (Grade: ${grade}, Chapter: ${chapter}, Difficulty: ${difficulty}${
          materialType.trim().toLowerCase() === "question paper" && maxMarks
            ? `, Max Marks: ${maxMarks}`
            : ""
        })\n\n${generatedQuestions}`
      );
    } catch (err) {
      alert("An error occurred while generating the material.");
    } finally {
      setGenerating(false);
    }
  };

  // Helper: should maxMarks be required?
  const isQuestionPaper = materialType.trim().toLowerCase() === "question paper";
  const isLessonPlan = materialType.trim().toLowerCase() === "lesson plan";

  // Optional: Reset difficulty when switching to "Lesson plan"
  const handleMaterialTypeChange = (type: string) => {
    setMaterialType(type);
    if (type.trim().toLowerCase() === "lesson plan") {
      setDifficulty(""); // Reset or set to a default
    } else if (difficulty === "") {
      setDifficulty("easy"); // Or your desired default
    }
  };

  return (
    <div className="app-container">
      <h1 className="main-title">Question Generator</h1>
      <form
        className="input-form"
        onSubmit={(e) => {
          e.preventDefault();
          handleGenerate();
        }}
      >
        <div className="input-row">
          <GradeSelector value={grade} onChange={setGrade} />
          <ChapterInput value={chapter} onChange={setChapter} />
        </div>
        <div className="input-row">
          <MaterialTypeSelector value={materialType} onChange={handleMaterialTypeChange} />
          {!isLessonPlan && (
            <DifficultyChooser value={difficulty} onChange={setDifficulty} />
          )}
        </div>
        {isQuestionPaper && (
          <div className="input-row">
            <label>
              Maximum Marks:&nbsp;
              <input
                type="number"
                min={1}
                value={maxMarks}
                onChange={e => setMaxMarks(e.target.value === "" ? "" : Number(e.target.value))}
                required={isQuestionPaper}
                placeholder="Enter total marks"
                style={{ width: "110px" }}
              />
            </label>
          </div>
        )}
        <button
          className="generate-btn"
          type="submit"
          disabled={
            !chapter ||
            generating ||
            (isQuestionPaper && (!maxMarks || isNaN(Number(maxMarks)) || Number(maxMarks) < 1))
          }
        >
          {generating ? "Generating..." : "Generate"}
        </button>
      </form>

      {output && (
        <div className="output-section">
          <OutputDisplay title="Generated Output" content={output} />
          <DownloadButtons
            onDownloadPDF={handleDownloadPDF}
            onDownloadWord={handleDownloadWord}
          />
        </div>
      )}
    </div>
  );
};

export default App;
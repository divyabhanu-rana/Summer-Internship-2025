import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .deepseek_infer import ask_deepseek

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstores")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Explicit mapping of chapter names (normalized) to vectorstore files
# Update these mappings as per your actual chapters and files!
CHAPTER_FILE_MAP = {
    # Grade 1 EVS
    ("1", "myself(introduction)"): "evs/1/G1EVS-01_vectors.json",
    ("1", "my body"): "evs/1/G1EVS-01_vectors.json",
    ("1", "keeping clean"): "evs/1/G1EVS-02_vectors.json",
    ("1", "my family"): "evs/1/G1EVS-02_vectors.json",
    ("1", "food we eat"): "evs/1/G1EVS-03_vectors.json",
    ("1", "my home"): "evs/1/G1EVS-03_vectors.json",
    ("1", "water"): "evs/1/G1EVS-03_vectors.json",
    ("1", "clothes we wear"): "evs/1/G1EVS-03_vectors.json",
    ("1", "my neighbourhood"): "evs/1/G1EVS-04_vectors.json",
    ("1", "neighbourhood helpers"): "evs/1/G1EVS-04_vectors.json",
    ("1", "festivals"): "evs/1/G1EVS-04_vectors.json",
    ("1", "good habits and manners"): "evs/1/G1EVS-04_vectors.json",
    ("1", "means of transport & communication"): "evs/1/G1EVS-04_vectors.json",
    ("1", "plants around us"): "evs/1/G1EVS-05_vectors.json",
    ("1", "animals around us 1"): "evs/1/G1EVS-05_vectors.json",
    ("1", "earth and sky"): "evs/1/G1EVS-05_vectors.json",
    # Grade 3 EVS
    ("3", "family and relatives"): "evs/3/G3EVS-01_vectors.json",
    ("3", "our body"): "evs/3/G3EVS-01_vectors.json",
    ("3", "living and non-living things"): "evs/3/G3EVS-01_vectors.json",
    ("3", "plants: our green friends"): "evs/3/G3EVS-02_vectors.json",
    ("3", "animals around us 3"): "evs/3/G3EVS-02_vectors.json",
    ("3", "the beautiful birds"): "evs/3/G3EVS-02_vectors.json",
    ("3", "the food we eat"): "evs/3/G3EVS-02_vectors.json",
    ("3", "shelter: my sweet home"): "evs/3/G3EVS-03_vectors.json",
    ("3", "clothes: a basic need"): "evs/3/G3EVS-03_vectors.json",
    ("3", "means of transport"): "evs/3/G3EVS-03_vectors.json",
    ("3", "health is happiness"): "evs/3/G3EVS-04_vectors.json",
    ("3", "avoid accidents and stay safe"): "evs/3/G3EVS-04_vectors.json",
    ("3", "festivals and celebrations"): "evs/3/G3EVS-04_vectors.json",
    ("3", "air and water"): "evs/3/G3EVS-05_vectors.json",
    ("3", "the earth and other heavenly bodies"): "evs/3/G3EVS-05_vectors.json",
    ("3", "maps and sketches"): "evs/3/G3EVS-05_vectors.json",
    ("3", "the art of pottery"): "evs/3/G3EVS-06_vectors.json",
    ("3", "things around us"): "evs/3/G3EVS-06_vectors.json",
    # Grade 4 EVS
    ("4", "our family"): "evs/4/G4EVS-01_vectors.json",
    ("4", "know your tongue and teeth"): "evs/4/G4EVS-01_vectors.json",
    ("4", "animals around us 4"): "evs/4/G4EVS-02_vectors.json",
    ("4", "birds: beaks and claws"): "evs/4/G4EVS-02_vectors.json",
    ("4", "parts of a plant"): "evs/4/G4EVS-03_vectors.json",
    ("4", "food and health"): "evs/4/G4EVS-03_vectors.json",
    ("4", "from farm to our plate"): "evs/4/G4EVS-03_vectors.json",
    ("4", "means of recreation"): "evs/4/G4EVS-04_vectors.json",
    ("4", "travel and money"): "evs/4/G4EVS-04_vectors.json",
    ("4", "transport and communication"): "evs/4/G4EVS-04_vectors.json",
    ("4", "safety first"): "evs/4/G4EVS-05_vectors.json",
    ("4", "india: a land of rich culture"): "evs/4/G4EVS-05_vectors.json",
    ("4", "we need each-other"): "evs/4/G4EVS-05_vectors.json",
    ("4", "water for living"): "evs/4/G4EVS-06_vectors.json",
    ("4", "waste management"): "evs/4/G4EVS-06_vectors.json",
    ("4", "know about matter"): "evs/4/G4EVS-07_vectors.json",
    ("4", "measurement"): "evs/4/G4EVS-07_vectors.json",
    # Grade 5 EVS (example, update as needed)
    ("5", "the changing families"): "evs/5/G5EVS-01_vectors.json",
    ("5", "breathing in, breathing out"): "evs/5/G5EVS-01_vectors.json",
    ("5", "wellness: health and hygiene"): "evs/5/G5EVS-01_vectors.json",
    ("5", "super senses of animals"): "evs/5/G5EVS-02_vectors.json",
    ("5", "adaptation in plants"): "evs/5/G5EVS-02_vectors.json",
    ("5", "from taste to digestion"): "evs/5/G5EVS-02_vectors.json",
    ("5", "life in water"): "evs/5/G5EVS-03_vectors.json",
    ("5", "preservation of food"): "evs/5/G5EVS-03_vectors.json",
    ("5", "games we play"): "evs/5/G5EVS-04_vectors.json",
    ("5", "safety during calamities"): "evs/5/G5EVS-04_vectors.json",
    ("5", "save fuels"): "evs/5/G5EVS-05_vectors.json",
    ("5", "dignity of labour"): "evs/5/G5EVS-05_vectors.json",
    ("5", "our heritage buildings"): "evs/5/G5EVS-05_vectors.json",
    ("5", "every drop is precious"): "evs/5/G5EVS-05_vectors.json",
    ("5", "force: push or pull"): "evs/5/G5EVS-06_vectors.json",
    ("5", "simple machines"): "evs/5/G5EVS-06_vectors.json",
    ("5", "shelter for all"): "evs/5/G5EVS-06_vectors.json",
    ("5", "forest and tribal life"): "evs/5/G5EVS-06_vectors.json",
    ("5", "the spirit of adventure"): "evs/5/G5EVS-07_vectors.json",
    ("5", "adventure in space"): "evs/5/G5EVS-07_vectors.json",
    # Grade 9 English (example, update as needed)
    # --- BEEHIVE TEXTBOOK ---
    ("9", "beehive: the fun they had"): "Grade 9/English/iebe101_vectors.json",
    ("9", "beehive: the road not taken"): "Grade 9/English/iebe101_vectors.json",
    ("9", "beehive: the sound of music"): "Grade 9/English/iebe102_vectors.json",
    ("9", "beehive: wind"): "Grade 9/English/iebe102_vectors.json",
    ("9", "beehive: the little girl"): "Grade 9/English/iebe103_vectors.json",
    ("9", "beehive: rain on the roof"): "Grade 9/English/iebe103_vectors.json",
    ("9", "beehive: a truly beautiful mind"): "Grade 9/English/iebe104_vectors.json",
    ("9", "beehive: the lake isle of innisfree"): "Grade 9/English/iebe104_vectors.json",
    ("9", "beehive: the snake and the mirror"): "Grade 9/English/iebe105_vectors.json",
    ("9", "beehive: a legend of the northland"): "Grade 9/English/iebe105_vectors.json",
    ("9", "beehive: my childhood"): "Grade 9/English/iebe106_vectors.json",
    ("9", "beehive: no men are foreign"): "Grade 9/English/iebe106_vectors.json",
    ("9", "beehive: reach for the top"): "Grade 9/English/iebe107_vectors.json",
    ("9", "beehive: on killing a tree"): "Grade 9/English/iebe107_vectors.json",
    ("9", "beehive: kathmandu"): "Grade 9/English/iebe108_vectors.json",
    ("9", "beehive: a slumber did my spirit seal"): "Grade 9/English/iebh108_vectors.json",
    ("9", "beehive: if i were you"): "Grade 9/English/iebe109_vectors.json",

    # --- MOMENTS TEXTBOOK ---
    ("9", "moments: the lost child"): "Grade 9/English/iemo101_vectors.json",
    ("9", "moments: the adventures of toto"): "Grade 9/English/iemo102_vectors.json",
    ("9", "moments: iswaran the storyteller"): "Grade 9/English/iemo103_vectors.json",
    ("9", "moments: in the kingdom of fools"): "Grade 9/English/iemo104_vectors.json",
    ("9", "moments: the happy prince"): "Grade 9/English/iemo105_vectors.json",
    ("9", "moments: weathering the storm in ersama"): "Grade 9/English/iemo106_vectors.json",
    ("9", "moments: the last leaf"): "Grade 9/English/iemo107_vectors.json",
    ("9", "moments: a house is not a home"): "Grade 9/English/iemo108_vectors.json",
    ("9", "moments: the accidental tourist"): "Grade 9/English/iemo109_vectors.json",
    ("9", "moments: the beggar"): "Grade 9/English/iemo110_vectors.json",

    # Grade 10 English (example, update as needed)
    # --- FIRST FLIGHT TEXTBOOK ---
    ("10", "first flight: a letter to god"): "Grade 10/English/jeff101_vectors.json",
    ("10", "first flight: dust of snow"): "Grade 10/English/jeff101_vectors.json",
    ("10", "first flight: fire and ice"): "Grade 10/English/jeff101_vectors.json",
    ("10", "first flight: nelson mandela: long walk to freedom"): "Grade 10/English/jeff102_vectors.json",
    ("10", "first flight: a tiger in the zoo"): "Grade 10/English/jeff102_vectors.json",
    ("10", "first flight: two stories about flying"): "Grade 10/English/jeff103_vectors.json",
    ("10", "first flight: his first flight"): "Grade 10/English/jeff103_vectors.json",
    ("10", "first flight: black aeroplane"): "Grade 10/English/jeff103_vectors.json",
    ("10", "first flight: how to tell wild animals"): "Grade 10/English/jeff103_vectors.json",
    ("10", "first flight: the ball poem"): "Grade 10/English/jeff103_vectors.json",
    ("10", "first flight: from the diary of anne frank"): "Grade 10/English/jeff104_vectors.json",
    ("10", "first flight: amanda!"): "Grade 10/English/jeff104_vectors.json",
    ("10", "first flight: glimpses of india"): "Grade 10/English/jeff105_vectors.json",
    ("10", "first flight: a baker from goa"): "Grade 10/English/jeff105_vectors.json",
    ("10", "first flight: coorg"): "Grade 10/English/jeff105_vectors.json",
    ("10", "first flight: tea from assam"): "Grade 10/English/jeff105_vectors.json",
    ("10", "first flight: the trees"): "Grade 10/English/jeff105_vectors.json",
    ("10", "first flight: mijbil the otter"): "Grade 10/English/jeff106_vectors.json",
    ("10", "first flight: fog"): "Grade 10/English/jeff106_vectors.json",
    ("10", "first flight: madam rides the bus"): "Grade 10/English/jeff107_vectors.json",
    ("10", "first flight: the tale of custard the dragon"): "Grade 10/English/jeff107_vectors.json",
    ("10", "first flight: the sermon at benares"): "Grade 10/English/jeff108_vectors.json",
    ("10", "first flight: for anne gregory"): "Grade 10/English/jeff108_vectors.json",
    ("10", "first flight: the proposal"): "Grade 10/English/jeff109_vectors.json",

    # --- FOOTPRINTS WITHOUT FEET TEXTBOOK ---
    ("10", "footprints: a triumph of surgery"): "Grade 10/English/jefp101_vectors.json",
    ("10", "footprints: the thief's story"): "Grade 10/English/jefp102_vectors.json",
    ("10", "footprints: the midnight visitor"): "Grade 10/English/jefp103_vectors.json",
    ("10", "footprints: a question of trust"): "Grade 10/English/jefp104_vectors.json",
    ("10", "footprints: footprints without feet"): "Grade 10/English/jefp105_vectors.json",
    ("10", "footprints: the making of a scientist"): "Grade 10/English/jefp106_vectors.json",
    ("10", "footprints: the necklace"): "Grade 10/English/jefp107_vectors.json",
    ("10", "footprints: the hack driver"): "Grade 10/English/jefp108_vectors.json",
    ("10", "footprints: bholi"): "Grade 10/English/jefp109_vectors.json",
    ("10", "footprints: the book that saved the earth"): "Grade 10/English/jefp110_vectors.json",
    # Add more mappings as needed for other grades/subjects
}

def normalize_chapter(text):
    # Normalize for mapping AND UI input
    return (
        text.strip().lower()
        .replace("’", "'").replace("‘", "'")
        .replace("–", "-").replace("—", "-")
        .replace(".", "").replace("  ", " ")
        .replace("&", "and")  # Optional: unify ampersand
    )

def ensure_chapter_list(chapter):
    if isinstance(chapter, str):
        return [c.strip() for c in chapter.split(",") if c.strip()]
    elif isinstance(chapter, list):
        chapters = []
        for item in chapter:
            if isinstance(item, str):
                chapters.extend([c.strip() for c in item.split(",") if c.strip()])
        return chapters
    else:
        return [str(chapter).strip()] if chapter else []

def get_vectorstore_filename(grade: str, chapter: str) -> str:
    grade_num = ''.join(filter(str.isdigit, str(grade)))
    chapter_key = normalize_chapter(chapter)
    key = (grade_num, chapter_key)
    if key in CHAPTER_FILE_MAP:
        print(f"Exact mapping found for: {key} → {CHAPTER_FILE_MAP[key]}")
        return CHAPTER_FILE_MAP[key]
    # Try partial match (optional)
    for (g, ch), filename in CHAPTER_FILE_MAP.items():
        if g == grade_num and chapter_key in ch:
            print(f"Partial mapping found for: {chapter_key} in {ch} → {filename}")
            return filename
    print(f"ERROR: No mapping found for grade '{grade_num}' and chapter '{chapter_key}'")
    raise ValueError(f"Cannot match chapter name to any vectorstore file: {chapter} (normalized: {chapter_key})")

def generate_material(request):
    print("\n--- Starting material generation ---")
    grade = request.grade        # e.g. "Grade 4" or "4"
    chapters = ensure_chapter_list(getattr(request, "chapter", ""))
    material_type = request.material_type
    difficulty = request.difficulty
    max_marks = getattr(request, "max_marks", None)

    print(f"Grade: {grade}")
    print(f"Raw chapters: {chapters}")
    print(f"Material type: {material_type}, Difficulty: {difficulty}, Max marks: {max_marks}")

    # Gather vectorstore files for all chapters
    vectorstore_files = []
    for chapter in chapters:
        print(f"Processing chapter: '{chapter}' (normalized: '{normalize_chapter(chapter)}')")
        try:
            vf = get_vectorstore_filename(grade, chapter)
            vectorstore_files.append(vf)
        except Exception as e:
            print(f"ERROR during mapping: {e}")
            raise

    print(f"Vectorstore files to use: {vectorstore_files}")

    # Collect vectors from all relevant files
    vectors = []
    for i, vectorstore_file in enumerate(vectorstore_files):
        vectorstore_path = os.path.join(VECTORSTORE_DIR, vectorstore_file)
        print(f"Loading vectorstore file for chapter '{chapters[i]}': {vectorstore_path}")
        if not os.path.exists(vectorstore_path):
            print(f"ERROR: File does not exist: {vectorstore_path}")
            raise FileNotFoundError(f"Vectorstore file not found for {grade}, {chapters[i]}: {vectorstore_file}")
        try:
            with open(vectorstore_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                print(f"Loaded {len(data)} records from {vectorstore_path}")
                vectors.extend(data)
        except Exception as e:
            print(f"ERROR reading/parsing JSON from {vectorstore_path}: {e}")
            raise

    print(f"Total vectors loaded: {len(vectors)} from chapters: {chapters}")

    if not vectors:
        print("ERROR: No vectors found. Vectorstore files may be empty or corrupted.")
        raise ValueError("No vector data available for the requested chapters.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Loaded embedding model.")

    user_query = (
        f"Create a {material_type.lower()} for {grade}, Chapters: '{', '.join(chapters)}', with {difficulty.lower()} difficulty."
    )
    query_vec = model.encode([user_query])[0]
    print("Encoded query:", user_query)

    def cosine_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    scored_chunks = [
        (cosine_sim(query_vec, entry["embedding"]), entry["text"])
        for entry in vectors if "embedding" in entry and "text" in entry
    ]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    top_chunks = [text for _, text in scored_chunks[:3]]
    print("Top chunks selected for context:")
    for idx, chunk in enumerate(top_chunks):
        print(f"  [{idx+1}] {chunk[:120]}...")  # Print first 120 chars

    # ---- CONTEXT-AWARE, ANTI-HALLUCINATION PROMPT ----
    cbse10_pattern = """
For Class 10 Question Papers, strictly follow this structure for the entire paper:

Section A: Multiple Choice Questions (MCQs): Questions 1-18, 1 mark each, no internal choice.
Section A: Assertion-Reason Questions: Questions 19-20, 1 mark each, no internal choice.
Section B: Very Short Answer (VSA) Questions: Questions 21-25, 2 marks each, 2 questions have internal choice.
Section C: Short Answer (SA) Questions: Questions 26-31, 3 marks each, 2 questions have internal choice.
Section D: Long Answer (LA) Questions: Questions 32-35, 5 marks each, 2 questions have internal choice.
Section E: Case Study-Based Questions: Questions 36-38, 4 marks each, all have internal choice.

Show section labels, marks per section, question numbers, and clearly specify internal choice as per the above structure. The sum of marks must match the total.
""".strip()

    prompt = (
        f"You are an expert educator. "
        f"Based ONLY on the following material provided from the backend/data/ directory of the project, "
        f"which is the {grade} textbook, Chapters: '{', '.join(chapters)}', "
        f"generate a {material_type.lower()} suitable for students. "
        f"Distribute the questions and content across ALL the listed chapters, ensuring each chapter is represented in the final output. "
        +
        (
            f"For question papers, the total maximum marks is {max_marks}. "
            if material_type.strip().lower() == "question paper" and max_marks else ""
        )
        +
        (
            "\n" + cbse10_pattern + "\n"
            if material_type.strip().lower() == "question paper" and (grade == "10" or grade == "Grade 10")
            else
            (
                f"Divide the paper into sections as follows: "
                f"SECTION-A should be worth 10% of total marks ({int(0.1*max_marks)} marks), "
                f"SECTION-B should be 50% of total marks ({int(0.5*max_marks)} marks), "
                f"SECTION-C should be 40% of total marks ({int(0.4*max_marks)} marks). "
                "Distribute the questions and marks accordingly. Clearly mention the marks for each section and each question. "
                if material_type.strip().lower() == "question paper" and max_marks else ""
            )
        )
        +
        (
            "For lesson plans, provide a clear sequence of teaching objectives, key points, teaching steps, activities, and assessment, but use ONLY content from the provided context. "
            "The lesson plan MUST align with the principles and guidelines of the Government of India's NEW EDUCATION POLICY (NEP), 2020, explicitly mentioning inclusive education. "
            "All activities should be designed to be accessible and supportive for students with disabilities, focusing on universal design for learning, differentiated instruction, and reasonable accommodations. "
            "Highlight how each activity addresses diverse learning needs and promotes participation by students with disabilities. "
            if material_type.strip().lower() == "lesson plan" else ""
        )
        +
        f"The questions, activities, or plan should be at a {difficulty.lower()} difficulty level, "
        f"and must be strictly derived ONLY from the provided context. "
        f"Do NOT use your own knowledge or add facts that are not in the context. "
        f"Do not hallucinate or invent information. "
        f"If the context does not provide enough material, only use what is available and do not make up content.\n\n"
        "IMPORTANT: Do NOT skip, summarize, or combine questions. Write out every question in full. Placeholders, continuations, or summaries (such as 'questions 11-18 continue similarly...' or 'remaining questions follow the same pattern') are strictly NOT allowed.\n"
        "For Class 10 and Class 12: It is absolutely critical to provide EVERY required question in FULL, without omission or summarization, as these are board-level papers. No skipping, summarizing, or use of placeholders is permitted under any circumstances. Every question must be explicitly written out.\n\n"
        f"---\n"
        f"Context:\n"
        f"{chr(10).join(top_chunks)}\n"
        f"---\n"
        f"Instructions:\n"
        +
        (
            "- If material type is lesson plan, structure as: Objectives, Key Points, Teaching Steps, Activities, Assessment, based ONLY on context.\n"
            if material_type.strip().lower() == "lesson plan" else ""
        )
        +
        (
            "- If material type is question paper, ensure each section and each question has marks shown, and the total matches the given maximum marks.\n"
            if material_type.strip().lower() == "question paper" and max_marks else ""
        )
        +
        f"- Generate only the {material_type.lower()}, not answers.\n"
        f"- Cover all main concepts found in the context, and ensure questions/content are distributed across ALL listed chapters.\n"
        f"- Ensure alignment with {difficulty.lower()} level.\n"
        f"- Number the questions or activities clearly.\n"
        f"- Do not repeat instructions or context in output.\n"
        f"- Do not hallucinate or use any information outside the provided context.\n"
        f"- Do not use any markdown syntax (e.g., *, **, ---, etc.); output must be in plain text only.\n"
    )

    print("\nPrompt for Deepseek:\n")
    print(prompt[:1200] + ("..." if len(prompt) > 1200 else ""))  # Print first 1200 chars

    print("\nSending to Deepseek...")
    response = ask_deepseek(prompt)
    print("Deepseek returned.")
    return response

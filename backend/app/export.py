import tempfile
import os
import subprocess
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_to_docx(text: str, filename: str) -> str:
    """
    Exports the given text to a word (.docx) file (plain text, for non-math subjects).
    Returns the path to the saved file.
    """
    doc = Document()
    for para in text.split('\n'):
        doc.add_paragraph(para)
    doc.save(filename)
    return filename

def export_to_pdf(text: str, filename: str) -> str:
    """
    Exports the given text to a PDF file (plain text, for non-math subjects).
    Returns the path to the saved file.
    """
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    margin = 40
    x = margin
    y = height - margin
    line_height = 14

    # SIMPLE WORD WRAPPING AND NEWLINES
    for para in text.split('\n\n'):
        for line in para.split('\n'):
            lines = []
            while len(line) > 0:
                if c.stringWidth(line) < (width - 2 * margin):
                    lines.append(line)
                    line = ""
                else:
                    # FINDING THE APPROXIMATE WRAPPING POSITION
                    wrap_pos = min(len(line), int((width - 2 * margin) / 7))
                    # ATTEMPT TO WRAP AT THE LAST SPACE BEFORE wrap_pos
                    space_pos = line.rfind(' ', 0, wrap_pos)
                    if space_pos == -1:
                        space_pos = wrap_pos
                    lines.append(line[:space_pos])
                    line = line[space_pos:].lstrip()

            for l in lines:
                if y < margin + line_height:
                    c.showPage()
                    y = height - margin
                c.drawString(x, y, l)  
                y -= line_height
        y -= line_height  # EXTRA SPACING BETWEEN PARAGRAPHS

    c.save()
    return filename

def export_with_pandoc(text: str, filename: str, filetype: str) -> str:
    """
    Uses Pandoc to export Markdown (with LaTeX math) to PDF or DOCX (for math/science subjects).
    Returns the path to the saved file.
    """
    # Write the text to a temporary markdown file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode="w", encoding="utf-8") as temp_md:
        # If the text does not use $...$ or $$...$$ for math, but uses \(...\) or \[...\], convert:
        # (Uncomment below if you want auto-conversion, otherwise expect $...$ input)
        # import re
        # text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
        # text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
        temp_md.write(text)
        temp_md.flush()
        md_path = temp_md.name

    # Determine output extension and Pandoc format
    if filetype == "pdf":
        out_ext = ".pdf"
        pandoc_args = ["pandoc", md_path, "-o", filename, "--pdf-engine=xelatex"]
    elif filetype in ("docx", "word"):
        out_ext = ".docx"
        pandoc_args = ["pandoc", md_path, "-o", filename]
    else:
        os.unlink(md_path)
        raise ValueError(f"Unsupported file type: choose 'pdf' or 'docx' (or 'word'). Got: {filetype}")

    # Make sure filename has the correct extension
    if not filename.endswith(out_ext):
        filename += out_ext

    # Run Pandoc
    try:
        subprocess.run(pandoc_args, check=True)
    except subprocess.CalledProcessError as e:
        os.unlink(md_path)
        raise RuntimeError(f"Pandoc failed: {e}")

    # Clean up the temporary markdown file
    os.unlink(md_path)
    return filename

def export_text(text: str, filetype: str = "pdf", use_pandoc: bool = False) -> str:
    """
    Exports text to either a PDF or Word File.
    - If use_pandoc is True, uses Pandoc (for math subjects).
    - If use_pandoc is False, uses plain export (for non-math subjects).
    Returns the path to the saved file.
    """
    # -- AUTO-CONVERT \(...\) to $...$ and \[...\] to $$...$$ for Pandoc if needed --
    # For best results, your frontend should send math as $...$ and $$...$$ for block/inline math.
    # If you want auto-conversion, uncomment below:
    # import re
    # if use_pandoc:
    #     text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    #     text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)

    if use_pandoc:
        suffix = ".pdf" if filetype == "pdf" else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            filename = temp_file.name
        return export_with_pandoc(text, filename, filetype)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{filetype}') as temp_file:
            filename = temp_file.name
        if filetype == "pdf":
            return export_to_pdf(text, filename)
        elif filetype == "docx" or filetype == "word":
            return export_to_docx(text, filename)
        else:
            os.unlink(filename)  # CLEANING UP THE TEMP FILE OF UNSUPPORTED TYPE
            raise ValueError(f"Unsupported file type: choose 'pdf' or 'docx' (or 'word'). Got: {filetype}")
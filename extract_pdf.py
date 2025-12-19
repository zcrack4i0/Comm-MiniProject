import sys

try:
    from pypdf import PdfReader
except ImportError:
    try:
        import PyPDF2
        PdfReader = PyPDF2.PdfReader
    except ImportError:
        print("Please install pypdf or PyPDF2: pip install pypdf")
        sys.exit(1)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
            return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

if __name__ == "__main__":
    import sys
    # Set UTF-8 encoding for stdout
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    pdf_path = "Mini-Project F25.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    # Save to a text file for easier reading
    with open("project_requirements.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("Text extracted and saved to project_requirements.txt")
    print(f"Extracted {len(text)} characters from PDF")


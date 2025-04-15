from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf4llm
import fitz
import pdfplumber

def extract_text_from_pdf(pdf_file):
    pdf_content = pdf_file.read()
    # doc = fitz.open(stream=pdf_content)
    # # text = ""
    # # for page_num in range(doc.page_count):
    # #     page = doc.load_page(page_num)  
    # #     text += page.get_text()
    # text = pymupdf4llm.to_markdown(doc)
    # return text
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text



def split_into_chunks(text, chunk_size: int = 10000, chunk_overlap: int = 500):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.create_documents([text])
    # chunk_texts = [chunk.page_content for chunk in chunks]
    return chunks

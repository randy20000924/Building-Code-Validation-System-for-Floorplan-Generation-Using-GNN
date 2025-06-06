import fitz  # PyMuPDF
import pdfplumber
import pandas as pd
import torch
import numpy as np
import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Extract text from PDF while keeping page numbers
def extract_text_from_pdf(pdf_path):
    text_by_page = []
    doc = fitz.open(pdf_path)

    for i in tqdm(range(len(doc)), desc="Extracting text from PDF", unit="page"):
        try:
            text = doc[i].get_text("text")  # Extract plain text
            if text and len(text.strip()) > 10:  # Filter out very short texts
                text_by_page.append((i, text.strip()))  # (Page number, Content)
        except Exception as e:
            print(f"Error: Failed to extract text from page {i+1}, Error message: {e}")
            continue
    return text_by_page

# Extract tables from PDF
def extract_tables_from_pdf(pdf_path):
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i in tqdm(range(len(pdf.pages)), desc="Extracting tables from PDF", unit="page"):
            page = pdf.pages[i]
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    df = pd.DataFrame(table)  # Convert to DataFrame
                    # Check if it contains relevant keywords
                    if df.astype(str).apply(lambda x: x.str.contains("bedroom|kitchen|bathroom|room|dwelling|floor area", case=False, na=False)).any().any():
                        table_pages.append(i)
    return sorted(set(table_pages))

# Define keywords for retrieval (BM25 exact match + NLP semantic search)
KEYWORDS = ["bathroom", "kitchen", "bedroom", "room"]

# Initialize BM25 for keyword-based retrieval
def build_bm25_index(paragraphs):
    tokenized_corpus = [p.split() for p in paragraphs]  # Tokenization
    return BM25Okapi(tokenized_corpus)

# Initialize NLP retrieval model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1", device=device)

# Convert paragraphs into embeddings (limit length to avoid truncation)
def encode_paragraphs(paragraphs):
    embeddings = []
    for i in tqdm(range(0, len(paragraphs), 16), desc="Computing embeddings", unit="batch"):
        batch = [p[:512] for p in paragraphs[i:i+16]]  # Limit max length to 512
        batch_embeddings = model.encode(batch, convert_to_numpy=True, device=device)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)  # Merge all embeddings

# BM25 keyword-based retrieval
def search_relevant_pages_bm25(bm25, query, text_by_page, top_k=10):
    scores = bm25.get_scores(query.split())  # Compute BM25 scores
    top_indices = np.argsort(scores)[-top_k:][::-1]  # Get top-K relevant pages
    return sorted(set(text_by_page[i][0] for i in top_indices if scores[i] > 0))

# NLP semantic retrieval
def search_relevant_pages_nlp(query, text_by_page, embeddings, top_k=10):
    query_embedding = model.encode([query], convert_to_numpy=True, device=device)
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top-K relevant pages
    return sorted(set(text_by_page[i][0] for i in top_indices))

# Main execution
pdf_path = "data/bc_building_code.pdf"
text_by_page = extract_text_from_pdf(pdf_path)
if not text_by_page:
    print("Failed to extract PDF content. Please check the document.")
    exit()

paragraphs = [t[1] for t in text_by_page]  # Extract only text
bm25 = build_bm25_index(paragraphs)  # Build BM25 index
embeddings = encode_paragraphs(paragraphs)  # Compute embeddings

# Find relevant pages
all_relevant_pages = set()

# Extract tables from PDF and find pages containing 'bedroom, kitchen, bathroom'
table_pages = extract_tables_from_pdf(pdf_path)
all_relevant_pages.update(table_pages)

# Perform BM25 + NLP dual retrieval
for query in tqdm(KEYWORDS, desc="Searching for relevant keywords", unit="keyword"):
    bm25_pages = search_relevant_pages_bm25(bm25, query, text_by_page, top_k=10)
    nlp_pages = search_relevant_pages_nlp(query, text_by_page, embeddings, top_k=10)
    all_relevant_pages.update(bm25_pages + nlp_pages)

if all_relevant_pages:
    # Extract relevant pages from original PDF
    doc = fitz.open(pdf_path)
    new_pdf = fitz.open()

    for page_num in tqdm(sorted(all_relevant_pages), desc="Saving selected pages", unit="page"):
        new_pdf.insert_pdf(doc, from_page=page_num, to_page=page_num)

    # Generate timestamp for filename
    output_pdf_path = f"data/bc_building_code_filtered.pdf"

    # Save the new PDF
    new_pdf.save(output_pdf_path)
    new_pdf.close()

    print(f"Completed. Relevant pages have been saved to {output_pdf_path}")
else:
    print("No relevant pages were found.")

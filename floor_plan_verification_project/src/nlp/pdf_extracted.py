import pdfplumber
import pandas as pd
import json
from tqdm import tqdm

# Enhanced PDF extraction to support both text and tables
def extract_pdf_content(pdf_path):
    text_content = ""
    table_content = []

    print("Starting PDF text and table extraction...")

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(tqdm(pdf.pages, desc="Extracting PDF pages", unit="page")):
            # Extract plain text
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"

            # Extract tables
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                df.dropna(how='all', inplace=True)  # Remove rows where all values are NaN
                table_content.append(df.to_dict(orient="records"))  # Convert DataFrame to dictionary format

    print("PDF text and table extraction completed.")
    
    return text_content, table_content

# Save extracted content to JSON
def save_to_json(text, tables, output_path):
    data = {
        "text": text,
        "tables": tables
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    pdf_path = "data/bc_building_code_filtered.pdf"
    output_json_path = "data/bc_building_code_extracted.json"

    print("Processing PDF...")
    building_code_text, table_content = extract_pdf_content(pdf_path)
    save_to_json(building_code_text, table_content, output_json_path)

    print(f"PDF content saved to {output_json_path}")

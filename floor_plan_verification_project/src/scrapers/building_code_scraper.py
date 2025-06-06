# building_code_scraper.py
import os
import requests
from bs4 import BeautifulSoup

# Define the target URL for the BC Building Code
URL = "https://www2.gov.bc.ca/gov/content/industry/construction-industry/building-codes-standards/bc-codes/2024-bc-codes"

# Define storage paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
PARENT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))  
DATA_DIR = os.path.join(PARENT_DIR, "data")  
PDF_FILE = os.path.join(DATA_DIR, "BC_Building_Code.pdf")  

# Access the content of the URL
HEADERS = {"User-Agent": "Mozilla/5.0"}
response = requests.get(URL, headers=HEADERS)

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find all <a> tags with href attribute
    for link in soup.find_all("a", href=True):
        href = link["href"]

        # Find the href of the PDF file
        if "_web_version" in href and href.endswith(".pdf"):
            pdf_url = href if href.startswith("http") else "https://www2.gov.bc.ca" + href
            print(f"Download link: {pdf_url}")
            
            # Download PDF
            pdf_response = requests.get(pdf_url, headers=HEADERS, stream=True)
            
            with open(PDF_FILE, "wb") as pdf_file:
                for chunk in pdf_response.iter_content(chunk_size=1024):
                    if chunk:
                        pdf_file.write(chunk)
            
            print(f"✅ PDF downloaded: {PDF_FILE}")
            break
else:
    print(f"❌ Error code: {response.status_code}")
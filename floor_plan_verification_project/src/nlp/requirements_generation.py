import json
import ollama
import re
import os

def extract_room_size_requirements(building_code_data, output_path):
    print("Extracting room size requirements...")

    text_content = building_code_data["text"]
    table_content = "\n\n".join([
        json.dumps(table, indent=4, ensure_ascii=False)
        for table in building_code_data["tables"]
    ])

    prompt = f"""
    You are an expert in building codes. Extract the minimum area required for different rooms 
    (bedroom, kitchen, bathroom) from the following British Columbia Building Code text and tables:

    **Text Content:**
    {text_content}

    **Table Content:**
    {table_content}

    We only need information about bedrooms, kitchens and bathrooms. Please give me a single floating-point number. Do not provide a range, explanation, or multiple values. Just one specific float, like 3.14.
    Provide the results in the following structured format:
    - Bedroom: X m²
    - Kitchen: Y m²
    - Bathroom: Z m²

    """

    model = "llama3:8b"
    response = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}])

    result_text = response["message"]["content"]
    print("LLM Response:\n", result_text)

    matches = re.findall(r'(\b\w+):\s*([\d.]+)', result_text)

    # 將結果轉為 dict
    room_areas = {room: float(area) for room, area in matches}

    with open(output_path, 'w') as f:
        json.dump(room_areas, f, indent=2)



if __name__ == "__main__":
    json_path = "data/bc_building_code_extracted.json"
    output_path = "data/room_min_sizes.json"

    print("Loading extracted building code data...")
    with open(json_path, "r", encoding="utf-8") as f:
        building_code_data = json.load(f)

    extract_room_size_requirements(building_code_data, output_path)

import json
import ollama

def extract_room_size_requirements(building_code_data):
    print("Extracting room size requirements...")

    text_content = building_code_data["text"]
    table_content = "\n\n".join([json.dumps(table, indent=4, ensure_ascii=False) for table in building_code_data["tables"]])

    prompt = f"""
    You are an expert in building codes. Extract the minimum required sizes for different rooms 
    (bedroom, kitchen, bathroom) from the following British Columbia Building Code text and tables:

    **Text Content:**
    {text_content}

    **Table Content:**
    {table_content}

    Provide the results in the following structured format:
    - Bedroom: X m²
    - Kitchen: Y m²
    - Bathroom: Z m²
    """

    model = "llama3"
    response = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}])
    
    print("Room size extraction completed.")
    return response["message"]["content"]

def validate_floorplan(room_requirements, annotaion_path):
    print("Validating Floor Plan compliance with building codes...")

    prompt = f"""
    You are a professional architect and building code expert.
    Your task is to check whether the provided 2D Floor Plan complies with the British Columbia Building Code.

    **Extracted Building Code Room Size Requirements:**
    {room_requirements}

    **Floor Plan OCR Extracted JSON:**
    {annotaion_path}

    Compare the room sizes in the floor plan with the required sizes.
    If any room is smaller than the required size, suggest possible modifications.
    """

    model = "llama3"
    response = ollama.chat(model=model, messages=[{"role": "system", "content": prompt}])
    
    print("Floor Plan validation completed.")
    return response["message"]["content"]

if __name__ == "__main__":
    json_path = "data/bc_building_code_extracted.json"
    annotaion_path = "data/2D_floorplan_extracted.json"

    print("Loading extracted building code data...")
    with open(json_path, "r", encoding="utf-8") as f:
        building_code_data = json.load(f)

    room_size_requirements = extract_room_size_requirements(building_code_data)
    print("Extracted room size requirements:\n", room_size_requirements)

    validation_result = validate_floorplan(room_size_requirements, annotaion_path)
    print("Floor Plan Validation Result:\n", validation_result)

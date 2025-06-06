import os
import json
from tqdm import tqdm

# human_annotated_tags folder
input_dir = "data/human_annotated_tags"
output_json_path = "data/human_annotated_tags.json"

def convert_txt_to_json(input_dir, output_json_path):
    annotated_data = {}

    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]

    print(f"Found {len(txt_files)} txt files. Processing...")

    for filename in tqdm(txt_files, desc="Converting txt to JSON", unit="file"):
        file_path = os.path.join(input_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            annotated_data[filename] = f.read().strip()

    # save to JSON
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(annotated_data, f, indent=4, ensure_ascii=False)

    print(f"Conversion completed. JSON saved to {output_json_path}")

if __name__ == "__main__":
    convert_txt_to_json(input_dir, output_json_path)

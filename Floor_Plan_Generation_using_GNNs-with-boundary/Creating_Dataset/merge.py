import json

with open("C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Creating_Dataset/generating-graphs.ipynb", "r", encoding="utf-8") as f:
    notebook = json.load(f)

merged_code = ""

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cleaned_lines = []
        for line in cell["source"]:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                cleaned_lines.append(line)
        if cleaned_lines:
            merged_code += "".join(cleaned_lines) + "\n\n"

with open("C:/Users/jensonyu/Documents/ENGR project/Floor_Plan_Generation_using_GNNs-with-boundary/Creating_Dataset/merged_output.py", "w", encoding="utf-8") as f:
    f.write(merged_code)


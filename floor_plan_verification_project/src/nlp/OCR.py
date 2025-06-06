import easyocr
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import re
import json

# Load image
image_path = 'data/floor_images/1_16.jpg'
image = Image.open(image_path)

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the language code

# Perform OCR
results = reader.readtext(
    image_path,
    contrast_ths=0.7,
    adjust_contrast=0.5,
    canvas_size=1024,
)

output_text = []

# Display results
for (bbox, text, prob) in results:
    output_text.append(text)

# Remove number prefixes like "1. "
processed_output = [re.sub(r'^\d+\.\s*', '', text) for text in output_text]
print("Processed Output:", processed_output)

# Merge every two lines
merged_output = [processed_output[i] + ' ' + processed_output[i+1] if i + 1 < len(processed_output) else processed_output[i]
                 for i in range(0, len(processed_output), 2)]

# Save to JSON
with open("data/2D_floorplan_extracted.json", "w") as f:
    json.dump(merged_output, f, indent=4)

print("Merged Output:", merged_output)

# Plot image with OCR bounding boxes and labels
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(image)

for (bbox, text, prob) in results:
    top_left = bbox[0]
    bottom_right = bbox[2]
    rect = patches.Rectangle(
        top_left,
        bottom_right[0] - top_left[0],
        bottom_right[1] - top_left[1],
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )
    ax.add_patch(rect)
    ax.text(top_left[0], top_left[1] - 10, text, fontsize=12, color='blue',
            bbox=dict(facecolor='white', alpha=0.5))

plt.axis('off')
plt.tight_layout()
plt.show()

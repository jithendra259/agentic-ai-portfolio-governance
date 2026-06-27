import fitz
import os

doc = fitz.open("main.pdf")
print("Total pages:", len(doc))

# We'll save the images in the artifacts directory
out_dir = r"C:\Users\jithe\.gemini\antigravity\brain\cbcdfb4f-340a-45e0-b56f-49628a389419"
os.makedirs(out_dir, exist_ok=True)

# Page numbers in python are 0-indexed. 
# Page 90 is index 89, Page 91 is index 90, Page 38 is index 37.
pages_to_render = [37, 89, 90, 94]

for p_num in pages_to_render:
    if p_num < len(doc):
        page = doc[p_num]
        pix = page.get_pixmap(dpi=150)
        output_path = os.path.join(out_dir, f"page_{p_num+1}_render.png")
        pix.save(output_path)
        print(f"Saved page {p_num+1} to {output_path}")

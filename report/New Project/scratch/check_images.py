import fitz # PyMuPDF
doc = fitz.open("main.pdf")
print("Total pages:", len(doc))
for i in range(len(doc)):
    page = doc[i]
    images = page.get_images(full=True)
    if images:
        print(f"Page {i+1} has {len(images)} images")

"""Render main.pdf to render/slide-NN.png (and 2x2 contact sheets) for visual checks. pv2 venv: pypdfium2 + PIL."""
import sys
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image
here = Path(__file__).parent
out = here / "render"; out.mkdir(exist_ok=True)
pdf = pdfium.PdfDocument(str(here / "main.pdf"))
pages = []
for i in range(len(pdf)):
    im = pdf[i].render(scale=1600 / pdf[i].get_width()).to_pil()
    im.save(out / f"slide-{i+1:02d}.png"); pages.append(im)
print(len(pdf), "pages")
for k in range(0, len(pages), 4):
    grp = pages[k:k + 4]; w, h = grp[0].size
    sheet = Image.new("RGB", (2 * w, 2 * h), "white")
    for j, im in enumerate(grp):
        sheet.paste(im, ((j % 2) * w, (j // 2) * h))
    sheet.save(out / f"sheet-{k//4+1}.png")

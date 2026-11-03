"""Draws the app icon (indigo tile, brass Tamil 'கு' and a sound wave) and writes Kuralkoodu.icns via iconutil."""
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

out = Path(sys.argv[1] if len(sys.argv) > 1 else "build")
out.mkdir(parents=True, exist_ok=True)
S = 1024
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle([64, 64, S - 64, S - 64], radius=200, fill=(20, 26, 44, 255))
d.rounded_rectangle([64, 64, S - 64, S - 64], radius=200, outline=(59, 71, 112, 255), width=10)
font = None
for f in ["/System/Library/Fonts/Supplemental/Tamil Sangam MN.ttc", "/System/Library/Fonts/Supplemental/Tamil MN.ttc",
          "/Library/Fonts/Arial Unicode.ttf"]:
    if Path(f).exists():
        font = ImageFont.truetype(f, 470)
        break
if font:
    d.text((S / 2, S * 0.44), "கு", font=font, fill=(233, 178, 76, 255), anchor="mm")
else:
    d.ellipse([S * 0.3, S * 0.2, S * 0.7, S * 0.6], outline=(233, 178, 76, 255), width=48)
import math
pts = [(160 + i * 2, 790 + 55 * math.sin(i / 22) * math.exp(-((i - 350) / 260) ** 2)) for i in range(352)]
d.line(pts, fill=(63, 176, 210, 255), width=26, joint="curve")
iconset = out / "Kuralkoodu.iconset"
iconset.mkdir(exist_ok=True)
for size in (16, 32, 128, 256, 512):
    img.resize((size, size), Image.LANCZOS).save(iconset / f"icon_{size}x{size}.png")
    img.resize((size * 2, size * 2), Image.LANCZOS).save(iconset / f"icon_{size}x{size}@2x.png")
img.save(out / "icon_1024.png")
if sys.platform == "darwin":
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out / "Kuralkoodu.icns")], check=True)
print("icon written to", out)

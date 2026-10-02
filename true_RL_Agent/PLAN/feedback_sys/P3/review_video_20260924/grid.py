import sys, numpy as np, imageio_ffmpeg
from PIL import Image, ImageDraw
path, out, steps = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(",")]
cols = int(sys.argv[4]) if len(sys.argv) > 4 else 4
crop = sys.argv[5] if len(sys.argv) > 5 else None
want = {s - 1: s for s in steps}
reader = imageio_ffmpeg.read_frames(path)
meta = next(reader); W, H = meta["size"]
frames = {}
for i, buf in enumerate(reader):
    if i in want:
        frames[want[i]] = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
    if i > max(want): break
reader.close()
tw, th = 640, 360
tiles = []
for s in steps:
    if s not in frames: continue
    im = Image.fromarray(frames[s])
    if crop:
        x0, y0, x1, y1 = [int(v) for v in crop.split(":")]
        im = im.crop((x0, y0, x1, y1))
    im = im.resize((tw, th))
    from PIL import ImageOps
    im = ImageOps.equalize(im)
    ImageDraw.Draw(im).text((8, 8), f"step {s}", fill=(255, 255, 0))
    tiles.append(im)
rows = (len(tiles) + cols - 1) // cols
g = Image.new("RGB", (cols * tw, rows * th))
for k, t in enumerate(tiles):
    g.paste(t, ((k % cols) * tw, (k // cols) * th))
g.save(out)
print(out, len(tiles), W, H)

import os, sys, argparse, traceback
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path
from PIL import Image
import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
import o_voxel

ap = argparse.ArgumentParser()
ap.add_argument("--in", dest="inp", required=True, help="image file or folder")
ap.add_argument("--out", required=True, help="output folder for GLBs")
ap.add_argument("--tex", type=int, default=2048, help="texture size (2048 or 4096)")
ap.add_argument("--faces", type=int, default=200000, help="decimation target")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--seed", type=int, default=42, help="sampling seed (pipeline default 42)")
a = ap.parse_args()

src = Path(a.inp).expanduser()
out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
exts = {".png", ".jpg", ".jpeg", ".webp"}
files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in exts)
print(f"{len(files)} image(s) to process", flush=True)

pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()

ok, bad = 0, []
for f in files:
    dst = out / f"{f.stem}.glb"
    if dst.exists() and not a.overwrite:
        print(f"skip {f.name} (exists)", flush=True); continue
    try:
        print(f"==> {f.name}", flush=True)
        mesh = pipeline.run(Image.open(f), seed=a.seed)[0]
        if len(files) == 1:  # one image per process: drop the models so mesh cleanup gets the VRAM
            pipeline.models.clear(); pipeline.image_cond_model = pipeline.rembg_model = None
            import gc; gc.collect(); torch.cuda.empty_cache()
        mesh.simplify(16777216)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
            coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=a.faces, texture_size=a.tex,
            remesh=True, remesh_band=1, remesh_project=0, verbose=False)
        glb.export(str(dst), extension_webp=True)
        print(f"saved {dst}", flush=True); ok += 1
    except Exception:
        traceback.print_exc(); bad.append(f.name)
    finally:
        torch.cuda.empty_cache()

print(f"done: {ok} ok, {len(bad)} failed {bad}")

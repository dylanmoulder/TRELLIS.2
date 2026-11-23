"""
TRELLIS.2 photos -> GLB.

  One GLB per image (original mode):
      python -u gen.py --in <image or folder> --out <folder>

  Plan mode (several jobs, each job fuses 1+ photos of ONE object into one GLB):
      python -u gen.py --plan plan.json

  plan.json:
      {
        "out": "/mnt/c/Users/me/renders",          # output folder
        "tex": 2048, "faces": 200000, "seed": 42,  # defaults for every job (optional)
        "resolution": "1024",                      # "512" fast | "1024" standard | "1536" high
        "preview": true,                           # save <name>_preview.png (4 views)
        "overwrite": false,
        "jobs": [
          {"name": "ring_combined", "images": ["front.jpg", "side.jpg"],
           "mode": "stochastic", "weights": [2, 1], "seed": 7},
          {"name": "ring_variant", "images": ["other.jpg"]}
        ]
      }

Paths may be WSL paths, C:\\... paths or \\\\wsl.localhost\\<distro>\\... paths.
Relative image paths in a plan are resolved against the plan file's folder.
"""
import os, re, json, argparse, traceback, gc
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
RESOLUTIONS = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}
MODES = {"stochastic", "multidiffusion"}


def log(*a):
    print(*a, flush=True)


def wsl_path(p, base: Path = None) -> Path:
    """Accept WSL paths, C:\\... paths and \\\\wsl.localhost\\<distro>\\... paths."""
    s = str(p).strip().strip('"')
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', s)
    if m:
        s = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace('\\', '/')
    m = re.match(r'^[\\/]{2}wsl(?:\.localhost|\$)[\\/][^\\/]+[\\/]?(.*)$', s, re.I)
    if m:
        s = '/' + m.group(1).replace('\\', '/')
    path = Path(s).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path


def safe_name(name: str) -> str:
    return re.sub(r'[^\w\-. ]+', '_', str(name)).strip(' .') or 'model'


def load_image(path: Path):
    from PIL import Image, ImageOps
    img = Image.open(path)
    return ImageOps.exif_transpose(img)  # phone photos: honour the rotation the user sees


def build_jobs(a, ap):
    """Returns (out_dir, jobs, preview, overwrite). Validates everything before the slow model load."""
    if a.plan:
        plan_path = wsl_path(a.plan)
        plan = json.loads(plan_path.read_text())
        out = wsl_path(plan.get("out") or a.out or plan_path.parent)
        d = {"tex": plan.get("tex", a.tex), "faces": plan.get("faces", a.faces),
             "seed": plan.get("seed", a.seed), "resolution": str(plan.get("resolution", "1024"))}
        jobs = []
        for j in plan.get("jobs", []):
            imgs = [wsl_path(p, plan_path.parent) for p in j.get("images", [])]
            jobs.append(dict(
                name=safe_name(j.get("name") or (imgs[0].stem if imgs else "model")),
                images=imgs,
                mode=j.get("mode", "stochastic"),
                weights=j.get("weights"),
                seed=int(j.get("seed", d["seed"])),
                tex=int(j.get("tex", d["tex"])),
                faces=int(j.get("faces", d["faces"])),
                resolution=str(j.get("resolution", d["resolution"])),
            ))
        preview = bool(plan.get("preview", True))
        overwrite = bool(plan.get("overwrite", a.overwrite))
    else:
        if not (a.inp and a.out):
            ap.error("use --in and --out, or --plan")
        src, out = wsl_path(a.inp), wsl_path(a.out)
        files = [src] if src.is_file() else sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
        jobs = [dict(name=f.stem, images=[f], mode="stochastic", weights=None, seed=a.seed,
                     tex=a.tex, faces=a.faces, resolution="1024") for f in files]
        preview, overwrite = a.preview, a.overwrite

    errors, seen = [], set()
    for j in jobs:
        n = j["name"]
        if n in seen: errors.append(f"{n}: duplicate job name")
        seen.add(n)
        if not j["images"]: errors.append(f"{n}: no images")
        for p in j["images"]:
            if not p.is_file(): errors.append(f"{n}: image not found: {p}")
        if j["mode"] not in MODES: errors.append(f"{n}: mode must be one of {sorted(MODES)}")
        if j["resolution"] not in RESOLUTIONS: errors.append(f"{n}: resolution must be one of {list(RESOLUTIONS)}")
        w = j["weights"]
        if w is not None and (len(w) != len(j["images"]) or any(x < 0 for x in w) or sum(w) <= 0):
            errors.append(f"{n}: weights need one value >= 0 per image, not all zero")
    if errors:
        raise SystemExit("plan has problems:\n  " + "\n  ".join(errors))
    return out, jobs, preview, overwrite


_ENVMAP = None

def save_preview(mesh, path: Path):
    """2x4 grid: shaded (forest HDRI) on top, normals below, four angles. Best effort."""
    global _ENVMAP
    import cv2, numpy as np, torch
    from PIL import Image
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap
    if _ENVMAP is None:
        hdr = cv2.cvtColor(cv2.imread(str(ROOT / 'assets/hdri/forest.exr'), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        _ENVMAP = EnvMap(torch.tensor(hdr, dtype=torch.float32, device='cuda'))
    views = render_utils.render_snapshot(mesh, resolution=512, r=2, fov=36, nviews=4, envmap=_ENVMAP, verbose=False)
    rows = [np.concatenate(views[k], axis=1) for k in ("shaded", "normal") if k in views]
    grid = Image.fromarray(np.concatenate(rows, axis=0))
    grid = grid.resize((grid.width * 3 // 4, grid.height * 3 // 4), Image.Resampling.LANCZOS)
    grid.save(path)


def write_results(out: Path, records: dict):
    f = out / "results.json"
    try:
        old = json.loads(f.read_text()) if f.exists() else {}
    except Exception:
        old = {}
    old.update(records)
    f.write_text(json.dumps(old, indent=2))


def main():
    ap = argparse.ArgumentParser(description="TRELLIS.2 photos -> GLB")
    ap.add_argument("--in", dest="inp", help="image file or folder (one GLB per image)")
    ap.add_argument("--out", help="output folder for GLBs")
    ap.add_argument("--plan", help="JSON plan; each job fuses one or more photos of one object into one GLB")
    ap.add_argument("--tex", type=int, default=2048, help="texture size (2048 or 4096)")
    ap.add_argument("--faces", type=int, default=200000, help="decimation target")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="sampling seed (pipeline default 42)")
    ap.add_argument("--preview", action="store_true", help="--in mode: also save <name>_preview.png")
    a = ap.parse_args()

    out, jobs, preview, overwrite = build_jobs(a, ap)
    out.mkdir(parents=True, exist_ok=True)
    todo = []
    for j in jobs:
        if (out / f"{j['name']}.glb").exists() and not overwrite:
            log(f"skip {j['name']} (exists)")
        else:
            todo.append(j)
    log(f"{len(todo)} model(s) to make, {sum(len(j['images']) for j in todo)} image(s) -> {out}")
    if not todo:
        log("done: 0 ok, 0 failed []"); return

    import torch, o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
    pipeline.cuda()

    ok, bad = 0, []
    for idx, j in enumerate(todo):
        name, imgs = j["name"], j["images"]
        dst = out / f"{name}.glb"
        rec = {"glb": None, "preview": None, "images": [str(p) for p in imgs], "mode": j["mode"],
               "weights": j["weights"], "seed": j["seed"], "resolution": j["resolution"], "status": "failed"}
        try:
            how = f"{len(imgs)} images fused [{j['mode']}]" if len(imgs) > 1 else imgs[0].name
            log(f"==> {name}: {how}")
            pil = [load_image(p) for p in imgs]
            ptype = RESOLUTIONS[j["resolution"]]
            if len(pil) == 1:
                mesh = pipeline.run(pil[0], seed=j["seed"], pipeline_type=ptype)[0]
            else:
                mesh = pipeline.run_multi_image(pil, seed=j["seed"], mode=j["mode"],
                                                weights=j["weights"], pipeline_type=ptype)[0]
            if idx == len(todo) - 1:  # last model of this process: drop the models so mesh cleanup gets the VRAM
                pipeline.models.clear(); pipeline.image_cond_model = pipeline.rembg_model = None
                gc.collect(); torch.cuda.empty_cache()
            mesh.simplify(16777216)
            if preview:
                try:
                    pv = out / f"{name}_preview.png"
                    save_preview(mesh, pv)
                    rec["preview"] = str(pv); log(f"preview {pv}")
                except Exception as e:
                    log(f"(preview skipped: {type(e).__name__}: {e})")
                finally:
                    torch.cuda.empty_cache()
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
                coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=j["faces"], texture_size=j["tex"],
                remesh=True, remesh_band=1, remesh_project=0, verbose=False)
            glb.export(str(dst), extension_webp=True)
            rec.update(glb=str(dst), status="ok")
            log(f"saved {dst}"); ok += 1
        except Exception as e:
            traceback.print_exc(); bad.append(name)
            rec["error"] = f"{type(e).__name__}: {e}"
        finally:
            torch.cuda.empty_cache()
            write_results(out, {name: rec})

    log(f"done: {ok} ok, {len(bad)} failed {bad}")


if __name__ == "__main__":
    main()

import io, json, os, re, subprocess, uuid
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP, Image

ROOT = Path.home() / "TRELLIS.2"
PY = Path.home() / "miniconda3/envs/trellis2/bin/python"
EXTS = {".png", ".jpg", ".jpeg", ".webp"}
JOBS = {}

INSTRUCTIONS = """
TRELLIS.2 turns photos into textured 3D models (GLB) on the user's 8GB GPU (WSL2). Paths can be
Windows (C:\\...) or WSL (/home/..., /mnt/c/...).

When the user gives you pictures to turn into 3D, work like this:
1. Call look_at_images on their files or folder so you actually see every photo.
2. Tell them briefly what you see: which photos look like the SAME object from different angles,
   which show a DIFFERENT object or variant, and differences that matter (shape, stones/details,
   colour/material, missing parts, busy backgrounds, blur, crops, hands or reflections).
3. Ask what they want before rendering. Typical questions: fuse these photos into one model, or
   one model each? Which photo is the main/front view (it goes first / gets the highest weight)?
   Leave any photo out? Quality: 512 fast, 1024 standard, 1536 high. Texture 2048 or 4096?
   A couple of seeds as variations?
4. Sum up the plan in a few lines, then call start_render ONCE with every job (the model takes
   minutes to load, so batch everything into one run).
5. A render takes several minutes per model: check job_status occasionally, not in a tight loop.
   When it finishes, call view_result to show the preview and give the GLB path.

Limits to be upfront about: TRELLIS.2 only follows pictures. It cannot apply text edits
("make the band thinner", "make the stone oval"); for that the user needs a photo or edited image
showing the change. Fusing photos of different objects gives a muddled hybrid, not a clean merge.
Multi-photo fusion is training-free and experimental: if a fused result looks worse than the best
single photo, offer a single-photo render of that one. Clean photos on plain backgrounds work best.
One GPU job at a time; it will refuse to start while another TRELLIS job or the Gradio app runs.
""".strip()

mcp = FastMCP("trellis2", instructions=INSTRUCTIONS)


def wsl_path(p: str) -> Path:
    """Accept WSL paths, C:\\... paths and \\\\wsl.localhost\\<distro>\\... paths."""
    s = str(p).strip().strip('"')
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', s)
    if m:
        s = f"/mnt/{m.group(1).lower()}/" + m.group(2).replace('\\', '/')
    m = re.match(r'^[\\/]{2}wsl(?:\.localhost|\$)[\\/][^\\/]+[\\/]?(.*)$', s, re.I)
    if m:
        s = '/' + m.group(1).replace('\\', '/')
    return Path(s).expanduser()


def _expand_images(paths: list[str]) -> list[Path]:
    files = []
    for raw in paths:
        p = wsl_path(raw)
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        else:
            files.append(p)
    return files


def _gpu_busy() -> list[str]:
    """Any TRELLIS generation or Gradio app already running (started by anyone)."""
    busy = []
    for d in Path('/proc').iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='ignore')
            if 'python' not in cmd or not re.search(r'(^|[ /])(gen|app|app_texturing|example)\.py( |$)', cmd):
                continue
            in_repo = str(ROOT) in cmd or Path(os.readlink(d / 'cwd')) == ROOT
        except OSError:
            continue
        if in_repo:
            busy.append(f"pid {d.name}: {cmd.strip()[:160]}")
    return busy


def _launch(args: list[str]) -> str:
    running = [j for j, (p, _) in JOBS.items() if p.poll() is None]
    busy = _gpu_busy()
    if running or busy:
        return ("GPU is busy (8GB: one job at a time). "
                + (f"Running job(s): {', '.join(running)}. " if running else "")
                + ("\n" + "\n".join(busy) if busy else ""))
    jid = uuid.uuid4().hex[:8]
    log = Path(f"/tmp/trellis_{jid}.log")
    p = subprocess.Popen([str(PY), "-u", "gen.py", *args], cwd=ROOT,
                         stdout=open(log, "w"), stderr=subprocess.STDOUT)
    JOBS[jid] = (p, log)
    return f"started {jid} (log {log})"


def _thumb(path: Path, max_size: int) -> Image:
    from PIL import Image as PIL, ImageOps
    im = ImageOps.exif_transpose(PIL.open(path))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = PIL.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    im = im.convert("RGB")
    im.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return Image(data=buf.getvalue(), format="jpeg")


@mcp.tool(structured_output=False)
def look_at_images(paths: list[str], max_size: int = 768) -> list:
    """Show you the user's photos (files and/or folders) so you can compare them before rendering.
    Returns each photo with its number, file name and size. Call this first in the multi-photo flow."""
    files = _expand_images(paths)
    if not files:
        return ["No images found (.png .jpg .jpeg .webp)."]
    out = []
    for i, f in enumerate(files[:12], 1):
        if not f.is_file():
            out.append(f"Image {i}: NOT FOUND {f}"); continue
        try:
            from PIL import Image as PIL
            with PIL.open(f) as im:
                w, h = im.size
            out += [f"Image {i}: {f.name}  ({w}x{h})  {f}", _thumb(f, max_size)]
        except Exception as e:
            out.append(f"Image {i}: could not open {f}: {e}")
    if len(files) > 12:
        out.append(f"...and {len(files) - 12} more not shown (max 12 at a time).")
    return out


class RenderJob(BaseModel):
    name: str = Field(description="Output file name without extension, e.g. 'ring_combined'.")
    images: list[str] = Field(description="Photos of ONE object. One photo = normal render; 2+ photos are "
                                          "fused into one model. Put the main/front-on photo first.")
    mode: Literal["stochastic", "multidiffusion"] = Field(
        "stochastic", description="Fusion mode for 2+ photos: 'stochastic' (fast) or 'multidiffusion' "
                                  "(about N times slower, steadier blend).")
    weights: Optional[list[float]] = Field(None, description="Optional importance per photo, e.g. [2, 1, 1].")
    seed: Optional[int] = Field(None, description="Seed; change it for a different variation.")


@mcp.tool()
def start_render(jobs: list[RenderJob], output_dir: Optional[str] = None,
                 resolution: Literal["512", "1024", "1536"] = "1024",
                 tex: int = 2048, faces: int = 200000, preview: bool = True,
                 overwrite: bool = False) -> str:
    """Render the agreed plan: every job becomes one GLB (plus a preview PNG). All jobs run in one
    process so the model loads once. output_dir defaults to a 'trellis_out' folder next to the first
    photo. Returns a job id for job_status / view_result."""
    if not jobs:
        return "No jobs given."
    first = _expand_images(jobs[0].images)
    out = wsl_path(output_dir) if output_dir else (first[0].parent / "trellis_out" if first else ROOT / "outputs")
    plan_jobs = []
    for j in jobs:
        imgs = [str(p) for p in _expand_images(j.images)]
        missing = [p for p in imgs if not Path(p).is_file()]
        if missing:
            return f"Job '{j.name}': image(s) not found: {missing}"
        d = {"name": j.name, "images": imgs, "mode": j.mode}
        if j.weights is not None: d["weights"] = j.weights
        if j.seed is not None: d["seed"] = j.seed
        plan_jobs.append(d)
    plan = {"out": str(out), "resolution": resolution, "tex": tex, "faces": faces,
            "preview": preview, "overwrite": overwrite, "jobs": plan_jobs}
    out.mkdir(parents=True, exist_ok=True)
    plan_file = out / f"plan_{datetime.now():%Y%m%d_%H%M%S}.json"
    plan_file.write_text(json.dumps(plan, indent=2))
    msg = _launch(["--plan", str(plan_file)])
    return f"{msg}\noutput: {out}\nplan: {plan_file}"


@mcp.tool()
def start_batch(input_path: str, output_dir: str, tex: int = 2048, faces: int = 200000) -> str:
    """Start generating one GLB per image from an image or folder of images. Returns a job id."""
    return _launch(["--in", str(wsl_path(input_path)), "--out", str(wsl_path(output_dir)),
                    "--tex", str(tex), "--faces", str(faces)])


@mcp.tool()
def job_status(job_id: str) -> str:
    """Running/finished state plus the last lines of the job log."""
    if job_id not in JOBS:
        busy = _gpu_busy()
        return "unknown job (the MCP server may have restarted)" + ("\nGPU in use by:\n" + "\n".join(busy) if busy else "")
    p, log = JOBS[job_id]
    state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
    lines = [l for l in log.read_text(errors="ignore").replace("\r", "\n").splitlines() if l.strip()]
    return state + "\n" + "\n".join(lines[-20:])


@mcp.tool()
def list_outputs(output_dir: str) -> list[str]:
    """List GLB files in an output folder."""
    return sorted(str(p) for p in wsl_path(output_dir).glob("*.glb"))


@mcp.tool(structured_output=False)
def view_result(output_dir: str, name: Optional[str] = None) -> list:
    """Show the preview render(s) of finished models (4 angles, shaded on top, normals below) and
    their GLB paths. Without a name, shows every model in results.json."""
    out = wsl_path(output_dir)
    rf = out / "results.json"
    if not rf.exists():
        return [f"No results.json in {out} yet. GLBs there: {[p.name for p in out.glob('*.glb')]}"]
    results = json.loads(rf.read_text())
    names = [name] if name else list(results)
    items = []
    for n in names:
        r = results.get(n)
        if r is None:
            items.append(f"{n}: not in results.json (have: {list(results)})"); continue
        info = f"{n}: {r['status']}  glb={r.get('glb')}  photos={len(r.get('images', []))} mode={r.get('mode')}"
        if r.get("error"): info += f"  error={r['error']}"
        items.append(info)
        pv = r.get("preview")
        if pv and Path(pv).is_file():
            items.append(_thumb(Path(pv), 1536))
    return items


if __name__ == "__main__":
    mcp.run()

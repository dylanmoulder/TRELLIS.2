import subprocess, uuid, os
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("trellis2")
ROOT = Path.home() / "TRELLIS.2"
PY = Path.home() / "miniconda3/envs/trellis2/bin/python"
JOBS = {}

@mcp.tool()
def start_batch(input_path: str, output_dir: str, tex: int = 2048, faces: int = 200000) -> str:
    """Start generating GLBs from an image or folder of images (WSL paths). Returns a job id."""
    if any(p.poll() is None for p, _ in JOBS.values()):
        return "A job is already running (8GB GPU: one at a time). Check job_status."
    jid = uuid.uuid4().hex[:8]
    log = Path(f"/tmp/trellis_{jid}.log")
    p = subprocess.Popen(
        [str(PY), "-u", "gen.py", "--in", input_path, "--out", output_dir,
         "--tex", str(tex), "--faces", str(faces)],
        cwd=ROOT, stdout=open(log, "w"), stderr=subprocess.STDOUT)
    JOBS[jid] = (p, log)
    return f"started {jid}"

@mcp.tool()
def job_status(job_id: str) -> str:
    """Running/finished state plus the last lines of the job log."""
    if job_id not in JOBS: return "unknown job"
    p, log = JOBS[job_id]
    state = "running" if p.poll() is None else f"finished (exit {p.returncode})"
    return state + "\n" + "\n".join(log.read_text().splitlines()[-15:])

@mcp.tool()
def list_outputs(output_dir: str) -> list[str]:
    """List GLB files in an output folder."""
    return sorted(str(p) for p in Path(output_dir).expanduser().glob("*.glb"))

if __name__ == "__main__":
    mcp.run()

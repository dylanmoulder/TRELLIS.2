"""Torch-free checks of the multi-image code (safe while another GPU job holds most of the RAM)."""
import ast, json, sys, tempfile, types
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P = ROOT / "trellis2/pipelines"
fails = 0

def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1

# ---- pull the real helper functions out of the pipeline file without importing torch
src = (P / "trellis2_image_to_3d.py").read_text()
tree = ast.parse(src)
import typing
from fractions import Fraction
ns = {"List": typing.List, "Optional": typing.Optional, "Fraction": Fraction}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("_stochastic_inference", "_multidiffusion_inference"):
        exec(compile(ast.Module([node], []), "pipe", "exec"), ns)
    if isinstance(node, ast.ClassDef):
        for f in node.body:
            if isinstance(f, ast.FunctionDef) and f.name == "_normalize_weights":
                f.decorator_list = []
                exec(compile(ast.Module([f], []), "pipe", "exec"), ns)
norm = ns["_normalize_weights"]

# ---- real CFG + interval mixins, on a fake linear "flow model"
mix = {}
exec((P / "samplers/classifier_free_guidance_mixin.py").read_text(), mix)
exec((P / "samplers/guidance_interval_mixin.py").read_text(), mix)

class T(np.ndarray):
    def std(self, dim=None, keepdim=False, **kw):
        return np.asarray(self).std(axis=tuple(dim), keepdims=keepdim).view(T)
    @property
    def ndim(self): return np.asarray(self).ndim

seen_kwargs = []
class FakeBase:
    def _inference_model(self, model, x_t, t, cond=None, **kw):
        seen_kwargs.append(set(kw))
        return (x_t * 0.5 + np.asarray(cond).mean(axis=1) * (1 + t)).view(T)   # linear in cond
    def _pred_to_xstart(self, x_t, t, pred): return x_t - t * pred
    def _xstart_to_pred(self, x_t, t, x0): return (x_t - x0) / t

class Sampler(mix["GuidanceIntervalSamplerMixin"], mix["ClassifierFreeGuidanceSamplerMixin"], FakeBase):
    pass

ns["samplers"] = types.SimpleNamespace(FlowEulerSampler=FakeBase)
s = Sampler()
rng = np.random.default_rng(0)
x = rng.normal(size=(1, 8)).view(T)
conds = rng.normal(size=(3, 5, 8))
neg = np.zeros((1, 5, 8))
params = dict(neg_cond=neg, guidance_strength=7.5, guidance_rescale=0.7, guidance_interval=(0.6, 1.0))

# weights
check("weights default equal", norm(3, None) == [1/3] * 3)
check("weights normalised", np.allclose(norm(3, [2, 1, 1]), [.5, .25, .25]))
for bad in ([1, 1], [-1, 1, 1], [0, 0, 0]):
    try: norm(3, bad); check(f"weights {bad} rejected", False)
    except ValueError: check(f"weights {bad} rejected", True)

# stochastic: which image each step uses
picked = []
def spy(model, x_t, t, cond, **kw):
    picked.append(int(np.argmax([np.allclose(cond, conds[i:i+1]) for i in range(3)]))); return x_t
f = ns["_stochastic_inference"](spy, norm(3, None))
for _ in range(12): f(None, x, 0.5, conds, **params)
check(f"stochastic equal weights cycles 0,1,2 ({picked[:6]})", picked == [0, 1, 2] * 4)
picked.clear()
f = ns["_stochastic_inference"](spy, norm(3, [2, 1, 1]))
for _ in range(12): f(None, x, 0.5, conds, **params)
check(f"stochastic [2,1,1] -> 6/3/3 and starts with photo 1 ({picked})",
      picked[0] == 0 and [picked.count(i) for i in range(3)] == [6, 3, 3])

# multidiffusion == real mixin chain on the weighted-average cond (model is linear in cond)
for w in ([1, 0, 0], [1, 1, 1], [2, 1, 1]):
    wn = norm(3, w)
    md = ns["_multidiffusion_inference"](s, wn)
    avg = sum(wi * conds[i:i+1] for i, wi in enumerate(wn))
    for t in (0.95, 0.7, 0.3):      # inside and outside the guidance interval
        got = md(None, x, t, conds, concat_cond="X", **params)
        want = s._inference_model(None, x, t, avg, concat_cond="X", **params)
        check(f"multidiffusion w={w} t={t} matches CFG+interval+rescale", np.allclose(got, want))
check("extra kwargs (concat_cond) reach the model, sampler kwargs don't",
      all(k == {"concat_cond"} for k in seen_kwargs))

# ---- gen.py plan parsing (gen.py imports nothing heavy at top level)
import gen
tmp = Path(tempfile.mkdtemp())
for n in ("a.png", "b.png"): (tmp / n).write_bytes(b"x")
win = "C:\\Windows\\win.ini"
plan = {"out": str(tmp / "out"), "seed": 5, "resolution": "512", "jobs": [
    {"name": "fused/one", "images": ["a.png", str(tmp / "b.png")], "weights": [2, 1], "mode": "multidiffusion"},
    {"images": [win], "seed": 9}]}
(tmp / "plan.json").write_text(json.dumps(plan))
ap = gen.argparse.ArgumentParser()
A = types.SimpleNamespace(plan=str(tmp / "plan.json"), inp=None, out=None, tex=2048, faces=200000, seed=42, overwrite=False, preview=False)
out, jobs, preview, overwrite = gen.build_jobs(A, ap)
check("plan: relative path resolved", jobs[0]["images"][0] == tmp / "a.png")
check("plan: name sanitised", jobs[0]["name"] == "fused_one")
check("plan: C:\\ path -> /mnt/c", str(jobs[1]["images"][0]) == "/mnt/c/Windows/win.ini")
check("plan: defaults + overrides", jobs[1]["seed"] == 9 and jobs[0]["seed"] == 5 and jobs[1]["resolution"] == "512" and preview)
check("plan: unnamed job named after photo", jobs[1]["name"] == "win")
bad = {"jobs": [{"name": "x", "images": ["missing.png"], "mode": "nope", "weights": [1, 2]}]}
(tmp / "bad.json").write_text(json.dumps(bad))
A.plan = str(tmp / "bad.json")
try:
    gen.build_jobs(A, ap); check("bad plan rejected", False)
except SystemExit as e:
    msg = str(e); print("   ", msg.replace("\n", "\n    "))
    check("bad plan rejected with all 3 problems", all(k in msg for k in ("not found", "mode", "weights")))
A.plan, A.inp, A.out = None, str(tmp), str(tmp / "o")
_, jobs, _, _ = gen.build_jobs(A, ap)
check("--in folder mode still one job per image", [j["name"] for j in jobs] == ["a", "b"])

# ---- MCP server helpers
import mcp_server as m
check("wsl_path C:", str(m.wsl_path(r"C:\Users\dylan\Pictures\x.jpg")) == "/mnt/c/Users/dylan/Pictures/x.jpg")
check("wsl_path UNC", str(m.wsl_path(r"\\wsl.localhost\Ubuntu\home\dylan\TRELLIS.2\a.png")) == "/home/dylan/TRELLIS.2/a.png")
check("wsl_path WSL unchanged", str(m.wsl_path("/home/dylan/x.png")) == "/home/dylan/x.png")
ex = sorted(p for p in (ROOT / "assets").rglob("*") if p.suffix.lower() in m.EXTS)[:3]
assert len(ex) == 3, ex
res = m.look_at_images([str(p) for p in ex])
check(f"look_at_images returns label+image per photo ({len(ex)} photos)",
      len(res) == 2 * len(ex) and isinstance(res[1], m.Image) and res[0].startswith("Image 1:"))
print("   busy:", m._gpu_busy())
launched = []
m._launch = lambda args: (launched.append(args), "started TEST")[1]
r = m.start_render([m.RenderJob(name="x", images=[str(tmp / "nope.png")])])
check("start_render refuses missing photo without launching", "not found" in r and not launched)
r = m.start_render([m.RenderJob(name="combo", images=[str(p) for p in ex[:2]], weights=[2, 1]),
                    m.RenderJob(name="solo", images=[str(ex[2])], seed=3)], output_dir=str(tmp / "render"), resolution="512")
pf = Path(launched[0][1])
p = json.loads(pf.read_text())
check("start_render writes plan and launches gen.py --plan", launched[0][0] == "--plan" and p["resolution"] == "512"
      and p["jobs"][0]["weights"] == [2, 1] and "seed" not in p["jobs"][0] and p["jobs"][1]["seed"] == 3)
_, pj, _, _ = gen.build_jobs(types.SimpleNamespace(plan=str(pf), inp=None, out=None, tex=2048, faces=200000, seed=42, overwrite=False, preview=False), ap)
check("gen.py accepts the plan the MCP server writes", [j["name"] for j in pj] == ["combo", "solo"])
(tmp / "render/results.json").write_text(json.dumps({"combo": {"status": "ok", "glb": "/x.glb", "preview": str(ex[0]), "images": ["a", "b"], "mode": "stochastic"}}))
vr = m.view_result(str(tmp / "render"))
check("view_result returns summary + preview image", len(vr) == 2 and "combo: ok" in vr[0] and isinstance(vr[1], m.Image))
tools = {t.name for t in m.mcp._tool_manager.list_tools()}
check(f"MCP tools registered {sorted(tools)}", {"look_at_images", "start_render", "view_result", "start_batch", "job_status", "list_outputs"} <= tools)
check("MCP instructions set", "look_at_images" in (m.mcp.instructions or ""))

print(f"\n{'ALL PASSED' if not fails else f'{fails} FAILED'}")
sys.exit(1 if fails else 0)

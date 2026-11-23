# TRELLIS.2 image-to-3D (local, RTX 3070 Ti 8GB, WSL2)

Generate GLBs from reference images:
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate trellis2
    cd ~/TRELLIS.2
    python -u gen.py --in <image or folder> --out <output folder>     # one GLB per image
    python -u gen.py --plan plan.json                                 # multi-photo jobs (see gen.py docstring)

Multi-photo flow (branch multi-image, fork github.com/dylanmoulder/TRELLIS.2):
- pipeline.run_multi_image(images, mode='stochastic'|'multidiffusion', weights=[...]) fuses several
  photos of ONE object into one model (training-free, ported from TRELLIS v1). Main photo first.
- The trellis2 MCP server (mcp_server.py) drives it from chat: look_at_images -> ask the user about
  differences / what they want -> start_render (one call, all jobs) -> job_status -> view_result.
  Its workflow instructions live in INSTRUCTIONS in mcp_server.py.
- Torch-free tests: python tests/test_multi_image.py

Rules:
- Only one GPU job at a time. Never run while the Gradio app (app.py) is open.
  The MCP server refuses to start a job while any gen.py/app.py from this folder is running.
- Do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments (crashes WSL here).
- Model load takes minutes; always process a whole folder / whole plan in one run.
- A running gen.py holds ~23GB of WSL RAM: don't import torch in another process meanwhile.
- Existing GLBs are skipped; use --overwrite to redo. Failures are listed at the end and in results.json.
- If OOM: lower --tex to 1024 or --faces to 100000, and close Windows apps using the GPU.
- Input: clean single plane, plain background works best. Output is PBR GLB.
- TRELLIS.2 only follows pictures: no text edits ("make the band thinner").

## Synthetic dataset build guide (R2R / RxR)

This folder contains utilities to produce annotated navigation data and chain-of-thought (CoT) reasoning for visual navigation.

### 1) Prepare environment

Set API configs (or pass via CLI):
```bash
export OPENAI_API_KEY=YOUR_KEY
export OPENAI_BASE_URL=https://api.your-endpoint/v1/
export OPENAI_MODEL=gemini-2.5-pro
```

Recommended Python packages are in the project `requirements.txt`.

### 2) Prepare inputs

- R2R input: a JSON with `episodes` each containing `scene_id`, `reference_path`, `instruction`, etc.
- RxR input: a JSON/JSONL with entries containing `scan`, `path`, `instruction`, etc. (standard RxR schema).
- Optional `--images-root`: folder containing Matterport skybox images organized as:
  - `<images-root>/<scan>/<scan>/matterport_skybox_images/*_skybox{1..5}_sami.jpg`

### 3) Generate annotated dataset

Use `annotate_nav.py` to produce a unified annotated file that includes a `navigation_actions` list and optional `images` per episode.

R2R → annotated
```bash
python build_data/annotate_nav.py \
  --format r2r \
  --input /path/to/r2r.json \
  --output build_data/train_annotated.json \
  --images-root /path/to/scans
```

RxR → annotated
```bash
python build_data/annotate_nav.py \
  --format rxr \
  --input /path/to/rxr.jsonl \
  --output build_data/rxr_annotated.json \
  --images-root /path/to/scans
```

Notes:
- R2R: `navigation_actions` are derived from `reference_path` by segment distance and yaw on the XZ plane.
- RxR: `navigation_actions` connect consecutive `path` viewpoint IDs (no metric distances without pose graph). `reference_viewpoints` is also included.

### 4) Build CoT synthetic data

Use `build_cot.py` to call the multimodal LLM with the instruction and images to generate reasoning (<think>) and pair it with ground-truth actions (<answer>).

Example (R2R or RxR annotated):
```bash
python build_data/build_cot.py \
  --input build_data/train_annotated.json \
  --output build_data/cot_results.json \
  --model ${OPENAI_MODEL} \
  --base-url ${OPENAI_BASE_URL} \
  --api-key ${OPENAI_API_KEY} \
  --save-every 5 \
  --reuse-every-n 3 \
  --max-images 10
```

Key flags:
- `--reuse-every-n`: call the LLM every N episodes and reuse reasoning for the episodes in between (speed/cost control). Set to `1` to call for every episode.
- `--max-images`: limit images per episode (e.g., 10).

Output format (excerpt):
```json
{
  "total_episodes": 123,
  "processed_date": "2025-10-30",
  "model": "gemini-2.5-pro",
  "results": [
    {
      "episode_id": 1,
      "trajectory_id": 4,
      "scene_id": "mp3d/scan/scan.glb",
      "instruction": "...",
      "start_position": [...],
      "goal_position": [...],
      "geodesic_distance": 6.42,
      "num_images": 10,
      "images": ["/abs/path/skybox1.jpg", "..."],
      "cot_data": "<think>...reasoning...</think>\n\n<answer>\n[...]\n</answer>",
      "reference_viewpoints": ["vp1", "vp2", "..."]
    }
  ]
}
```

### 5) Tips

- For RxR, exact distances/angles require loading the pose graph; current actions encode viewpoint connectivity.
- Ensure image paths are accessible to avoid missing-image warnings.
- Tune temperature/max tokens via env or flags to control generation length and style.



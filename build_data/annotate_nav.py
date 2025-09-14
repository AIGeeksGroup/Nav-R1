import json
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


def _compute_yaw_deg(dx: float, dz: float) -> float:
    """Return heading yaw in degrees in the XZ-plane, range [-180, 180)."""
    # atan2(y, x) but our forward plane is (dx, dz)
    yaw_rad = math.atan2(dz, dx)
    yaw_deg = math.degrees(yaw_rad)
    # Normalize to [-180, 180)
    if yaw_deg >= 180.0:
        yaw_deg -= 360.0
    if yaw_deg < -180.0:
        yaw_deg += 360.0
    return yaw_deg


def build_navigation_actions(reference_path: List[List[float]]) -> List[Dict[str, Any]]:
    """
    Convert a reference polyline (sequence of 3D points) into a simple list of
    navigation actions. Each action describes a straight-line movement from one
    waypoint to the next with associated distance and heading.
    """
    actions: List[Dict[str, Any]] = []
    if not reference_path or len(reference_path) < 2:
        return actions

    for idx in range(len(reference_path) - 1):
        x0, y0, z0 = reference_path[idx]
        x1, y1, z1 = reference_path[idx + 1]

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        yaw_deg = _compute_yaw_deg(dx, dz)

        actions.append({
            "step_index": idx,
            "action": "MOVE",
            "from": [x0, y0, z0],
            "to": [x1, y1, z1],
            "distance": distance,
            "heading_yaw_deg": yaw_deg,
        })

    return actions


def _extract_scan_id(scene_id: str) -> Optional[str]:
    """Extract scan id (e.g., '7y3sRwLe3Va') from scene_id like 'mp3d/7y3sRwLe3Va/7y3sRwLe3Va.glb'."""
    try:
        parts = scene_id.split("/")
        # Expecting ["mp3d", scan_id, scan_id.glb]
        if len(parts) >= 2:
            return parts[1]
    except Exception:
        pass
    return None


def _collect_skybox_images(images_root: Optional[str], scene_id: str) -> List[str]:
    """Collect skybox image paths for a given scene_id under images_root.

    Expected layout: <images_root>/<scan_id>/<scan_id>/matterport_skybox_images/*_skybox{1..5}_sami.jpg
    Returns a sorted list of absolute string paths; empty list if none or images_root not provided.
    """
    if not images_root:
        return []
    scan_id = _extract_scan_id(scene_id)
    if not scan_id:
        return []
    base = Path(images_root) / scan_id / scan_id / "matterport_skybox_images"
    if not base.exists():
        return []
    # Collect files matching pattern; include skybox1..5 for all panos
    files = sorted([str(p) for p in base.glob("*_skybox*_sami.jpg")])
    return files


def _read_json_or_jsonl(path: str) -> Union[Dict[str, Any], List[Any]]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL
        lines = [line for line in text.splitlines() if line.strip()]
        return [json.loads(line) for line in lines]


def annotate_r2r(input_path: str, output_path: str, images_root: Optional[str] = None) -> None:
    """Read R2R-like episodes and write an annotated file with navigation_actions."""
    data = _read_json_or_jsonl(input_path)
    if isinstance(data, list):
        # Wrap if the file is a raw list of episodes
        data = {"episodes": data}

    episodes = data.get("episodes", [])
    out_episodes: List[Dict[str, Any]] = []

    for ep in episodes:
        ref_path = ep.get("reference_path", [])
        nav_actions = build_navigation_actions(ref_path)

        # Copy episode and add navigation_actions
        new_ep = dict(ep)
        new_ep["navigation_actions"] = nav_actions
        # Optionally attach images list from scene dataset
        scene_id = new_ep.get("scene_id", "")
        images = _collect_skybox_images(images_root, scene_id)
        if images:
            new_ep["images"] = images
        out_episodes.append(new_ep)

    out_data = {"episodes": out_episodes}

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)


def _build_actions_from_viewpoints(viewpoints: List[str]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    if not viewpoints or len(viewpoints) < 2:
        return actions
    for idx in range(len(viewpoints) - 1):
        actions.append({
            "step_index": idx,
            "action": "MOVE_NODE",
            "from_viewpoint": viewpoints[idx],
            "to_viewpoint": viewpoints[idx + 1],
            "distance": None,
            "heading_yaw_deg": None,
        })
    return actions


def annotate_rxr(input_path: str, output_path: str, images_root: Optional[str] = None) -> None:
    """Read RxR entries and write a unified episodes JSON with navigation_actions."""
    data = _read_json_or_jsonl(input_path)
    # RxR files are typically a list of entries
    entries: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("data", [])

    out_episodes: List[Dict[str, Any]] = []
    for e in entries:
        scan = e.get("scan", "")
        scene_id = f"mp3d/{scan}/{scan}.glb" if scan else ""
        viewpoints = e.get("path", [])
        nav_actions = _build_actions_from_viewpoints(viewpoints)

        images = _collect_skybox_images(images_root, scene_id)

        instruction_text = e.get("instruction", "")
        # Map to unified episode structure expected downstream
        ep: Dict[str, Any] = {
            "episode_id": e.get("instruction_id", 0),
            "trajectory_id": e.get("path_id", 0),
            "scene_id": scene_id,
            "start_position": None,
            "start_rotation": None,
            "info": {"geodesic_distance": None},
            "goals": [{"position": None, "radius": None}],
            "instruction": {
                "instruction_text": instruction_text,
                "language": e.get("language"),
                "heading": e.get("heading"),
            },
            "images": images,
            "navigation_actions": nav_actions,
            "reference_viewpoints": viewpoints,
            "split": e.get("split"),
            "annotator_id": e.get("annotator_id"),
            "timed_instruction": e.get("timed_instruction"),
            "edit_distance": e.get("edit_distance"),
        }
        out_episodes.append(ep)

    out_data = {"episodes": out_episodes}
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Annotate episodes (R2R or RxR) with navigation_actions.")
    parser.add_argument("--input", required=True, help="Input JSON/JSONL path")
    parser.add_argument("--output", default="build_data/train_annotated.json", help="Output annotated JSON path")
    parser.add_argument("--images-root", default=None, help="Root directory of scans for skybox images")
    parser.add_argument("--format", choices=["r2r", "rxr"], required=True, help="Input dataset format")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        exit(1)

    if args.format == "r2r":
        annotate_r2r(args.input, args.output, images_root=args.images_root)
    else:
        annotate_rxr(args.input, args.output, images_root=args.images_root)
    print(f"✓ Annotated file written to: {args.output}")





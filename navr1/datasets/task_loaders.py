from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass
import os
import json

from torch.utils.data import Dataset, DataLoader


class JsonlDataset(Dataset):
    """
    Generic JSONL dataset for embodied tasks.
    Expected fields per line: instruction (str), optional: history_images, target, action_space, labels
    """

    def __init__(self, jsonl_path: str):
        self.items: List[Dict[str, Any]] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    self.items.append(obj)
                except Exception:
                    continue

        if not self.items:
            # Minimal stub to keep pipeline runnable
            self.items = [
                {
                    "instruction": "Go to the kitchen and find the refrigerator.",
                    "history_images": [],
                    "action_space": ["forward", "left", "right", "stop"],
                    "target": "refrigerator",
                }
            ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        obj = self.items[idx]
        return {
            "instruction": obj.get("instruction", ""),
            "history_images": obj.get("history_images", []),
            "action_space": obj.get("action_space", ["forward", "left", "right", "stop"]),
            "target": obj.get("target"),
            "labels": obj.get("labels"),
        }


def _collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "instruction": [b.get("instruction", "") for b in batch],
        "history_images": [b.get("history_images", []) for b in batch],
        "action_space": [b.get("action_space", []) for b in batch],
        "target": [b.get("target") for b in batch],
        "labels": [b.get("labels") for b in batch],
    }


def _path_from_cfg(cfg, task: str, split: str) -> Optional[str]:
    task_cfg: Dict[str, Any] = getattr(cfg.task_finetune, task, {})
    dataset_path = task_cfg.get("dataset_path") if isinstance(task_cfg, dict) else None
    if not dataset_path:
        return None
    jsonl_path = os.path.join(dataset_path, f"{split}.jsonl")
    return jsonl_path if os.path.exists(jsonl_path) else None


def build_task_dataloader(cfg, task: str, split: str = "train", shuffle: bool = True) -> Iterable[Dict[str, Any]]:
    # Determine jsonl path for the given task
    jsonl_path = _path_from_cfg(cfg, task, split)

    if jsonl_path is None:
        # Fallback to generic dataset path if provided
        if getattr(cfg, "dataset", None) and getattr(cfg.dataset, "path", None):
            maybe = os.path.join(cfg.dataset.path, f"{split}.jsonl")
            jsonl_path = maybe if os.path.exists(maybe) else None

    # Create dataset (falls back to stub if file missing)
    ds = JsonlDataset(jsonl_path) if jsonl_path else JsonlDataset("/dev/null")

    # Batch size selection: prefer task-specific, else dataset batch size
    batch_size = 4
    task_cfg = getattr(cfg.task_finetune, task, {})
    if isinstance(task_cfg, dict):
        batch_size = int(task_cfg.get("batch_size", getattr(cfg.dataset, "batch_size", 4)))
    else:
        batch_size = getattr(cfg.dataset, "batch_size", 4)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=getattr(cfg.dataset, "num_workers", 2),
        collate_fn=_collate_fn,
    )



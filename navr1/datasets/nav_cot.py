from typing import Any, Dict, Iterable, List, Optional
from dataclasses import dataclass
import json
import os

from torch.utils.data import Dataset, DataLoader

from navr1.config import DatasetConfig


@dataclass
class NavExample:
    instruction: str
    history_images: List[str]
    action_space: List[str]
    target: Optional[str] = None


class NavCOTDataset(Dataset):
    def __init__(self, cfg: DatasetConfig, split: str):
        self.cfg = cfg
        self.split = split
        # Minimal stub data to make pipeline runnable; users can point to real data
        self.examples: List[NavExample] = [
            NavExample(
                instruction="Go to the kitchen and find the refrigerator.",
                history_images=[],
                action_space=["forward", "left", "right", "stop"],
                target="refrigerator",
            )
        ]

        if cfg.path and os.path.isdir(cfg.path):
            maybe_jsonl = os.path.join(cfg.path, f"{split}.jsonl")
            if os.path.exists(maybe_jsonl):
                self.examples = []
                with open(maybe_jsonl, "r") as f:
                    for line in f:
                        obj = json.loads(line)
                        self.examples.append(
                            NavExample(
                                instruction=obj.get("instruction", ""),
                                history_images=obj.get("history_images", []),
                                action_space=obj.get("action_space", []),
                                target=obj.get("target"),
                            )
                        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "instruction": ex.instruction,
            "history_images": ex.history_images,
            "action_space": ex.action_space,
            "target": ex.target,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "instruction": [b["instruction"] for b in batch],
        "history_images": [b["history_images"] for b in batch],
        "action_space": [b["action_space"] for b in batch],
        "target": [b["target"] for b in batch],
    }


def build_dataloader(cfg: DatasetConfig, split: Optional[str] = None, shuffle: Optional[bool] = None) -> Iterable[Dict[str, Any]]:
    ds = NavCOTDataset(cfg, split or cfg.split)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle if shuffle is None else shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

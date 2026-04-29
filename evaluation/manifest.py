"""Per-(method, dataset) generation manifest.

Tracks the inference hyperparameters used for the images under a directory.
On resume, the new manifest must match the existing one on every locked
field (everything except ``max_seed_generated``, which is informational —
the file system is the source of truth for what's already generated).
"""
import json
import os
from dataclasses import asdict, dataclass
from typing import Optional


MANIFEST_NAME = "manifest.json"


@dataclass
class GenerationManifest:
    method: str
    dataset: str
    checkpoint_id: Optional[str]
    num_inference_steps: int
    guidance_scale: float
    resolution: int
    scheduler_class: str
    max_seed_generated: int  # informational; -1 = nothing generated yet


_LOCKED_FIELDS = (
    "method",
    "dataset",
    "checkpoint_id",
    "num_inference_steps",
    "guidance_scale",
    "resolution",
    "scheduler_class",
)


def manifest_path(directory: str) -> str:
    return os.path.join(directory, MANIFEST_NAME)


def read_manifest(directory: str) -> Optional[GenerationManifest]:
    p = manifest_path(directory)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return GenerationManifest(**json.load(f))


def write_manifest(directory: str, manifest: GenerationManifest) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(manifest_path(directory), "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def check_consistency(existing: GenerationManifest, incoming: GenerationManifest) -> None:
    """Raise ValueError if any locked field differs."""
    for name in _LOCKED_FIELDS:
        ev = getattr(existing, name)
        iv = getattr(incoming, name)
        if ev != iv:
            raise ValueError(
                f"Manifest mismatch on '{name}': existing={ev!r} vs incoming={iv!r}. "
                f"Pass --force-regenerate or use a different output directory."
            )

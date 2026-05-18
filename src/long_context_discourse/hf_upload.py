"""Helpers for pushing the local results / dataset folders to the Hugging Face Hub.

We expose two entry points:

* :func:`upload_file` — single file uploads (small JSON results).
* :func:`upload_folder_resilient` — wraps :class:`HfApi.upload_large_folder`
  with batched commits to avoid the ``httpx.ReadTimeout`` you get from a
  single huge ``create_commit`` on a multi-GB tree.

Both bump ``HF_HUB_DOWNLOAD_TIMEOUT`` before any ``huggingface_hub`` import
in the CLI scripts; here we re-apply the value defensively in case this
module is imported first.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1800")

from huggingface_hub import HfApi  # noqa: E402  (imports after env tweak)

DEFAULT_IGNORE_PATTERNS: tuple[str, ...] = (
    "**/.git/**",
    "**/.git",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
)


def _api(token: str | None) -> HfApi:
    return HfApi(token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def upload_file(
    *,
    local_path: str | Path,
    path_in_repo: str,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = False,
    commit_message: str | None = None,
    token: str | None = None,
) -> str:
    """Upload one file (or pre-serialised payload) to a HF dataset/model repo."""
    api = _api(token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    info = api.upload_file(
        path_or_fileobj=str(Path(local_path).expanduser().resolve()),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Upload {path_in_repo}",
    )
    return info.commit_url if hasattr(info, "commit_url") else str(info)


def upload_folder_resilient(
    *,
    folder_path: str | Path,
    repo_id: str,
    repo_type: str = "dataset",
    private: bool = False,
    ignore_patterns: Iterable[str] = DEFAULT_IGNORE_PATTERNS,
    num_workers: int | None = None,
    token: str | None = None,
) -> None:
    """Upload an entire folder using the resumable large-folder API."""
    api = _api(token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "folder_path": str(Path(folder_path).expanduser().resolve()),
        "repo_type": repo_type,
        "private": private,
        "ignore_patterns": list(ignore_patterns) if ignore_patterns else None,
    }
    if num_workers is not None:
        kwargs["num_workers"] = num_workers
    api.upload_large_folder(**kwargs)

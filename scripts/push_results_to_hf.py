#!/usr/bin/env python3
"""Push the local results/ tree (or any folder) to a Hugging Face dataset repo.

Wraps :func:`hf_upload.upload_folder_resilient`, which uses
``HfApi.upload_large_folder`` (batched, resumable commits) to avoid the
``httpx.ReadTimeout`` you get from a single huge ``create_commit``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from long_context_discourse.hf_upload import upload_folder_resilient
from long_context_discourse.logging_utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Local folder to upload")
    parser.add_argument("--repo-id", required=True, help="username/dataset-name")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", default=None, help="Override HF_TOKEN env var")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    upload_folder_resilient(
        folder_path=Path(args.folder),
        repo_id=args.repo_id,
        private=args.private,
        token=args.token,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

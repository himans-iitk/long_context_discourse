#!/usr/bin/env python3
"""Run ``run_exp*`` across generated shard configs with a process pool.

Each subprocess executes the normal CLI entrypoint so logging and retries match
serial runs.

With ``--resume`` (default), skips a shard when its main JSON output exists and
the row count matches the expected size for that shard config (cheap restarts).

Usage:
  python scripts/run_parallel_stage.py --project-root . --stage exp1 --max-workers 8
  python scripts/run_parallel_stage.py --project-root . --stage exp2_pair --no-resume
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Support ``python scripts/run_parallel_stage.py`` when this interpreter does not
# have an editable install (e.g. conda ``python`` instead of ``.venv/bin/python``).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from long_context_discourse.shard_resume import plan_exp2_pair, skip_singleton_shard


def _shard_configs(shards_dir: Path, prefix: str) -> list[Path]:
    # Generator emits expN.shard_<model>.yaml; legacy runs used expN.shard.<model>.yaml
    paths = sorted(shards_dir.glob(f"{prefix}_*.yaml"))
    if not paths:
        paths = sorted(shards_dir.glob(f"{prefix}.*.yaml"))
    if not paths:
        raise FileNotFoundError(
            f"No shard configs matching {prefix}_*.yaml or {prefix}.*.yaml under {shards_dir}"
        )
    return paths


def _run_cmd(
    exe: str,
    project_root: Path,
    env_file: Path | None,
    argv: list[str],
) -> None:
    cwd = str(project_root)
    cmd = [exe, "-m", "long_context_discourse.scripts_entry", *argv]
    if env_file is not None:
        cmd.extend(["--env", str(env_file)])
    env = os.environ.copy()
    src = project_root.resolve() / "src"
    if (src / "long_context_discourse").is_dir():
        sep = os.pathsep
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) if not prev else f"{src}{sep}{prev}"
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def _worker_exp1(cfg: Path, exe: str, root: Path, env_file: Path | None, *, resume: bool) -> str:
    if skip_singleton_shard(cfg, resume=resume, stage="exp1"):
        print(f"  skip (complete): {cfg.stem}")
        return cfg.stem
    _run_cmd(exe, root, env_file, ["run_exp1", "--config", str(cfg)])
    return cfg.stem


def _worker_exp2_pair(cfg: Path, exe: str, root: Path, env_file: Path | None, *, resume: bool) -> str:
    plan = plan_exp2_pair(cfg, resume=resume)
    if plan == "skip":
        print(f"  skip (complete): {cfg.stem}")
        return cfg.stem
    if plan == "exp2b_only":
        _run_cmd(exe, root, env_file, ["run_exp2b", "--config", str(cfg)])
        return cfg.stem
    _run_cmd(exe, root, env_file, ["run_exp2", "--config", str(cfg)])
    _run_cmd(exe, root, env_file, ["run_exp2b", "--config", str(cfg)])
    return cfg.stem


def _worker_exp4(cfg: Path, exe: str, root: Path, env_file: Path | None, *, resume: bool) -> str:
    if skip_singleton_shard(cfg, resume=resume, stage="exp4"):
        print(f"  skip (complete): {cfg.stem}")
        return cfg.stem
    _run_cmd(exe, root, env_file, ["run_exp4", "--config", str(cfg)])
    return cfg.stem


def _worker_exp5(cfg: Path, exe: str, root: Path, env_file: Path | None, *, resume: bool) -> str:
    if skip_singleton_shard(cfg, resume=resume, stage="exp5"):
        print(f"  skip (complete): {cfg.stem}")
        return cfg.stem
    _run_cmd(exe, root, env_file, ["run_exp5", "--config", str(cfg)])
    return cfg.stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel shard runner for API experiments.")
    ap.add_argument("--project-root", type=Path, default=Path("."))
    ap.add_argument(
        "--shards-dir",
        type=Path,
        default=None,
        help="Defaults to <project-root>/configs/parallel/shards",
    )
    ap.add_argument(
        "--stage",
        choices=("exp1", "exp2_pair", "exp4", "exp5"),
        required=True,
    )
    ap.add_argument("--max-workers", type=int, default=11)
    ap.add_argument("--env-file", type=Path, default=None, help="Passed as --env to each run")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run all shards even when outputs already match expected row counts",
    )
    args = ap.parse_args()

    root = args.project_root.resolve()
    shards_dir = args.shards_dir
    if shards_dir is None:
        shards_dir = root / "configs" / "parallel" / "shards"
    else:
        shards_dir = shards_dir.resolve()

    exe = sys.executable
    env_path = args.env_file.resolve() if args.env_file else None
    resume = not args.no_resume

    prefix_map = {
        "exp1": "exp1.shard",
        "exp2_pair": "exp2.shard",
        "exp4": "exp4.shard",
        "exp5": "exp5.shard",
    }
    worker_map = {
        "exp1": _worker_exp1,
        "exp2_pair": _worker_exp2_pair,
        "exp4": _worker_exp4,
        "exp5": _worker_exp5,
    }
    prefix = prefix_map[args.stage]
    worker = worker_map[args.stage]
    configs = _shard_configs(shards_dir, prefix)

    max_workers = max(1, min(args.max_workers, len(configs)))
    print(
        f"Running {args.stage}: {len(configs)} shards with max_workers={max_workers} "
        f"(resume={'on' if resume else 'off'})"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(worker, cfg, exe, root, env_path, resume=resume) for cfg in configs
        ]
        for fut in as_completed(futures):
            name = fut.result()
            print(f"  done: {name}")


if __name__ == "__main__":
    main()

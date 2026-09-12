from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = CODE_DIR.parent
MAIN_PIPELINE = CODE_DIR / "scripts" / "main_pipeline.py"

# Shared grounding stores (seed CSVs + structured/raw online cache). They live under data/ (NOT
# per-preset RESULTS_DIR), so every preset reads the same POI groundings. But online grounding
# upserts new POIs into them mid-run, so the preset that runs first mutates what later presets
# see. To keep grounding IDENTICAL at each preset's start, we snapshot these before the batch and
# restore them before every preset (see --freeze-grounding).
GROUNDING_DIRS = [
    PROJECT_DIR / "data" / "grounding_seed",
    PROJECT_DIR / "data" / "cache" / "online_grounding",
]


def snapshot_grounding(dest: Path) -> list:
    """Copy the shared grounding stores into dest/<name>; return the ones that existed."""
    saved = []
    for d in GROUNDING_DIRS:
        if d.exists():
            shutil.copytree(d, dest / d.name, dirs_exist_ok=True)
            saved.append(d)
    return saved


def restore_grounding(src: Path) -> None:
    """Restore the shared grounding stores from a snapshot dir (overwrite current)."""
    for d in GROUNDING_DIRS:
        snap = src / d.name
        if snap.exists():
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(snap, d)


# Full first (baseline), then the single-factor ablations. Names must match
# main_pipeline.ABLATION_PRESETS; an unknown name makes the subprocess fail loudly.
ALL_PRESETS = ["full", "no_negotiation", "no_split", "no_grounding", "no_quality_control"]

# Args forwarded verbatim to main_pipeline (only those the user set on the command line).
_FORWARDED = ["mode", "first_n", "families", "start_from", "max_parallel"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="一次性跑完全部消融实验，每个预设存到独立文件夹。",)
    ap.add_argument(
        "--presets", default=",".join(ALL_PRESETS), help=f"逗号分隔的预设列表（默认全部）：{','.join(ALL_PRESETS)}",
    )
    ap.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "results" / "ablation"),
        help="各预设结果的根目录（默认 <project>/results/ablation）。每个预设写入 <root>/<preset>/。",
    )
    ap.add_argument(
        "--timestamp-root", action="store_true", help="在 output-root 下再套一层时间戳目录，避免覆盖历史运行。",
    )
    ap.add_argument(
        "--freeze-grounding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认开：第一个预设(full)跑完后冻结其 grounding(data/grounding_seed + data/cache/online_grounding)，"
        "后续每个消融预设开跑前还原它——所有消融都对着 full 产生的同一份 POI 落地评测(只差被消融的组件)。"
        "用 --no-freeze-grounding 关闭。",
    )
    ap.add_argument("--continue-on-error", action="store_true", help="某预设失败(rc!=0)后继续跑后续预设。")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令与目录，不实际运行。")
    # ---- forwarded to main_pipeline (only passed on if the user set them) ----
    ap.add_argument("--mode", choices=["all", "families", "first_n", "failed_only"], default=None)
    ap.add_argument("--first-n", type=int, default=None, help="mode=first_n 时取前 N 个家庭")
    ap.add_argument("--families", default=None, help="mode=families 时的 family_id 列表（逗号分隔）")
    ap.add_argument("--start-from", type=int, default=None, help=">0：只保留 family_id>=此值")
    ap.add_argument("--max-parallel", type=int, default=None, help="每个预设内并发运行的家庭数")
    return ap.parse_args()


def forwarded_args(args: argparse.Namespace) -> list[str]:
    """Build the main_pipeline CLI args, forwarding only options the user actually set."""
    out: list[str] = []
    for name in _FORWARDED:
        val = getattr(args, name)
        if val is None:
            continue
        out += [f"--{name.replace('_', '-')}", str(val)]
    return out


def run_one(preset: str, out_dir: Path, pass_args: list[str], args: argparse.Namespace) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["ABLATION_PRESET"] = preset
    env["FAMILY_MATRIP_RESULTS_DIR"] = str(out_dir)
    env["PYTHONIOENCODING"] = "utf-8"  # child emits UTF-8 (Chinese text + box chars)
    cmd = [sys.executable, str(MAIN_PIPELINE), *pass_args]
    log_path = out_dir / "run_all.log"

    header = (
        f"\n{'=' * 78}\n[run_all_ablations] preset={preset}\n"
        f"  results_dir = {out_dir}\n  cmd         = {' '.join(cmd)}\n{'=' * 78}"
    )
    print(header, flush=True)
    if args.dry_run:
        return {"preset": preset, "rc": 0, "seconds": 0.0, "dir": str(out_dir), "dry_run": True}

    t0 = time.time()
    # Stream the subprocess output to console AND the per-preset log file.
    with log_path.open("w", encoding="utf-8") as log:
        log.write(header + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(CODE_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    dt = time.time() - t0
    print(f"[run_all_ablations] preset={preset} 完成，rc={rc}，用时 {dt/60:.1f} 分钟\n", flush=True)
    return {"preset": preset, "rc": rc, "seconds": dt, "dir": str(out_dir), "dry_run": False}


def main() -> int:
    try:  # console may be gbk on zh-CN Windows; force UTF-8 so box chars / 中文 don't crash writes
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = parse_args()
    presets = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in presets if p not in ALL_PRESETS]
    if unknown:
        print(f"[run_all_ablations] 警告：未在已知列表里的预设 {unknown}（仍会尝试，交给 main_pipeline 校验）。")

    root = Path(args.output_root)
    if args.timestamp_root:
        root = root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    root.mkdir(parents=True, exist_ok=True)

    pass_args = forwarded_args(args)
    print(
        f"[run_all_ablations] 预设 {presets}\n  输出根目录: {root}\n  透传 main_pipeline 参数: {pass_args or '(默认 ORCH_*)'}"
    )

    # Grounding freeze (default ON): keep grounding IDENTICAL across all presets.
    #  - If `full` runs first: run it against the current stores, then snapshot the grounding it
    #    produced and restore that before every later preset (ablations reuse FULL's grounding).
    #  - If `full` is NOT first (e.g. excluded): snapshot the CURRENT on-disk grounding (typically a
    #    prior full run's) BEFORE the batch and restore it before EVERY preset.
    # Safe for no_grounding: its grounding-application steps are gated off by ONLINE_GROUNDING_ENABLED,
    # so a full seed can't leak concrete POIs into it.
    freeze = args.freeze_grounding and not args.dry_run
    grounding_snap = root / "_grounding_snapshot" if freeze else None
    prime_from_full = freeze and presets and presets[0] == "full"

    def _save_snap():
        if grounding_snap.exists():
            shutil.rmtree(grounding_snap)
        grounding_snap.mkdir(parents=True, exist_ok=True)
        return snapshot_grounding(grounding_snap)

    if freeze and not prime_from_full:
        saved = _save_snap()
        print(
            f"[run_all_ablations] full 不在首位——已冻结当前磁盘上的 grounding(全部消融复用): "
            f"{[p.name for p in saved]}"
        )

    results = []
    for i, preset in enumerate(presets):
        if grounding_snap is not None and grounding_snap.exists():
            restore_grounding(grounding_snap)
            print(f"[run_all_ablations] preset={preset}: 已还原 grounding 冻结快照")
        res = run_one(preset, root / preset, pass_args, args)
        results.append(res)
        if prime_from_full and i == 0 and res["rc"] == 0:
            saved = _save_snap()
            print(f"[run_all_ablations] 已冻结 'full' 的 grounding(后续消融复用): {[p.name for p in saved]}")
        if res["rc"] != 0 and not args.continue_on_error and not args.dry_run:
            print(
                f"[run_all_ablations] preset={preset} 失败(rc={res['rc']})，停止。用 --continue-on-error 可跳过继续。"
            )
            break

    # summary
    lines = ["", "=" * 78, "消融批量运行汇总", "=" * 78]
    lines.append(f"{'preset':<20} {'rc':>4} {'minutes':>9}   results_dir")
    for r in results:
        lines.append(f"{r['preset']:<20} {r['rc']:>4} {r['seconds']/60:>9.1f}   {r['dir']}")
    summary = "\n".join(lines)
    print(summary, flush=True)
    (root / "run_all_summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(f"\n汇总已写入: {root / 'run_all_summary.txt'}")

    return 0 if all(r["rc"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

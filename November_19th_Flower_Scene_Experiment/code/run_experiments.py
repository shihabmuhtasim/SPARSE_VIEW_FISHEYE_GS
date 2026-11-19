#!/usr/bin/env python3
"""Run the fixed 12-experiment study sequentially on a configurable Linux GPU host.

This portable launcher preserves the executed model and selector settings.
Completed results are hash-checked. Interrupted optimization restarts from seed
zero only with --restart-incomplete; no exact optimizer/EMA/RNG resume is claimed.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import traceback
from preflight import METHODS, COUNTS, PACKAGE, sha, validate, gpu_loader

GiB = 1024 ** 3


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def command(source, scene, output, manifest):
    return [sys.executable, "-u", str(source / "train_eval.py"), "-s", str(scene),
            "-m", str(output / "model"), "--eval", "--split_manifest", str(manifest),
            "--bs", "2", "--cvo_preset", "recip", "--iterations", "30000", "--seed", "0",
            "--disable_viewer", "--test_iterations", "7000", "30000", "--save_iterations", "30000",
            "--checkpoint_iterations", "10000", "20000", "25000"]


def inventory(run):
    return {str(p.relative_to(run)): sha(p) for p in sorted(run.rglob("*")) if p.is_file()
            and not any(x == "train" or x == "__pycache__" or x.startswith("chkpnt") for x in p.relative_to(run).parts)
            and p.name not in {"done.json", "completed_sha256.json", "temporary_cleanup.json"}}


def verify_finished(run):
    if not (run / "done.json").exists():
        return False
    hashes = json.loads((run / "completed_sha256.json").read_text())
    assert hashes, "Empty completion inventory"
    for name, expected in hashes.items():
        assert sha(run / name) == expected, "Completed result changed: " + str(run / name)
    return True


def preserve_partial(run):
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"_{time.time_ns() % 1000000:06d}"
    target = run.parent.parent / "interrupted_attempts" / run.name / stamp
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run), target)
    dump(target / "recovery.json", {"action": "Restart this unfinished run from seed 0",
         "reason": "Legacy checkpoints omit full RNG, CVO and EMA recovery state", "time": time.time()})
    run.mkdir()
    return target


def metrics_complete(model, tests):
    import math
    folder = model / "test/ours_30000"
    assert json.loads((folder / "view_order.json").read_text()) == sorted(tests)
    per = json.loads((model / "per_view_test.json").read_text())["ours_30000"]
    for split in ["train", "test"]:
        mean = json.loads((model / f"results_{split}.json").read_text())["ours_30000"]
        assert all(math.isfinite(mean[m]) for m in ["PSNR", "SSIM", "LPIPS"])
    mean = json.loads((model / "results_test.json").read_text())["ours_30000"]
    for metric in ["PSNR", "SSIM", "LPIPS"]:
        values = per[metric]
        assert set(values) == {f"{i:05d}.png" for i in range(len(tests))}
        assert all(math.isfinite(x) for x in values.values())
        assert abs(sum(values.values()) / len(tests) - mean[metric]) < 1e-6
    for i in range(len(tests)):
        assert all((folder / sub / f"{i:05d}.png").is_file() for sub in ["gt", "renders"])
    assert (model / "point_cloud/iteration_30000/point_cloud.ply").is_file()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["source", "scene-dir", "manifests", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Second location for verified final outputs")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--quota-root", type=Path)
    parser.add_argument("--quota-gib", type=float, default=45)
    parser.add_argument("--min-free-gib", type=float, default=20)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--prune-temporary", action="store_true", help="Remove checkpoints and train PNGs only after verified second copy")
    parser.add_argument("--validate-only", action="store_true", help="Validate CPU inputs and save exact commands; do not use the GPU")
    args = parser.parse_args()
    for name in ["source", "scene_dir", "manifests", "output", "archive", "quota_root"]:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.prune_temporary and args.archive is None:
        parser.error("--prune-temporary requires --archive for a verified second copy")
    if args.gpu < 0 or args.min_free_gib < 0 or args.quota_gib <= 0:
        parser.error("Invalid GPU index or storage limit")
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    # Reject overlapping result roots: archive/copy/prune must never recurse into themselves.
    if args.archive and (args.archive == root or args.archive in root.parents or root in args.archive.parents):
        raise ValueError("Output and archive must be separate, non-nested folders")
    lock = (root / "runner.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = validate(args.source, args.scene_dir, args.manifests)
    input_hashes = {name: sha(args.scene_dir / "images" / name)
                    for name in sorted(base["train_images"] + base["test_images"])}
    if (root / "input_images_sha256.json").exists():
        assert json.loads((root / "input_images_sha256.json").read_text()) == input_hashes, "An input photograph changed"
    else:
        dump(root / "input_images_sha256.json", input_hashes)
    config = {"scene": base["scene"], "source": str(args.source), "scene_dir": str(args.scene_dir),
              "base_sha256": sha(next(args.manifests.glob("*_fullview_split_v1.json"))),
              "manifest_sha256": {p.name: sha(p) for p in args.manifests.glob("*.json")},
              "methods": METHODS, "counts": COUNTS, "iterations": 30000, "seed": 0,
              "test_images": base["test_images"], "archive": str(args.archive) if args.archive else None}
    if (root / "config.json").exists():
        assert json.loads((root / "config.json").read_text()) == config, "Existing output belongs to different inputs/settings"
    else:
        dump(root / "config.json", config)
    control = root / "control"
    shutil.copytree(args.manifests, control / "manifests", dirs_exist_ok=True)
    plan = []
    for method in METHODS:
        for count in COUNTS:
            manifest = control / f"manifests/{base['scene']}_{method}_{count}_v1.json"
            plan.append({"run": f"{method}_{count}", "argv": command(args.source, args.scene_dir, root / "runs" / f"{method}_{count}", manifest)})
    dump(root / "run_plan.json", plan)
    if args.validate_only:
        print("PASS: all 12 inputs and commands validated; no training started.")
        return
    os.umask(0o077)
    env = dict(os.environ)
    for key in list(env):
        if key.startswith("DFGS_"):
            env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1", PYTHONPATH=str(args.source),
               TMPDIR=str(root / "cache/tmp"), TORCH_HOME=str(root / "cache/torch"),
               XDG_CACHE_HOME=str(root / "cache"), MPLCONFIGDIR=str(root / "cache/mpl"),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for sub in ["tmp", "torch", "mpl"]:
        (root / "cache" / sub).mkdir(parents=True, exist_ok=True)
    complete = []
    child = None

    def status(state, **data):
        record = {"state": state, "host": socket.gethostname(), "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  "completed_runs": complete, "total_runs": len(plan), **data}
        dump(root / "status.json", record)
        print(json.dumps(record), flush=True)

    def space():
        assert shutil.disk_usage(root).free > args.min_free_gib * GiB, "Output disk free-space guard"
        if args.archive:
            args.archive.mkdir(parents=True, exist_ok=True)
            assert shutil.disk_usage(args.archive).free > 3 * GiB, "Archive needs at least 3 GiB free"
        if args.quota_root:
            used = int(subprocess.check_output(["du", "-s", "-B1", str(args.quota_root)], text=True).split()[0])
            assert used + 3 * GiB < args.quota_gib * GiB, "Work quota guard: keep 3 GiB headroom"

    def gpu_wait():
        clear = 0
        while clear < 3:
            jobs = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            clear = clear + 1 if not jobs else 0
            if jobs:
                status("waiting_for_gpu", other_compute_pids=jobs)
            if clear < 3:
                time.sleep(5)

    def postprocess(method=None, count=None, prepare=False):
        cmd = [sys.executable, str(PACKAGE / "code/postprocess.py"), "--root", str(root)]
        if prepare:
            cmd += ["--prepare-plots"]
        if method:
            cmd += ["--method", method, "--count", str(count)]
        path = root / (f"postprocess_{method}_{count}.log" if method else "summary.log")
        with path.open("a") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    try:
        status("preflight")
        space()
        gpu_wait()
        with (root / "preflight.log").open("w") as log:
            cmd = [sys.executable, str(PACKAGE / "code/preflight.py"), "--source", str(args.source),
                   "--scene-dir", str(args.scene_dir), "--manifests", str(control / "manifests"), "--gpu", str(args.gpu), "--with-gpu"]
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        (root / "environment_packages.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        postprocess(prepare=True)
        for item in plan:
            name = item["run"]
            method, count_text = name.rsplit("_", 1)
            count = int(count_text)
            run = root / "runs" / name
            run.mkdir(parents=True, exist_ok=True)
            if verify_finished(run):
                if args.archive:
                    assert verify_finished(args.archive / "runs" / name), "Second copy failed verification"
                complete.append(name)
                status("completed_run_preserved", run=name)
                continue
            if (run / "model").exists() and not (run / "training_succeeded.json").exists():
                # A completed evaluation can be recovered without retraining.
                try:
                    metrics_complete(run / "model", base["test_images"])
                except (OSError, ValueError, KeyError, AssertionError):
                    if not args.restart_incomplete:
                        raise RuntimeError("Interrupted output found; rerun with --restart-incomplete to preserve it and restart only " + name)
                    preserved = preserve_partial(run)
                    status("restarting_unfinished_run", run=name, preserved_attempt=str(preserved))
                else:
                    dump(run / "training_succeeded.json", {"recovered_complete_metrics": True, "time": time.time()})
                    if not (run / "timing.json").exists():
                        dump(run / "timing.json", {"train_render_evaluate_seconds": None, "note": "Timing unavailable after interrupted completion marker"})
            if not (run / "training_succeeded.json").exists():
                space()
                gpu_wait()
                dump(run / "command.json", {"argv": item["argv"], "source_hashes": json.loads((PACKAGE / "provenance/training_source_sha256.json").read_text())})
                started = time.time()
                progress_time = started
                last = -1
                with (run / "train_eval.log").open("w") as log:
                    child = subprocess.Popen(item["argv"], cwd=args.source, env=env, stdout=log, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        with (run / "train_eval.log").open("rb") as stream:
                            stream.seek(max(0, stream.seek(0, 2) - 16000))
                            text = stream.read().decode(errors="replace")
                        matches = re.findall(r"(\d+)/30000", text)
                        iteration = int(matches[-1]) if matches else max(last, 0)
                        if iteration != last:
                            progress_time, last = time.time(), iteration
                        gpu = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
                        status("rendering_and_evaluating" if iteration >= 30000 else "training", run=name,
                               method=method, training_views=count, iteration=iteration,
                               progress_percent=round(iteration / 300, 2), pid=child.pid,
                               gpu_utilization_and_memory=gpu, elapsed_seconds=round(time.time() - started, 1))
                        if time.time() - progress_time > 1200:
                            raise RuntimeError("No iteration progress for 20 minutes: " + name)
                        time.sleep(20)
                if child.returncode:
                    raise RuntimeError(f"{name} exited {child.returncode}; see train_eval.log")
                metrics_complete(run / "model", base["test_images"])
                dump(run / "timing.json", {"train_render_evaluate_seconds": time.time() - started})
                dump(run / "training_succeeded.json", {"state": "complete", "iteration": 30000, "time": time.time()})
            status("postprocessing", run=name)
            postprocess(method, count)
            hashes = inventory(run)
            dump(run / "completed_sha256.json", hashes)
            if args.archive:
                space()
                dest = args.archive / "runs" / name
                shutil.copytree(run, dest, dirs_exist_ok=True, ignore=shutil.ignore_patterns("chkpnt*.pth", "train", "__pycache__"))
                for filename, expected in hashes.items():
                    assert sha(dest / filename) == expected, "Archive copy mismatch: " + filename
                dump(dest / "done.json", {"state": "complete", "verified_files": len(hashes), "time": time.time()})
            dump(run / "done.json", {"state": "complete", "verified_files": len(hashes), "time": time.time()})
            if args.prune_temporary:
                for path in (run / "model").glob("chkpnt*.pth"):
                    path.unlink()
                if (run / "model/train").exists():
                    shutil.rmtree(run / "model/train")
            complete.append(name)
            postprocess()
            if args.archive:
                for sub in ["summary", "plots", "control"]:
                    shutil.copytree(root / sub, args.archive / sub, dirs_exist_ok=True)
                for filename in ["config.json", "run_plan.json", "input_images_sha256.json", "environment_packages.txt", "preflight.log"]:
                    shutil.copy2(root / filename, args.archive / filename)
        status("complete", message="ALL 12 EXPERIMENTS HAVE BEEN COMPLETED")
        if args.archive:
            shutil.copy2(root / "status.json", args.archive / "status.json")
    except BaseException:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        status("failed", error=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()

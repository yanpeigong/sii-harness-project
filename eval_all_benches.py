import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm_client import count_messages_tokens, count_tokens
from run import load_jsonl, make_controlled_llm
from solution import MyHarness


@dataclass(frozen=True)
class EvalTask:
    group: str
    name: str
    train_path: Path
    dev_path: Path


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_tasks(root: Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []

    data_train = root / "data" / "train_dev.jsonl"
    data_dev = root / "data" / "test_dev.jsonl"
    if data_train.exists() and data_dev.exists():
        tasks.append(EvalTask("data", "data/dev", data_train, data_dev))

    for bench_idx in range(1, 5):
        bench_root = root / f"bench{bench_idx}"
        if not bench_root.exists():
            continue
        for train_path in sorted(bench_root.rglob("train*.jsonl")):
            suffix = train_path.stem[len("train") :]
            dev_path = train_path.with_name(f"test{suffix}.jsonl")
            if not dev_path.exists():
                continue
            task_dir = train_path.parent.relative_to(root).as_posix()
            if train_path.stem != "train":
                task_name = f"{task_dir}/{train_path.stem}"
            else:
                task_name = task_dir
            tasks.append(EvalTask(f"bench{bench_idx}", task_name, train_path, dev_path))

    return tasks


def evaluate_task(task: EvalTask, runs: int, workers: int, max_prompt_tokens: int):
    train = load_jsonl(str(task.train_path))
    dev = load_jsonl(str(task.dev_path))

    task_result = {
        "group": task.group,
        "task": task.name,
        "train_path": str(task.train_path),
        "dev_path": str(task.dev_path),
        "train_n": len(train),
        "dev_n": len(dev),
        "runs": [],
    }

    for run_idx in range(runs):
        tracker = {"prompt": 0, "completion": 0}
        lock = threading.Lock()
        llm = make_controlled_llm(max_prompt_tokens, tracker, lock)

        harness = MyHarness(llm, count_tokens, count_messages_tokens, max_prompt_tokens)
        for item in train:
            harness.update(item["text"], item["label"])

        predictions = [None] * len(dev)
        error_log = []
        t0 = time.time()

        def run_one(args_):
            idx, item = args_
            try:
                pred = harness.predict(item["text"])
                return idx, pred.strip(), None
            except Exception as exc:
                return idx, "", f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one, (idx, item)) for idx, item in enumerate(dev)]
            done = 0
            for future in as_completed(futures):
                idx, pred, err = future.result()
                predictions[idx] = pred
                if err:
                    error_log.append({"idx": idx, "error": err})
                done += 1
                sys.stdout.write(
                    f"\r    run {run_idx + 1}/{runs}: {done}/{len(dev)}"
                )
                sys.stdout.flush()
        print()

        elapsed = time.time() - t0
        correct = sum(1 for item, pred in zip(dev, predictions) if pred == item["label"])

        per_label = {}
        samples = []
        for idx, (item, pred) in enumerate(zip(dev, predictions)):
            gold = item["label"]
            is_correct = pred == gold
            stats = per_label.setdefault(gold, {"total": 0, "correct": 0})
            stats["total"] += 1
            stats["correct"] += int(is_correct)
            samples.append(
                {
                    "idx": idx,
                    "text": item["text"],
                    "gold": gold,
                    "pred": pred,
                    "correct": is_correct,
                }
            )

        for stats in per_label.values():
            stats["accuracy"] = stats["correct"] / stats["total"] * 100.0

        task_result["runs"].append(
            {
                "run": run_idx + 1,
                "correct": correct,
                "total": len(dev),
                "accuracy": correct / len(dev) * 100.0 if dev else 0.0,
                "elapsed_seconds": elapsed,
                "prompt_tokens": tracker["prompt"],
                "completion_tokens": tracker["completion"],
                "errors": error_log,
                "per_label": dict(sorted(per_label.items())),
                "samples": samples,
            }
        )

    return task_result


def summarize_results(results: list[dict]):
    summary_rows = []
    label_rows = []
    prediction_rows = []
    group_totals = {}

    grand_correct = 0
    grand_total = 0
    task_avg_accs = []

    for result in results:
        runs = result["runs"]
        total_correct = sum(run["correct"] for run in runs)
        total_items = sum(run["total"] for run in runs)
        accuracies = [run["accuracy"] for run in runs]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        task_avg_accs.append(avg_acc)
        grand_correct += total_correct
        grand_total += total_items

        group = result["group"]
        group_totals.setdefault(group, {"correct": 0, "total": 0, "task_accs": []})
        group_totals[group]["correct"] += total_correct
        group_totals[group]["total"] += total_items
        group_totals[group]["task_accs"].append(avg_acc)

        prompt_per_item = (
            sum(run["prompt_tokens"] for run in runs) / total_items if total_items else 0.0
        )
        completion_per_item = (
            sum(run["completion_tokens"] for run in runs) / total_items
            if total_items
            else 0.0
        )

        summary_rows.append(
            {
                "group": result["group"],
                "task": result["task"],
                "train_path": result["train_path"],
                "dev_path": result["dev_path"],
                "train_n": result["train_n"],
                "dev_n": result["dev_n"],
                "runs": len(runs),
                "run_accuracies": ";".join(f"{acc:.4f}" for acc in accuracies),
                "avg_accuracy": f"{avg_acc:.4f}",
                "total_correct": total_correct,
                "total_items": total_items,
                "prompt_tokens_per_item": f"{prompt_per_item:.2f}",
                "completion_tokens_per_item": f"{completion_per_item:.2f}",
                "elapsed_seconds_total": f"{sum(run['elapsed_seconds'] for run in runs):.2f}",
                "error_count": sum(len(run["errors"]) for run in runs),
            }
        )

        for run in runs:
            for label, stats in run["per_label"].items():
                label_rows.append(
                    {
                        "group": result["group"],
                        "task": result["task"],
                        "run": run["run"],
                        "label": label,
                        "correct": stats["correct"],
                        "total": stats["total"],
                        "accuracy": f"{stats['accuracy']:.4f}",
                    }
                )
            for sample in run["samples"]:
                prediction_rows.append(
                    {
                        "group": result["group"],
                        "task": result["task"],
                        "run": run["run"],
                        "idx": sample["idx"],
                        "gold": sample["gold"],
                        "pred": sample["pred"],
                        "correct": sample["correct"],
                        "text": sample["text"],
                    }
                )

    group_summary = {}
    for group, stats in sorted(group_totals.items()):
        micro = stats["correct"] / stats["total"] * 100.0 if stats["total"] else 0.0
        macro = (
            sum(stats["task_accs"]) / len(stats["task_accs"])
            if stats["task_accs"]
            else 0.0
        )
        group_summary[group] = {
            "micro_accuracy": micro,
            "macro_task_accuracy": macro,
            "correct": stats["correct"],
            "total": stats["total"],
            "tasks": len(stats["task_accs"]),
        }

    overall = {
        "micro_accuracy": grand_correct / grand_total * 100.0 if grand_total else 0.0,
        "macro_task_accuracy": (
            sum(task_avg_accs) / len(task_avg_accs) if task_avg_accs else 0.0
        ),
        "correct": grand_correct,
        "total": grand_total,
        "tasks": len(results),
    }

    return summary_rows, label_rows, prediction_rows, group_summary, overall


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MyHarness on data and all bench1-bench4 tasks."
    )
    parser.add_argument("--workers", type=int, default=25)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only print discovered tasks; do not call the LLM.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    tasks = discover_tasks(root)
    if not tasks:
        raise SystemExit("No train/test jsonl pairs found.")

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")

    print(f"Discovered {len(tasks)} tasks.")
    print(
        f"Settings: workers={args.workers}, runs={args.runs}, "
        f"max_prompt_tokens={args.max_prompt_tokens}"
    )
    if args.list_only:
        for idx, task in enumerate(tasks, start=1):
            print(
                f"  [{idx}] {task.name}: "
                f"{_rel(task.train_path, root)} -> {_rel(task.dev_path, root)}"
            )
        return

    results = []
    started_at = datetime.now().astimezone()
    for idx, task in enumerate(tasks, start=1):
        print(
            f"\n[{idx}/{len(tasks)}] {task.name} "
            f"({_rel(task.train_path, root)} -> {_rel(task.dev_path, root)})"
        )
        result = evaluate_task(task, args.runs, args.workers, args.max_prompt_tokens)
        results.append(result)
        accuracies = [run["accuracy"] for run in result["runs"]]
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        print(
            f"    avg={avg_acc:.2f}% "
            f"runs={', '.join(f'{acc:.2f}%' for acc in accuracies)}"
        )

    summary_rows, label_rows, prediction_rows, group_summary, overall = summarize_results(
        results
    )

    finished_at = datetime.now().astimezone()
    stamp = finished_at.strftime("%Y%m%d_%H%M%S")
    results_dir = root / args.results_dir
    base = results_dir / f"eval_all_{stamp}"

    payload = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "settings": {
            "workers": args.workers,
            "runs": args.runs,
            "max_prompt_tokens": args.max_prompt_tokens,
        },
        "overall": overall,
        "groups": group_summary,
        "tasks": results,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (base.with_suffix(".json")).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(base.with_suffix(".summary.csv"), summary_rows)
    write_csv(base.with_suffix(".labels.csv"), label_rows)
    write_csv(base.with_suffix(".predictions.csv"), prediction_rows)

    print("\nGroup averages:")
    for group, stats in group_summary.items():
        print(
            f"  {group}: micro={stats['micro_accuracy']:.2f}% "
            f"macro={stats['macro_task_accuracy']:.2f}% "
            f"({stats['tasks']} tasks)"
        )
    print(
        f"\nOverall: micro={overall['micro_accuracy']:.2f}% "
        f"macro={overall['macro_task_accuracy']:.2f}% "
        f"({overall['tasks']} tasks)"
    )
    print(f"\nWrote: {base.with_suffix('.json')}")
    print(f"Wrote: {base.with_suffix('.summary.csv')}")
    print(f"Wrote: {base.with_suffix('.labels.csv')}")
    print(f"Wrote: {base.with_suffix('.predictions.csv')}")


if __name__ == "__main__":
    main()

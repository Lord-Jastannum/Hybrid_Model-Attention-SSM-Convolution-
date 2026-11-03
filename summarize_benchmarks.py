"""Aggregate multi-seed benchmark results and create comparison plots."""
import argparse, csv, json, statistics
from pathlib import Path
import matplotlib.pyplot as plt

METRICS = [
    "test_perplexity", "test_loss", "parameters",
    "inference_tokens_per_sec", "batch_latency_ms_median",
    "batch_latency_ms_p95", "peak_train_allocated_mb",
    "total_training_seconds", "checkpoint_mb",
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--project-dir",type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    root=args.project_dir/"outputs"/"benchmarks"
    source=root/"benchmark_summary.json"
    if not source.exists():
        raise FileNotFoundError(f"Run run_benchmarks.py first: {source}")
    rows=json.loads(source.read_text(encoding="utf-8"))
    grouped={}
    for row in rows: grouped.setdefault(row["variant"],[]).append(row)
    aggregate=[]
    for variant, items in grouped.items():
        rec={"variant":variant,"runs":len(items),"seeds":", ".join(str(x["seed"]) for x in items)}
        for metric in METRICS:
            vals=[float(x[metric]) for x in items if x.get(metric) is not None]
            rec[metric+"_mean"]=statistics.mean(vals) if vals else None
            rec[metric+"_std"]=statistics.stdev(vals) if len(vals)>1 else (0.0 if vals else None)
        aggregate.append(rec)
    out=root/"benchmark_aggregate.csv"
    with out.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0].keys()))
        w.writeheader(); w.writerows(aggregate)
    (root/"benchmark_aggregate.json").write_text(json.dumps(aggregate,indent=2),encoding="utf-8")

    plots=root/"plots"; plots.mkdir(exist_ok=True)
    def bar(metric, title, ylabel, filename, lower_is_better=False):
        names=[r["variant"] for r in aggregate]
        means=[r[metric+"_mean"] for r in aggregate]
        errs=[r[metric+"_std"] or 0 for r in aggregate]
        fig,ax=plt.subplots(figsize=(10,5))
        ax.bar(names,means,yerr=errs,capsize=4)
        ax.set_title(title); ax.set_ylabel(ylabel)
        ax.tick_params(axis="x",rotation=18)
        ax.grid(axis="y",alpha=.25)
        fig.tight_layout(); fig.savefig(plots/filename,dpi=180); plt.close(fig)
    bar("test_perplexity","Test perplexity by architecture","Perplexity","test_perplexity.png")
    bar("inference_tokens_per_sec","Inference throughput by architecture","Tokens / second","inference_throughput.png")
    bar("parameters","Trainable parameters by architecture","Parameters","parameter_count.png")
    bar("peak_train_allocated_mb","Peak training GPU memory","Allocated MB","peak_gpu_memory.png")
    print(f"Saved aggregate table: {out}")
    print(f"Saved plots: {plots}")

if __name__=="__main__": main()

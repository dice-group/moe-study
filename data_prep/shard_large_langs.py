from concurrent.futures import ThreadPoolExecutor
import subprocess
from pathlib import Path

SRC = Path("/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples")
DST = Path("/scratch/hpc-prf-merlin/project_data/moe_study/fw_samples/samples/samples_sharded")
DST.mkdir(exist_ok=True)

THRESHOLD = 500 * 2**20  # 500 MB

def shard_dir(d):
    size = sum(f.stat().st_size for f in d.glob("*.gz"))
    if size < THRESHOLD:
        return
    print(f"Sharding {d.name} ({size/1e9:.1f} GB)")
    out = DST / d.name
    out.mkdir(exist_ok=True)
    subprocess.run(f"zcat {d}/*.jsonl.gz | split -l 2000000 - {out}/part_", shell=True)
    subprocess.run(f"pigz -p 4 {out}/part_*", shell=True)

dirs = [d for d in SRC.iterdir() if d.is_dir()]
with ThreadPoolExecutor(max_workers=4) as ex:
    ex.map(shard_dir, dirs)

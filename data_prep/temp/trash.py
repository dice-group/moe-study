from datasets import load_from_disk

dataset = load_from_disk("/scratch/hpc-prf-merlin/project_data/moe_study/tokenized/hin_Deva/00000.jsonl/shard_0000/")
print(dataset.column_names)
print(dataset[0])
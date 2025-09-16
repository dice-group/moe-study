import os
import json
from datasets import load_dataset


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data_dir")
    os.makedirs(output_dir, exist_ok=True)

    # --- Stage 1: Multilingual Data from FineWeb-v2 ---
    stage1_config = {"de": 1000, "fr": 1000, "es": 1000, "it": 1000, "pt": 1000}
    lang_map_s1 = {"de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "it": "ita_Latn", "pt": "por_Latn"}
    stage1_output_path = os.path.join(output_dir, "fineweb_stage1.jsonl")

    print("Preparing Stage 1 dataset (Multilingual)...")
    stage1_samples = []
    for lang_code, num_samples in stage1_config.items():
        config_name = lang_map_s1[lang_code]
        print(f"  -> Sampling {num_samples} examples for '{lang_code}' ({config_name})")
        dataset = load_dataset("HuggingFaceFW/fineweb-v2", name=config_name, split="train", streaming=True)

        for i, example in enumerate(dataset):
            if i >= num_samples:
                break
            stage1_samples.append({"text": example["text"], "lang": lang_code, "id": example["id"]})

    with open(stage1_output_path, "w", encoding="utf-8") as f:
        for entry in stage1_samples:
            f.write(json.dumps(entry) + "\n")
    print(f"Stage 1 dataset saved to {stage1_output_path}\n")

    # --- Stage 2: English Data from FineWeb v1 ---
    stage2_samples_needed = 1000
    stage2_output_path = os.path.join(output_dir, "fineweb_stage2.jsonl")

    print(f"Preparing Stage 2 dataset ({stage2_samples_needed} English examples)...")
    dataset_s2 = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

    stage2_samples = []
    for example in dataset_s2:
        if len(stage2_samples) >= stage2_samples_needed:
            break

        dump_id = example.get("dump", "")
        if dump_id.endswith("-en"):
            stage2_samples.append({"text": example["text"], "lang": "en", "id": example["id"]})

    with open(stage2_output_path, "w", encoding="utf-8") as f:
        for entry in stage2_samples:
            f.write(json.dumps(entry) + "\n")
    print(f"Stage 2 dataset saved to {stage2_output_path}\n")

    print("Sampling complete.")


if __name__ == "__main__":
    main()
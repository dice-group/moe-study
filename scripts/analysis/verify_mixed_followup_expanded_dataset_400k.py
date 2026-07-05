from pathlib import Path

from datasets import load_from_disk

from mixed_followup_dataset_utils import dataset_distribution, read_json, strip_zero_counts


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = REPO_ROOT / "outputs/pipeline_runs/moe_study_mixed_followup_expanded_data_rerun_10_langs"
TOKENIZED = RUN_ROOT / "tokenized_expanded_400k_train"
AUDIT_DIR = RUN_ROOT / "tokenized_expanded_400k_train_audit"
EXPECTED_SPLITS = {"train": 400_000, "validation": 7_705}


def main() -> int:
    target = read_json(AUDIT_DIR / "expanded_distribution_target.json")
    actual = read_json(AUDIT_DIR / "expanded_distribution_actual.json")
    newness = read_json(AUDIT_DIR / "newness_validation.json")
    counted = dataset_distribution(load_from_disk(str(TOKENIZED)), include_doc_counts=False)

    errors: list[str] = []
    if counted["splits"] != EXPECTED_SPLITS:
        errors.append(f"loaded dataset split counts differ from expected: {counted['splits']} != {EXPECTED_SPLITS}")
    if actual.get("splits") != EXPECTED_SPLITS:
        errors.append(f"audit split counts differ from expected: {actual.get('splits')} != {EXPECTED_SPLITS}")
    if strip_zero_counts(counted["samples_per_language_by_split"].get("train", {})) != strip_zero_counts(
        target["target_train_by_language"]
    ):
        errors.append("loaded train language distribution differs from target")
    if strip_zero_counts(counted["samples_per_language_by_split"].get("validation", {})) != strip_zero_counts(
        target["target_validation_by_language"]
    ):
        errors.append("loaded validation language distribution differs from target")
    if counted["duplicate_doc_chunk_keys"] != 0 or actual.get("duplicate_doc_chunk_keys") != 0:
        errors.append(
            "duplicate (doc_id, chunk_idx) keys found: "
            f"loaded={counted['duplicate_doc_chunk_keys']} audit={actual.get('duplicate_doc_chunk_keys')}"
        )
    if int(newness.get("new_vs_old_source_key_overlap", -1)) != 0:
        errors.append(f"new source keys overlap old source keys: {newness.get('new_vs_old_source_key_overlap')}")
    if int(newness.get("new_train_vs_validation_source_key_overlap", -1)) != 0:
        errors.append(
            "new train source keys overlap new validation source keys: "
            f"{newness.get('new_train_vs_validation_source_key_overlap')}"
        )

    status = "PASS" if not errors else "FAIL"

    for key, value in {
        "status": status,
        "tokenized": TOKENIZED.relative_to(REPO_ROOT),
        "audit_dir": AUDIT_DIR.relative_to(REPO_ROOT),
        "rows": counted["rows"],
        "train": counted["splits"].get("train", 0),
        "validation": counted["splits"].get("validation", 0),
        "duplicate_doc_chunk_keys": counted["duplicate_doc_chunk_keys"],
        "old_source_keys": newness.get("old_source_keys", 0),
        "new_train_source_keys": newness.get("new_train_source_keys", 0),
        "new_validation_source_keys": newness.get("new_validation_source_keys", 0),
        "new_vs_old_source_key_overlap": newness.get("new_vs_old_source_key_overlap", 0),
        "new_train_vs_validation_source_key_overlap": newness.get("new_train_vs_validation_source_key_overlap", 0),
    }.items():
        print(f"{key}={value}")

    train = counted["samples_per_language_by_split"].get("train", {})
    validation = counted["samples_per_language_by_split"].get("validation", {})
    print("language,train,validation,total")
    for language in sorted(set(train) | set(validation)):
        train_count = int(train.get(language, 0))
        validation_count = int(validation.get(language, 0))
        print(f"{language},{train_count},{validation_count},{train_count + validation_count}")

    if errors:
        for error in errors:
            print(f"error={error}")

    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import sys
import argparse
import os
import yaml

from llamafactory_moe_study.hooks.patch_llamafactory import apply_patch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to LLaMA-Factory config file")
    parser.add_argument("--use_method", default="adamole")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj")
    parser.add_argument("--adamole_num_experts", type=int, default=4)
    parser.add_argument("--adamole_max_threshold", type=float, default=0.25)
    parser.add_argument("--adamole_aux_coef", type=float, default=0.01)
    args, extra = parser.parse_known_args()

    # Debug: show YAML content before passing it on
    print("==== DEBUG: Checking YAML config at runtime ====")
    with open(args.config, "r") as f:
        yaml_content = f.read()
        print(yaml_content)

    parsed_yaml = yaml.safe_load(yaml_content)
    if "model_name_or_path" not in parsed_yaml:
        print("⚠️  ERROR: model_name_or_path missing in YAML!")

    apply_patch(
        use_method=args.use_method,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=set(args.lora_targets.split(",")),
        adamole_aux_coef=args.adamole_aux_coef,
    )

    sys.argv = ["llamafactory", "train", args.config]
    print("==== DEBUG: sys.argv passed to llamafactory ====", sys.argv)
    print(sys.argv)

    from llamafactory.cli import main as lf_main
    # TODO: Wrap model
    # TODO: inject adapters

    lf_main()

if __name__ == "__main__":
    main()

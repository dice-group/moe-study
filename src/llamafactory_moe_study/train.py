from llamafactory_moe_study.hooks.patch_llamafactory import apply_patch

def main():
    apply_patch()

    from llamafactory.cli import main as lf_main
    # TODO: Wrap model
    # TODO: inject adapters

    lf_main()

if __name__ == "__main__":
    main()

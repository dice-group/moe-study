import os
from hooks.patch_llamafactory import apply_patch
from core.wrappers import AuxLossWrapper
from methods.registry import get_selected_method

def main():
    apply_patch()

    from llamafactory.cli.train import main as lf_main
    # TODO: Wrap model
    # TODO: inject adapters

    lf_main()

if __name__ == "__main__":
    main()

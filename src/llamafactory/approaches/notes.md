# approaches

Each approach implements a tiny interface to keep everything swappable.

- `base.py`: defines `LoraKnobs` and `MethodAdapter` with:
    - `inject(model, knobs) -> model`
    - `aux_loss(model) -> scalar tensor`

- `registry.py`: maps `USE_METHOD` (the different approaches) to a `MethodAdapter` instance.
  - Add your method here once implemented.
  - all approaches should be registered here after implementation

- `adamole/`: AdaMoLE logic only (PEFT and LORA implementation were pulled our and should be kept as dependency).

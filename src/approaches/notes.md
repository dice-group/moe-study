# approaches

Each approach implements a tiny interface to keep everything swappable.

- `registry.py`: maps `USE_METHOD` (the different approaches) to a `MethodAdapter` instance.
  - Add your method here once implemented.
  - all approaches should be registered here after implementation

- `adamole`: AdaMoLE logic only, try to use PEFt and LORA utils as dependency 

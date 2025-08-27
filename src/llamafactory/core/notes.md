# core

Shared utilities that are method-agnostic.

- We need inject functionality which  replaces selected `nn.Linear` modules in-place with adapter-wrapped versions.

- We need wrappers which wraps a model to append aux-loss to the primary loss.
  - eg for AdaMoLE an AuxLossWrapper
  - should work with any method exposing a function such as `model.get_aux_loss()`

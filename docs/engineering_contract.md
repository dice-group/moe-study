# Engineering Contract

This repo should stay centered on the modular Megatron Bridge pipeline while
supporting multilingual MoE scaling-law and architecture-comparison
experiments.

## Core Rule

Experiments should usually change configs, profiles, Bridge recipes, scaling
matrices, benchmark choices, and experiment launchers. Core pipeline code should
change only when a stage contract or shared runtime capability genuinely needs
to evolve.

## Change Layers

- **Stable pipeline spine:** registry, context, config loading, executors,
  state/cache logic, stage order, and `plan/run/validate` contracts. Changes
  here need strong justification and relevant tests.
- **Stage-owned logic:** sampling, tokenizer training, tokenization, audit,
  training, checkpoint conversion, loss, benchmark, baseline, and summary code
  should keep logic, helpers, artifacts, and validation inside the owning stage.
- **Experiment layer:** MoE sweeps, scaling matrices, architecture comparison
  launchers, Bridge recipe variants, and profile wiring should live outside
  core pipeline mechanics whenever possible.

## Quality Attributes

- **Reproducible:** configs, seeds, manifests, and output lineage are explicit.
- **Modular:** stages communicate through documented artifacts, not hidden
  cross-stage state or loose-file inference.
- **Maintainable:** files are named clearly, obsolete artifacts are removed, and
  generic helper dumping grounds are avoided.
- **Auditable:** corpus sampling, tokenization, training, evaluation, and
  summaries produce inspectable manifests or checks.
- **Agent-friendly:** docs and layout make the correct edit location obvious.

## KISS and scientific-change rules

- Keep one canonical implementation and one documented command for each
  maintained workflow. Delete superseded, unreferenced implementation after
  retaining any required contract or evidence in its canonical location.
- Add an abstraction only when it gives a clear owner to a cohesive contract
  or removes demonstrated duplication. A speculative helper, framework, or
  configuration layer is not a cleanup.
- Prefer a small stage-owned module and explicit arguments over generic utility
  modules, hidden globals, or environment-dependent defaults. Cluster paths,
  accounts, partitions, and executables belong in named runtime profiles.
- Preserve public import and CLI facades when splitting a large module, but put
  new behavior in the owning package. Do not duplicate logic between the facade
  and extracted modules.
- Treat sealed manifests, result tables, and receipts as immutable inputs. A
  refactor must not silently regenerate training evidence or change numerical
  semantics; prove parity with focused fixture or CPU checks instead.
- State the independent experimental unit, seed scope, split boundary, planned
  configuration, and observed artifacts separately. Never turn checkpoints,
  languages, or repeated readouts into independent training replicates.
- Tests follow scientific and lifecycle risk, not file count. The three
  maintained suite manifests cover active contracts; historical recovery and
  cluster execution remain explicit opt-in surfaces.

## Output and Log Contract

Stage artifacts should remain stable under the profile `output_root`; logs
should remain grouped by pipeline invocation under
`<output_root>/logs/attempts/<attempt_id>/`. Do not make stage artifacts depend
on an attempt id unless the profile is explicitly modeling a separate
experiment. Parallel experiments should use distinct `output_root` values or
distinct `training.runs[*].id` values.

## Review and CI/CD Split

Codex PR review should enforce modularity, maintainability, cleanliness, and
repo hygiene. CI/CD should enforce deterministic correctness through config
validation, unit tests, dry-runs, fixture smoke tests, and later scheduled
cluster smoke checks.

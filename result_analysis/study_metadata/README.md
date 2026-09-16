# Training statistics and language sampling

The five requested quantities are available in the CSV files below.
Training statistics cover **159 runs**: 55 rank-sweep, 68 expert-sweep,
22 mixed-follow-up, 7 final-96, and 7 final-200 runs.

| Information | File | Column(s) |
|---|---|---|
| Tokens processed | [runs.csv](runs.csv) | `input_token_positions_seen` |
| Optimizer steps | [runs.csv](runs.csv) | `optimizer_steps` |
| Throughput (tokens/second) | [runs.csv](runs.csv) | `input_token_positions_per_second` |
| Training duration (hours) | [runs.csv](runs.csv) | `trainer_runtime_hours` |
| Language sampling proportions in the prepared training data | [languages.csv](languages.csv) | `dataset_train_document_share`, `dataset_train_example_share` |

In `runs.csv`, identify each training by `stage` and `run_id`. For the final
experiments, select `final_96` or `final_200` in `stage`.

Use the run's `dataset` value to find its language distribution in
`languages.csv`. Each row identifies a `language`; multiply the share by 100
to obtain a percentage. `train_documents` and `train_examples` give the
corresponding counts. [datasets.csv](datasets.csv) provides dataset totals
and construction sampling rates.

For definitions, provenance, and methodological notes, see
[technical notes](technical_notes.md).

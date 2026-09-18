# reports/

This directory holds generated evaluation output: recorded experiment runs
(`reports/experiments/*.json`) and any future benchmark reports.

`reports/experiments/` is intentionally excluded from version control (see
`.gitignore`). Every run in it is fully reproducible by re-running the
evaluation pipeline against the same dataset/model/prompt version, so
committing it would just be committing derived data that goes stale the
moment the dataset, model, or prompt changes.

When we later want to publish a specific, real benchmark result in the
README (Phase 15), we will copy that run's JSON, or a summary of it, into a
deliberately named, committed file here, for example
`reports/benchmark_2026-XX-XX_flan-t5-small_v1-vs-v2.json`, so the numbers
in the README are traceable back to an actual recorded execution rather
than regenerated silently.

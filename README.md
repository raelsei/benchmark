# Benchmark

AI model benchmark runs. Each entry is one task given to one model through a single prompt. The model's
delivered output is kept as-is, and a README records the prompt, the environment, what was actually
verified and the known limitations.

## Index

| Model | Task | Folder | What was built | Date |
|---|---|---|---|---|
| Claude Opus 5.5 (`anthropic/claude-opus-5-5`) | Photoreal real-time ocean, single `index.html` | [`opus-5-5/ocean`](opus-5-5/ocean/) | WebGL 2 GPU FFT ocean: JONSWAP + swell spectrum, 4 cascades, Jacobian foam, filtered sun glitter, atmospheric sky | 2026-09-27 |

## Layout

```text
<model>/<task>/
  README.md     # prompt (verbatim), environment, method, verification, limitations
  ...           # the delivered output, unmodified
```

- `<model>`: slug of the model id, e.g. `opus-5-5` for `anthropic/claude-opus-5-5`.
- `<task>`: short task slug, e.g. `ocean`. Use the same task slug (and the exact same prompt) across
  models so runs are directly comparable.

## Running a benchmark

Web benchmarks are static files. Serve them from the repository root:

```bash
python3 -m http.server 8000
# http://localhost:8000/<model>/<task>/
```

## What every benchmark README contains

1. **Original prompt**, verbatim and in its original language (a collapsible `<details>` block is fine).
2. **Model and run context:** model id, date, environment (agentic or chat, which tools were available),
   and the hardware/browser used for verification.
3. **Human intervention:** everything the human did beyond the prompt, or "none".
4. **Verified vs. not verified:** only claims backed by something that was actually run. Unmeasured
   numbers are not reported.
5. **Post-delivery changes:** anything changed after the model's final answer. Ideally nothing but file
   moves; confirm with a hash (for example `git hash-object`).
6. **Known limitations.**

## Rules

- Keep the delivered output unmodified. If a change is unavoidable, record it in the task README.
- Reuse the exact prompt when running the same task on another model.
- Keep one model/task pair per folder. Put repeated runs in sibling folders with a suffix
  (`ocean-2`, …) instead of overwriting.

## Adding a benchmark

1. Create `<model>/<task>/` and put the delivered output in it unchanged.
2. Write its `README.md` with the sections above.
3. Add a row to the [Index](#index).

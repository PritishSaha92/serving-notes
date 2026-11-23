# What the preparation timings can—and cannot—predict

This is a retrospective calculation added to the article on 23 November 2026. It is not a new GPU experiment or a profile of the official H200 run.

## Measured input

[Preparation timings](preparation-timings.json) extracts all 14 shape/mode medians from `serving/builder-artifacts/variant-repair/timing-v3.json`. It includes the original file's SHA256 and measurement scope. The cases are T = 8, 52, 56, 128, 256, 410 and 416, each with and without RoPE.

The saved benchmark uses hot, reused buffers on an NVIDIA L40: 100 calls per graph, 20 replays per sample, seven alternating pairs. It measures QK/RoPE preparation, casts and cache writes, excluding projections, FA3, CPU dispatch, communication and full serving. Across the 14 cases, median stock minus median repair rounds to 4.56–5.21 microseconds. That interval is a range across cases, not a confidence interval.

## Conditional calculation

The project's recorded target model has 62 attention layers. The source record includes the final-window report's 62-target/5-drafter layer accounting and the local 62-layer probe. This calculation assumes every target layer in an eligible forward receives the quoted savings. It does not claim that those local probes establish the H200 layer-by-layer cost or path coverage.

```
62 × 4.56 us = 282.72 us = 0.28272 ms
62 × 5.21 us = 323.02 us = 0.32302 ms
```

For an explicitly hypothetical 20 ms complete generation cycle, one such target forward and fully exposed savings:

```
time reduction = 0.28272 / 20 to 0.32302 / 20
               = 1.4136% to 1.6151%
speedup        = 20 / (20 - saving)
               ≈ 1.01434x to 1.01642x
```

These are time and speed ratios under the stated assumptions, not predictions of the competition's aggregate frontier metric. The denominator is not a measured H200 cycle time. The estimate is not a bound on H200 savings.

## Avoid double counting

- A target verification forward processes several proposed token positions together. The timing already covers its T positions: do not multiply again by draft width.
- If a cycle emits `a` accepted tokens, divide both cycle time and its saving by `a`; their fraction is unchanged, provided acceptance is unchanged.
- Eight ranks execute in parallel. Summing their saved device time does not give wall-clock savings.
- Prefill, fallback, actual activation frequency, hardware differences, overlap and added integration work all affect the end-to-end result.

The official results lack the per-call coverage and H200 preparation timings needed to resolve these factors. Neither this arithmetic nor a single lower score proves that the difference was noise, a regression, or an Amdahl-limited gain.

Run `python tools/check_performance_budget.py` from the repository root to verify the excerpt and arithmetic.

# Reduce Muse attention preparation launches while preserving stock output bytes

Candidate: `adaptive32fused-qkvstore-repair-04ec69722720`.
Patch SHA256: `04ec6972272094554f554bc864d08a872588e5e982acc8b1cbb87f447bef5fe2`.
Flags SHA256: `17501b5dcb9ff415c56c83f5d720521b47c4c7f59de9c49c1d9a7a4115d261c4`; stock flags, no CLI overrides.
Base: SGLang v0.5.23, commit `297e54fd105e974cdf401e5b4d20253a22c16f90`.

The stock FA3 preparation path normalizes Q/K, applies RoPE where applicable, converts the query and KV tensors to the required FP8 format, and writes the KV cache in separate operations. This patch folds the eligible query conversion and KV conversion/store into the QK preparation kernel, reducing launches and intermediate tensors. It retains stock BF16 rounding before the query's FP8 conversion, including the installed torch conversion's overflow and NaN behavior. K output remains BF16. Cache locations, padding and untouched slots retain stock behavior.

The model uses this path only when the existing FA3/cache/backend conditions permit it. Other backends, prefill, unsupported cache configurations and the unsupported stock BLOCK_M32 compilation family use the ordinary path. Tile selection derives from stock Q+K normalization rows, while the grid still covers the Q+K+V work. The BLOCK_M32 fallback is a general compilation-family rule based on tensor shape and head count, not a list of workload sizes.

This patch also retains the previously validated fused adaptive MoE implementation: one integer Triton kernel maps global routes to local experts and classifies occupancy; eligible small expert groups use M32 while other groups retain stock M48/M64. Precision settings, activation, normalization, route weights and collectives remain unchanged. It does not add prefix specialization, mixed chunking, shorter drafting, padding masking or runtime COUNT changes.

## Validation and limitations

Original finite query-byte discrepancies were preserved and independently reproduced. A controlled stock/fused tile and contraction matrix established layout-dependent reduction rounding in the original failing families. A tiling-only repair still failed the wider stock32 family, so that whole family now falls back through actual model dispatch. No tolerance was relaxed.

For the submitted repair, local L40 validation passed every T1–512 with both RoPE modes and two finite seeds (2,048 numerical cases), targeted boundary and exceptional/raw-bit cases, original failures before/after repair, changing-input/location graphs, cache-byte and untouched-slot checks, and actual model fallback tests. Existing registered and integration regressions also passed. Focused independent source/evidence review covered the exact final patch and the dense validation supplement.

The model integration tests used real preparation and cache operations with a stub attention consumer. Six supported SM90/PDL configurations compiled, but local tests did not execute H200 FA3/PDL or the complete multi-rank model. Whole-server startup/graph-pool memory and KV capacity remain official-evaluation questions. A separate 16-user check of this exact patch was requested; its state must be checked rather than assumed passed. Older A/B checks use different, locally blocked patches and do not validate this repair.

Matched hot-buffer L40 device preparation measurements showed approximately 4.56–5.21 microseconds per layer saved across the tested smaller sizes. They exclude CPU dispatch, projections, attention and complete serving, and do not establish an H200 speedup or score. The all-load scored run is an experiment; the earlier passing fused-stock entry remains preserved.

## Attribution

This is an independent adaptation of the pinned SGLang Triton QK/cache operations and Marlin integration. Existing upstream license and attribution are retained. Publicly available pinned FA3 source was reviewed for producer/consumer ordering; no FA3 implementation was copied into this patch. The patch contains four readable source files and no model-loading, tokenizer, weights or evaluation-harness changes.

# Independent dense-sweep addendum

Status: FINAL, revision1. Independent source/artifact review only; no GPU runs, official requests or production edits by this reviewer.

The original `INDEPENDENT_REVIEW.md` is unchanged at SHA256 `eae1edd79a3dcbe6a98c57ae17c72bb9a58d05f9889a380bac93ecdb3989d7fe`. This addendum supplies the additional numerical sweep evidence without changing its reviewed patch or limitations.

## Exact identity and completed numerical coverage

Candidate `adaptive32fused-qkvstore-repair-04ec69722720` remains patch `04ec6972272094554f554bc864d08a872588e5e982acc8b1cbb87f447bef5fe2`, QK source `80384367c2a48b4e1e21df25cf9fbf693e27e68330aea95ae7034fe417950775`. Independently rehashed both after the sweep.

Read the complete `probe_dense_sweep.py` and its imported production-reference/dispatch helpers. It asserts the candidate source hash before and after execution, verifies the reference source against pinned commit `297e54fd105e974cdf401e5b4d20253a22c16f90`, and compares real production functions with the appropriate stock BF16 query followed by the required torch FP8 cast. It uses actual target Q16/K2/D64 and production stride2304. The two seed families both generate finite normal-distributed BF16 inputs; positions/locations also vary deterministically.

Independently parsed the complete result and verified the exact set of2048 unique `(T,RoPE,seed)` tuples: every T1–512, both RoPE modes, seeds `T + int(RoPE)` and `T + 3000 + int(RoPE)`. Every row has all seven required checks true: query bytes, BF16 key bytes, full K/V cache bytes, query/key dtype and production stride. No missing or duplicate case, no recorded failure. The output reports complete=true and passed=true.

This is actual numerical eager coverage, unlike the earlier T1–512 scan which only enumerated `_triton_config` choices. It complements the existing exceptional-input and changing-input graph tests; it does not claim dense graph coverage at every count, exhaustive numerical input coverage, real-model FA3 consumption or H200 execution. It validates only the repaired B identity. Original A/B and the tiling-only intermediate remain blocked and preserved.

## Reconciliation of BOUNDARY_FAILURE_ASSESSMENT.md

Read the complete FINAL revision1 report and verified SHA256 `e2cd392f1dc1ce7ba425b8f00e5450d3326f3324355b3b778cbb5c34ca478ca1`. Preserve it as historical evidence; the following corrections are separate.

- The near-rounding-midpoint explanation is plausible supporting analysis, not permission to weaken byte gates. This reviewer did not independently rerun its CPU midpoint calculations. The owner's controlled production-kernel matrix and saved actual SM89 TTGIR/PTX provide stronger direct evidence that BLOCK_M changes reduction layout/grouping in the original failing families. Disabling contraction did not repair those mismatches and can add differences; no precision toggle was adopted.
- The proposed stock T409/410 comparison does **not** cross a stock threshold:18*409=7362 and18*410=7380 both select BLOCK_M8. Stock changes8→16 at T455/456 (8190/8208 rows). The fused original20*T path changes at410. Stock52/57 crosses4→8, but56/57 is the adjacent stock boundary. These are source arithmetic corrections, not new numerical GPU measurements.
- Absence of a difference in one stock-versus-stock sample would not universally refute a rounding/layout mechanism: selected values might not land near a rounding boundary. The controlled saved-fixture experiment is more informative than arbitrary same-leading-row samples. Stock batch-size variation, if observed, does not waive the required same-input stock comparison.
- The report's assumption that matching BLOCK_M/warps alone preserves every family was falsified by the later T3641 tests. The final repair therefore excludes the unsupported stock32 compilation family and restores ordinary model/backend cache ownership there. This fallback was validated with preserved failing fixtures and actual-model graph tests. No cause or intrinsic repair of the32-family arithmetic discrepancy is claimed.
- The original recommendation to numerically sweep both repaired A and B does not describe the implemented scope: only B was repaired and newly frozen. Its requested T1–512/two-seed/RoPE sweep is now complete; A remains blocked. No unsupported A-repair claim is made.
- Estimated mismatch rates, the asserted expected20 mismatches and “negligible” CTA cost are hypotheses/model estimates, not confidence intervals, correctness proofs or performance bounds. Actual local device timing is recorded separately and does not establish H200 serving gain.

## Conclusion

The missing dense numerical gate now passes for the exact unchanged repaired B patch. No new local blocker was found. This strengthens the local evidence; it does not validate queued old A/B requests by inheritance or remove the integrated H200, real FA3/PDL, official binary, whole-server memory and CPU-overhead limitations recorded in the original final review. The owner retains official-request and final-selection responsibility under the accepted experimental-risk policy.

## Artifact hashes

- `probe_dense_sweep.py`: `5404326cb5ac8502202654dec5c3edb4b55c691f7fc87f1a0f1d33bee6dbee0e`
- `dense-sweep-v3.json`: `ac0ab8b5f0633bf82931ccd8079cf6d1457e1041bfea078c3e8f1816284002ff`
- `dense-sweep-v3.log`: `e088fc6f6ca81e78622d2f8e37ddd2f07fc5e5d27d59f0802b9300feca3dd1ac`

Finalized UTC: 2026-11-20T01:34:25.661074+00:00

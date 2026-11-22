# One recorded query-byte failure

The interactive uses the original, blocked Variant B's **T = 410, RoPE off** fixture. This is a historical local L40 result, not the repaired candidate's result.

The [observation extract](rounding-observation.json) retains all three attempts for this case and the SHA256 of the source report. Each attempt records one differing query byte at token 404, column 627:

| Quantity | Value | Status |
|---|---:|---|
| Stock BF16 query element | 0.83984375 | Recorded GPU output |
| Stock FP8 query element | 0.8125 | Recorded GPU output |
| Original B FP8 query element | 0.875 | Recorded GPU output |
| Next BF16 number above 0.83984375 | 0.84375 | Exact arithmetic; not a captured intermediate |

The adjacent BF16 numbers have bit patterns `0x3f57` and `0x3f58`. Their distance is 0.00390625. The higher number is also exactly halfway between the E4M3 values 0.8125 (`0x35`) and 0.875 (`0x36`). Round-to-nearest, ties-to-even selects `0x36` at that tie.

This explains how one BF16 step can account for the observed FP8 jump. It does **not** reconstruct the fused GPU's exact pre-conversion intermediate or run its normalization in JavaScript. The stock BF16 point lies 43.75% of the way across the displayed FP8 interval; the neighbor lies at 50%.

The displayed E4M3 fields are `0 | 0110 | 101` for stock and `0 | 0110 | 110` for original B: one sign bit, four exponent bits and three fraction bits. The last two fraction bits differ (`0x35 XOR 0x36 = 0x03`). The display highlights those two bits; it does not confuse one differing byte with one flipped bit.

Run `python tools/check_rounding_example.py` from the notebook root to check the recorded sample values, the BF16 adjacency, the E4M3 rounding and the plotted position. This CPU check is an arithmetic check of the illustration, not a kernel regression test.

The original CPU analysis is `serving/research/final-opportunity-audit/probes/boundary_midpoint_analysis.txt`. Its historical recommendation was superseded by the completed repair. In particular, stock T409/T410 does not cross a stock tile boundary, and matching stock's tile did not resolve the wider BLOCK_M=32 family. The [final repair review](repair-review.md) and [dense-sweep review](dense-sweep-review.md) describe the retained fallback and tests.

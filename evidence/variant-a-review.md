# Variant A: independent review

**Status: FINAL, revision 1 (2026-11-19 20:10 UTC).**

Reviewer: independent research stream. I write only in `research/final-opportunity-audit/`. I changed no implementation, ran no GPU work, used no quota and submitted nothing.

`REPORT.md` (sha256 `2dbd9f6a…`) is **left unchanged on purpose**: the coding agent binds research hashes into its selection records. Its errors are corrected in section 2 here instead.

## Verdict

**No blocker found.** Nothing I found should invalidate the pending one-user check `1ca49f01cff9467a`.

- **Correctness argument:** cache, Q and K bytes match the fused-stock parent by source proof plus executed L40 tests.
- **TRT-LLM and unsupported backends:** unchanged.
- **Hierarchical-cache waits and slot-0 handling:** preserved.
- **Ordering:** the stock query cast keeps FA3 strictly ordered behind the PDL fused kernel.

**What remains unproven** is integrated H200 behavior: the real FA3 consumer, PDL on SM90, real pools with hierarchical-cache waits under graph replay, and every verify bucket above 8 tokens. Two practical limits matter for selection:

1. **The check will very likely return after the window.** Queues have run 14–18 h, so a result is improbable before the 02:00 UTC selection or even the 08:00 deadline. The coding agent's own rule then makes Variant A ineligible.
2. **Even a pass covers only the 1-user, 8-token verify bucket**, and the expected 1-user effect (about 1–1.5 % of a step, extrapolated from L40) is below that rung's 4–6 % run-to-run spread. A pass would show safety, not benefit.

**Reviewed identity:**

| Item | Value |
|---|---|
| Candidate | `adaptive32fused-kvstore-f82cf8dd9f16` |
| Patch | sha256 `f82cf8dd…00d4` (four files: the fused MoE file unchanged at source `c4c81206…`, plus the QK-norm kernel, the FA3 backend and the Muse model) |
| Flags | stock, sha256 `17501b5d…1c4`; no mixed chunking |
| Parent | fused-stock `adaptive32fused-a2d84a8290b0` (scored 1.0251) |

---

## 1. What changed [source: frozen patch]

- **`kernels/ops/attention/muse_spark_qk_norm_rope_store.py`**
  - **New kernel branch.** `FA3_FP8_STORE` constexpr. K is rounded to BF16, then converted with `tl.float8e4nv`, with NaN sign restored (`sign | 127`). V is converted from its raw BF16 value. The existing ±448 clamp and the BF16 pre-rounding of Q/K are bypassed on this branch; the TRT-LLM branch keeps both, unchanged.
  - **Bounds assert.** `tl.device_assert(loc in [0, kv_capacity))`, active through `debug=fa3_fp8_store`.
  - **`q_out`** stays BF16 when `fa3_fp8_store`.
  - **New methods:** `fa3_fp8_store_eligible` (Triton auto-selected, no forced backend, not torch.compile, BF16 input, Triton eligibility) and `forward_fa3_fp8_store`.
- **`layers/attention/flashattention_backend.py`**
  - `get_fa3_fp8_kv_write_buffers` calls `get_fused_kv_write_buffers(..., fa3_fp8=True)`.
  - The `fa3_fp8` branch requires FA3, unscaled non-MXFP8 E4M3, page size 1, `k_scale` and `v_scale` both None, and decode or target verify. It then reuses the existing exclusions (MLA, cross attention, translation, KV shard pool, skipped cache, head mismatch, CP, DCP mask), locations and shape checks, plus contiguity and FP8 dtype.
  - **Default callers keep the BF16-only behavior:** `fa3_fp8` is keyword-only and defaults to False. The other in-tree caller, `models/mimo_v2.py:811`, and the hybrid and base backends are unaffected.
- **`srt/models/muse_spark.py`**
  - `_fused_kv_store` uses the FA3 hook when the backend has it, and returns its result (possibly None) **before** the TRT-LLM branch.
  - `forward` takes the FA3 fused-store path only when the op is eligible, and otherwise drops `kv_store` back to the stock store.
  - `self.attn(..., save_kv_cache=kv_store is None)` is unchanged.

---

## 2. Corrections to my earlier `REPORT.md`

| Earlier claim (`REPORT.md` §2.1, §2.3) | Correction | Evidence |
|---|---|---|
| "PyTorch's E4M3 cast yields NaN at ≥ 480" | **Wrong for the pinned torch 2.14.1.** Finite overflow and ±inf saturate to 0x7E (±448). Rounding that would carry into 0x7F also saturates to 0x7E, so (464, 480) becomes 448. NaN becomes 0x7F with its sign preserved. | Pinned `pyproject.toml` requires `torch==2.14.1`. Header `torch/headeronly/util/Float8_e4m3fn.h` (sha256 `7570fe37…`), `fp8e4m3fn_from_fp32_value`, lines 171–243. It is a `C10_HOST_DEVICE` software conversion, so architecture-independent. |
| The kernel's clamp "also turns NaN into ±448", proposed as a parity issue | True for the **TRT-LLM** branch only. Variant A uses a new branch with no clamp: Triton's saturating conversion plus explicit NaN-sign restoration, which matches torch exactly. | Frozen patch; `conversion-v1.json` (direct Triton cast: 127 mismatches, all negative NaN; corrected: 0 of 65,536) |
| "Baseline writes padded rows into slot 0 (garbage); the fused kernel skips it" | **Wrong.** Stock `store_cache` skips slot 0 by default (`reserved_skip_index=0`), so both skip it. Stock also device-asserts indices outside `[0, size_limit)`, so the patch's new bounds assert mirrors stock rather than adding a new failure class. | `kernels/ops/kvcache/kvcache.py:80-119` |

The rest of `REPORT.md` §2 (kernel inventory, eligibility and ordering argument) stands. Its saving estimate remains a hypothesis; section 6 adds the coding agent's L40 measurement.

---

## 3. Review by priority

### 3.1 Conversion parity, including NaN

- **Source proof:** the torch 2.14.1 behavior above.
- **Triton side.** It lowers fp32 to `float8e4nv` as `cvt.rn.satfinite.e4m3x2.f32`. That instruction is present in all six SM90 PTX variants, the same as on SM89. It rounds to nearest even, saturates finite values, and returns canonical positive NaN. The patch restores the sign from the BF16-rounded fp32 bits: `(bits >> 24) & 128`.
- **Rounding chain:**
  - K: fp32 → BF16 (`.to(bf16)`, the same conversion the `k_out` store uses) → E4M3. Identical to stock's BF16 `k` followed by `.to(float8_e4m3fn)`.
  - V: raw BF16 → E4M3, identical to stock.
- **Executed tests (L40):**
  - All 65,536 BF16 bit patterns, through a replica kernel containing the same conversion lines. **The replica is not the production kernel.**
  - The production kernel: 48 path cases (T = 1, 8, 17, 32, 64, 128, 256, 700; RoPE and non-RoPE; random, adversarial and random-BF16-bit inputs). Q, K and cache bytes all match the pinned no-store kernel → torch cast → stock `store_cache`, including untouched and padding slots (`path-v1.json`).
- **SM90:** compiled only, never executed. The residual risk is an ISA-level conversion difference between SM89 and SM90; it is negligible for the same PTX instruction.

### 3.2 Preserving TRT-LLM and unsupported-backend behavior [source]

- **TRT-LLM.** `TRTLLMHAAttnBackend` subclasses `FlashInferAttnBackend` (`layers/attention/trtllm_mha_backend.py:176`), so it has no FA3 hook and reaches the unchanged TRT-LLM branch with clamp and FP8 query.
- **Other FA-family backends:**
  - FA4 instances and FA3 with a BF16 KV cache: the hook returns None. That equals stock's result, since they were never TRT-LLM.
  - `HybridAttnBackend` and the base backend lack the hook, so they fall through as before.
  - `MusaFlashAttentionBackend` inherits the hook but fails the Triton CUDA capability, so it falls back.
- **Forced native or other fused-op backend, and torch.compile:** `fa3_fp8_store_eligible` returns False and `kv_store` is dropped, giving the stock store. A test covers the forced torch backend.
- **Prefill, MIXED and EXTEND modes:** the hook returns None, so the stock store runs. Breakable-prefill replay, where the store is known to crash, stays excluded.

### 3.3 FA3 eligibility, full and SWA locations, padding, hierarchical-cache waits [source]

- **Locations.** They come from `KVWriteLoc.for_layer(forward_batch, layer, swa_loc=forward_metadata.swa_out_cache_loc)`, with the same full/SWA choice the stock BF16 fused hook uses. That agrees with the stock SWA pool's `set_kv_buffer` selection (`mem_cache/swa_memory_pool.py:253-290`). `KVWriteLoc.__post_init__` trims SWA locations to the token count (`mem_cache/memory_pool.py:1946-1955`).
- **Padding.**
  - Full layers: padded `out_cache_loc` uses the ZERO policy.
  - SWA layers: replay zeroes `cuda_graph_swa_out_cache_loc[n:]` past the live count and copies live IDs into the prefix; capture zeroes all of it (`flashattention_backend.py:614-627`).
  - Padded rows therefore target slot 0. The kernel skips it, and the bounds assert holds, as does stock's.
- **Hierarchical-cache waits.** The hook reads buffers with `get_kv_buffer`, which waits on the layer-transfer counter (`swa_memory_pool.py:232-234`) and returns FP8-typed views (`memory_pool.py`: `.view(self.dtype)`).
  - The wait now happens just before the fused kernel instead of inside FA3, so the overlap is marginally earlier and harmless.
  - Stock source explicitly anticipates fused writers that scatter through `get_kv_buffer` bypassing `_check_physical_write_loc` (`memory_pool.py:2108-2118`). The hook declines whenever translation is active.
  - Bypassing `set_kv_buffer` drops no hierarchical-cache side effect: `hicache_write_back_staging` is configured by the builders, not filled by the store.
- **Activation preconditions on H200** (FA3, `fp8_e4m3`, page size 1, unset scales) match the returned startup log and the official problem text ("kv_scale … 1 in the scored cases (SGLang leaves them unset)"). **Activation itself is not observable in returned logs:** the patch emits no log line. If any precondition failed on H200 the candidate would silently run the stock store, and a no-gain result could not be told apart from inactivity.

### 3.4 CUDA-graph replay, PDL ordering, retained query cast

- **Graph replay [executed, L40]:**
  - `graph-v2.json`: 72 changing-input, position and location replays at 8, 128 and 256 tokens, with stable addresses and an all-zero (fully padded) case.
  - `layers-v2.json`: 62 sequential layers across three retained buckets.
  - All bytes exact. These are kernel-level graphs, not the server's graph runner.
- **PDL [source plus SM90 compile; unexecuted].**
  - The fused kernel keeps its existing PDL entry wait and immediate trigger; `griddepcontrol` is present in the SM90 PTX.
  - The next kernel is PyTorch's ordinary `q.to(fp8)` cast in `prepare_paged_mha_query` (`flashattention_backend.py:1427-1439`; decode `:2186`; extend `is_prefill=True` still casts on FA3). It launches without the programmatic-serialization attribute (`PDL_SOURCE_NOTE.md`), so it starts only after the fused kernel completes, and FA3 follows it.
  - The KV writes are therefore complete before FA3 reads them. This is the stock ordering, because stock also had an ordinary kernel before FA3.
  - **The L40 cannot execute PDL, so it has never run.**
- **Query.** It stays BF16 out of the fused kernel and is cast by stock code, so its dtype and rounding are unchanged by construction (Variant A).

### 3.5 Production wiring against stubs (gaps)

| Covered by executed tests | Stubbed or not exercised |
|---|---|
| Real `MuseSpark1xAttention.forward`; real FA3 hook, eligibility and `KVWriteLoc`; full and SWA selection; decode and target verify; 18 fallback configurations; forced torch backend; prefill fallback (`test_integration.py`, `integration-isolated.log`: 29 passed plus 4 subtests) | FA3 backend built with `__new__` and hand-set attributes; **pool `get_kv_buffer` is a lambda** (no real `SWAKVPool` waits or FP8 views); `forward_metadata` fixed; **`self.attn` replaced by a recorder**, so the FA3 query cast and kernel never ran; `use_o_gate=False` in the fixture (the production QKV is row-strided with a gate; the regression test does cover a row-strided 2,304-wide QKV) |
| Production kernel bytes, eager and graph (sections 3.1 and 3.4); registered QK and store tests (27 passed) | Server graph runner with the real FA3 metadata refill and hook together; PDL; SM90 execution of any variant; **every verify bucket above T = 8 on H200** (the 1-user check only exercises bucket 1) |

### 3.6 Implementation defects that could invalidate the pending check

None found. Points checked:

- `get_fused_op_backend`, `auto_selected_backend` and the Triton `debug` launch option exist. The local suite imported and ran them on torch 2.14.1 with Triton 3.8.0.
- `debug=True` compiles exactly three assertion kinds into the SM90 PTX: Triton's int32 add and mul overflow sanitizers, and "KV slot out of bounds". Eligible shapes (at most 256 tokens, QKV stride 2,304, 64-wide heads) cannot overflow int32, and valid locations satisfy the bounds check.
- Every launch attribute the check needs is present: PDL (`launch_pdl=True`, `griddepcontrol` in SM90 PTX) and debug assertions.

---

## 4. Non-blocking issues (for the coding agent's record; no action needed before 02:00 UTC)

1. **No activation signal.** A one-time INFO log when the FA3 store path is selected during target-verify graph capture would appear in the returned startup log, since capture precedes the first request. That would settle "inactive or no gain". **Do not change the frozen candidate for this now.**
2. **`debug=True` in a production launch** keeps Triton's overflow sanitizers on. The cost is small relative to the saved launches, but maintainers may object in an upstream PR. A non-debug guard, or reliance on the existing `store_cache`-style assert, would be cleaner.
3. **The 1-user check leaves the 16–256-token verify kernels unexecuted on H200.** An all-load scored run would be their first H200 execution.
4. **The real-pool hierarchical-cache wait has only source evidence,** as does the earlier wait position.

## 5. Source, tests, compilation, H200

| Claim | Source proof | Executed (L40) | SM90 compiled | H200 executed |
|---|---|---|---|---|
| E4M3 conversion parity with NaN, overflow and infinity | yes (torch 2.14.1 header) | yes (65,536 replica, 48 production cases) | yes (`cvt.rn.satfinite`) | no |
| Q/K BF16 unchanged, query cast retained | yes | yes (path and graph bytes) | — | no |
| Full/SWA locations, padding slot 0, bounds | yes | partly (stub pool, fixed metadata) | — | no |
| Hierarchical-cache wait before write | yes | no (stubbed) | — | no |
| Graph replay with changing inputs | — | yes (kernel-level) | — | no |
| PDL ordering behind the query cast | yes (PyTorch launcher, CUDA PDL rules) | no (SM89 lacks PDL) | yes (`griddepcontrol`) | no |
| FA3 consumes the fused cache correctly | inferred (bytes identical) | no (consumer stubbed) | — | no |
| Fallbacks (TRT-LLM, other backends, prefill, MIXED, forced backend) | yes | yes (18 configurations plus 2) | — | no |

## 6. Performance evidence

- **Measured (L40, hot buffers, `timing-v1.json`):** per layer, stock QK-norm/RoPE + K and V casts + `store_cache` + query cast against Variant A's fused kernel + query cast.
  - Stock: 5.98–7.62 µs. Variant A: 2.55–3.97 µs.
  - That is about 3.4–3.7 µs per layer, about 0.21–0.23 ms per 62-layer step.
- **Not measured:** FA3, projections, distributed effects, H200 node costs.
- **Implication at 1 user** [hypothesis]: about 1.3 % of a 16.7 ms step. That is below the rung's 4–6 % spread. No score effect claimed.

## 7. Recommendation for the final window

- **Leave Variant A as the coding agent has classified it:** ineligible for the final scored slot unless its check returns a complete pass before 02:00 UTC, which is unlikely given the queue.
- **If it does return:** read it for correctness, later-turn agreement, memory, TTFT and startup, and treat any small 1-user ratio change as unresolved. A pass validates only the 8-token bucket.
- **The provisional fused-plus-mixed plan and the preserved fused-stock 1.0251 are unaffected** by anything found here.

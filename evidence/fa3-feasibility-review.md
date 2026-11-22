# Variant B and runtime COUNT: feasibility

**Status: FINAL, revision 2 (2026-11-20 00:00 UTC).** Supersedes the preliminary revision 1 (23:55 UTC). Variant B's ordering question is now answered from the pinned FA3 source.

Reviewer: independent research stream. No implementation changes, GPU runs, quota or submissions. Nothing here alters the coding agent's final-selection schedule. All pending candidates and earlier reviews are preserved.

Labels: **[source]** pinned tree `297e54fd`, its patches and the pinned FA3 source below · **[measured]** existing local or official numbers · **[inferred]** reasoning or arithmetic, not measured.

## Summary

| Item | Verdict |
|---|---|
| **A. Runtime COUNT** in the fused route classifier | Exact and simple. The exposure is real per first use, but its frequency is unknown, and there is no clear penalty in the official summaries. **Low merit: an upstream cleanup, not a final-window candidate.** |
| **B. Variant B** (the fused kernel emits the stock FP8 query) | **The ordering concern is resolved by the pinned FA3 source.** On every reachable path, FA3's Q/K/V loads are ordered after the fused kernel completes, whatever the fused kernel's PDL trigger does. Exact conversion is straightforward. **Merits implementation as a follow-up to Variant A** once Variant A has completed integrated H200 validation. Its gain is small and inferred, about 0.06–0.19 ms per step. |

---

## A. Runtime COUNT for `_map_split_routes_kernel`

**What already exists.**

- [source] The frozen fused patch `a2d84a82…7f6c` declares `COUNT`, `B` and `BINS` as `tl.constexpr`:
  - `COUNT` only bounds masks (`i < COUNT`).
  - `B` (the next power of two of the route count) and `BINS` set tensor shapes, and must stay compile-time.
- [source] The non-adaptive `_map_local_experts_kernel` already takes a runtime count.
- [measured, coding agent, `research/FUSED_COMPILATION_COVERAGE.md`]
  - Adaptive token counts are 116–512, so COUNT is 928–4,096, with 397 possible values.
  - Captured shapes are verify T 128–256 (9 values) and prefill buckets 128–512 (17 values).
  - 380 eligible counts fall outside them, and can arise only on eager fallback: prefill-graph replay rejected, or an eager verify.
  - On an L40 the first call per new COUNT took 51–102 ms (single process, local Triton 3.8.0); warm calls took 18.8–41.0 µs.

**Exact change.**

- `COUNT` becomes a runtime argument, with `@triton.jit(do_not_specialize=["COUNT"])` to avoid separate variants for odd counts.
- `B` and `BINS` stay `tl.constexpr`.
- That leaves at most three binaries, one per B (1,024, 2,048 and 4,096). Captured buckets already exercise all three, so they compile before serving [inferred].
- Masked lanes already load −1 into histogram bin 0, so outputs are identical [source].

**Is the exposure meaningful?**

- **Per event, it can be.** A stall of 51–102 ms on one rank holds the whole TP group at the next collective.
- **The bound** is at most 380 events per process, only if every uncaptured count actually occurs eagerly. The shared Triton disk cache's effect across ranks is unmeasured.
- **Frequency is unknown.** Returned logs stop at startup.
- **Official later-turn TTFT p90, fused against parent:** 0.225 against 0.224 s at 1 user, 0.260 against 0.254 at 4, 0.346 against 0.352 at 16, 2.915 against 3.101 at 32. **No clear penalty is observed in these summaries.** Cross-job p90 values cannot rule out rare first-use stalls or establish their frequency.

**Strongest counterargument.** It removes a bounded, possibly rare, unobserved cost at the price of a new candidate identity replacing a frozen kernel that passed four loads.

**Cheapest falsifying test** (coding agent's L40):

1. Warm the 26 captured counts.
2. Sweep all 397 eligible counts.
3. Expect zero new cubins with runtime COUNT (against up to 371 with constexpr), and bit-identical integer outputs.

This tests compilation exposure only, not serving benefit.

**Merit:** low. Not for the final window.

---

## B. Variant B: emit the exact stock FP8 query

### B.1 What already exists [source]

- The fused kernel's TRT-LLM branch already emits an FP8 query. Its clamp and pre-rounding differ from the FA3 stock semantics.
- Variant A (`f82cf8dd…00d4`) emits a BF16 query, then stock FA3 casts it with `q.to(self.kv_cache_dtype)` in `prepare_paged_mha_query` (`layers/attention/flashattention_backend.py:1427-1439`).
- Variant A's K/V conversion (BF16 rounding, saturating `cvt`, NaN-sign restore) is already validated for exact torch 2.14.1 parity: 65,536 replica cases plus 48 production-kernel byte cases (`builder-artifacts/variant-a/`).

### B.2 Exact change

- In the `FA3_FP8_STORE` branch, convert Q rows with the same code as K: `x.to(bf16) → saturating E4M3 → NaN-sign restore`. Store Q as E4M3, with `q_out` allocated as `float8_e4m3fn`.
- FA3's `q.to(fp8)` then returns the same tensor, because the dtype already matches, so the separate cast launch disappears: **62 graph nodes per verify step** beyond Variant A.
- Restrict to the existing Variant A gate: FA3, unscaled E4M3, page size 1, decode or target verify. Prefill and MIXED keep the stock path. That matters because prefill-graph eager attention sizes outputs from `q` (`layers/radix_attention.py:133-147`).

### B.3 Ordering: audited against the pinned launcher [source]

**Provenance.**

- The pinned `kernels/aot/CMakeLists.txt:83-89` fetches `sgl-project/sgl-attn` at commit `f89bc2306632d1ec5f97b014dded4254f5b4a907` (URL hash `418b5681…`).
- I downloaded these files from that commit directly (sha256 prefixes):

  | File | sha256 |
  |---|---|
  | `hopper/flash_fwd_launch_template.h` | `f2490ee9…` |
  | `hopper/flash_fwd_kernel_sm90.h` | `2fd9c61c…` |
  | `hopper/flash_prepare_scheduler.cu` | `bc45f3f7…` |
  | `hopper/flash_api.cpp` | `35f2f6f5…` |
  | `hopper/flash_fwd_combine_launch_template.h` | `893a780a…` |

**What the source shows.**

1. **No cluster launch for Muse.** The cluster path requires 2-byte types with head dimension ≥ 128, or head dimension 192, and no varlen (`flash_fwd_launch_template.h:226`). Muse's FP8, head-dimension-64, varlen calls take the non-cluster launch: `cutlass::kernel_launch(..., launch_with_pdl = Arch >= 90 && Varlen && !skip_scheduler_metadata_computation && prepare_varlen_pdl)` (`:206-207`).
2. **When the main kernel is PDL-launched, FA3's own prepare kernel runs first.** That requires `Varlen && !skip_scheduler_metadata_computation`, and in exactly that case FA3 first runs `prepare_varlen_num_blocks` (`:173-175`).
   - The prepare kernel is launched with an **ordinary triple-chevron launch** (`flash_prepare_scheduler.cu:232`).
   - It only triggers its dependents early (`:72`).
   - An ordinary launch starts only after the preceding stream work, here the fused kernel, completes. The fused kernel's entry trigger affects only PDL-attributed successors, and the prepare kernel is not one.
3. **Otherwise the main kernel is ordinarily launched.** That covers non-varlen calls, and calls where `scheduler_metadata` is supplied (`skip_scheduler_metadata_computation`, `flash_api.cpp:999`), as on full-attention decode steps per the official problem text.
4. **Inside the main kernel**, the producer warp calls `wait_on_dependent_grids()` before "Load Q, K, V" (`flash_fwd_kernel_sm90.h:350-370`). That wait orders it after the ordinarily launched prepare kernel.
5. **The combine kernel** (split-KV) is PDL-launched after the main kernel and reads only partial outputs (`flash_fwd_combine_launch_template.h:51`).

**Conclusion.** On every reachable path, an ordinary kernel boundary separates the fused kernel from FA3's first Q/K/V load:

- either the FA3 main kernel itself, launched ordinarily
- or FA3's ordinarily launched prepare kernel, which the PDL main kernel then waits on

Under CUDA-graph capture, ordinary launches become full-dependency edges. The safety does **not** rely on disabling the fused kernel's trigger, or on SASS wait or trigger presence.

**Dispatched-path checks [source].**

- Muse's calls are varlen: SGLang passes `cache_seqlens`, and `use_prepare_varlen = is_varlen` (`flash_api.cpp:971-972`). Batches up to 32 are within `PREPARE_VARLEN_MAX_BATCHES_1CTA = 992`.
- The FA3 backend's `q.dtype` uses are all on MLA-only paths (`flashattention_backend.py:1893-1945`, `:2350`).
- `padded_extend_real_tokens` returns None for target verify (`layers/attention/graph_utils.py:146-158`).
- So an FP8 query changes no backend branch on Muse's decode/verify path.

### B.4 Remaining uncertainty

- **Binary identity is not established.**
  - The installed `sglang-kernel==0.4.10` matches the pinned requirement (`python/pyproject.toml:82`). It is a scikit-build-core wheel, tag `cp310-abi3-manylinux2014_x86_64`, with `flash_ops.abi3.so` RECORD hash `rT5f1CmN…`. That supports relevance.
  - It does not prove the judge image's binary was built from `sgl-attn f89bc230`, or that the binary equals the wheel inspected here.
  - A SASS scan was started and **stopped as low-value**: instruction presence cannot prove launch attributes or wait placement.
- **Not executed anywhere:** the integrated FA3 path with a fused FP8 query, and PDL (the L40 lacks it).
- **Gain is inferred, not measured:** 62 nodes at an assumed 1–3 µs each, about 0.06–0.19 ms per step. The L40 timing for Variant A included the query cast in both arms, so it does not isolate this saving.

### B.5 Strongest counterargument

The extra saving is small and unmeasured. It builds on Variant A, which itself has no completed integrated H200 validation. And it couples the query's dtype to a fused kernel shared with the TRT-LLM branch, so a later change to FA3's launcher, or a different binary in the image, could reintroduce the ordering risk without warning.

### B.6 Cheapest falsifying tests

1. **Exactness (L40, no FA3 needed).** Extend Variant A's 48 path cases and its graph replays to compare Q bytes against `q_bf16.to(torch.float8_e4m3fn)`. Include all 65,536 BF16 patterns through the production kernel's Q rows.
2. **Ordering (any SM90 machine).**
   - Capture a target-verify graph: fused kernel → FA3, for both a sliding layer (no scheduler metadata) and a full-attention decode layer (with metadata).
   - Read edge data with `cudaGraphGetEdges_v2`. Expect a default (full) edge from the fused kernel into FA3's first kernel (the prepare kernel or the main kernel), and a programmatic edge only from prepare into main.
3. **Benefit (H200).** Paired 1-user per-user throughput against Variant A, within the rung's noise limits. A gain this small may not be resolvable in one run.

### B.7 Merit

- **Worth implementing as a follow-up once Variant A has a completed passing integrated H200 result.** The ordering obstacle is gone at the source level, and the conversion reuses validated code.
- **Not a final-window item:** an H200 result before the 04:00 UTC cutoff is unlikely given recent queue times. Live decisions remain with the coding agent.

## Evidence locations

- Downloaded FA3 sources: session scratchpad, `sglattn/`, hashes above. They are not copied into the repo, to keep third-party code out of the research directory.
- Local SASS mnemonic check (`ACQBULK` and `PREEXIT` on SM90a, from a 5-line test kernel): scratchpad `pdl/`. It is not used as evidence for the verdict.

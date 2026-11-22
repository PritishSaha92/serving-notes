# Sources & research trail

Companion to [Learning to optimize a serving system](index.html), by Pritish Saha. Citation audit: **22 November 2026**.

The article's measurements come from the saved project results. Public papers explain the underlying methods; upstream source establishes the implementation; other entrants' reports supplied hypotheses. These are different kinds of evidence.

**Reading history matters.** The initial proposal's source register explicitly marked the broad paper and course catalog unread. Later reviews examined additional implementation paths. The bibliography below adds appropriate conceptual credit without turning suggested reading into a claim that I personally watched every lecture or that the agents completed every paper. “Recorded review” describes the saved research notes, not an independent reproduction of the source's benchmarks.

## Papers behind the system {#papers}

These are conceptual references for the article. Their titles, author metadata and abstracts were checked for this citation pass; this is not a claim of a new full-paper replication study. The linked implementation and project evidence remain the authorities for this particular November 2026 stack.

1. **Gene M. Amdahl (1967).** [Validity of the single processor approach to achieving large scale computing capabilities](https://doi.org/10.1145/1465482.1465560). AFIPS Spring Joint Computer Conference, pp. 483–485. The article uses the familiar fixed-work speedup model to distinguish a component gain from a system gain; it is an illustrative calculation, not a fitted model of these runs.
2. **Lianmin Zheng et al. (2024).** [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104v2). Background for the serving runtime and cache reuse. The paper predates the pinned Muse implementation and does not specify its current DFlash, FA3 or Marlin paths.
3. **Elias Frantar, Roberto L. Castro, Jiale Chen, Torsten Hoefler and Dan Alistarh (2024).** [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models](https://arxiv.org/abs/2408.11743v1). Background for efficient weight-quantized inference. [Original implementation](https://github.com/IST-DASLab/marlin). The experiment uses SGLang's later NVFP4 MoE adaptation, not the paper's original kernel unchanged.
4. **Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani and Tri Dao (2024).** [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608v2). Credit for the downstream attention algorithm. The final patch changes input preparation, not the FA3 attention algorithm. The paper's reported speedups are not this project's results.
5. **Jian Chen, Yesheng Liang and Zhijian Liu (2026).** [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036v2). [ICML proceedings](https://proceedings.mlr.press/v306/chen26bv.html). Background for parallel drafting and target verification. The actual competition used the Muse-specific DFlash2 model and its fixed configuration; the original paper is not a substitute for that model card or worker code.
6. **Biao Zhang and Rico Sennrich (2019).** [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467v1). Credit for RMS normalization. The article's reduction-order bug was diagnosed from the exact kernel and saved failures, not derived from the paper.
7. **Jianlin Su et al. (2023 revision of a 2021 preprint).** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5). Credit for RoPE. The pinned Muse implementation determines its interleaving, lookup tables and rounding sequence.

## Contract, models & pinned implementation {#implementation}

The base is **SGLang `297e54fd105e974cdf401e5b4d20253a22c16f90`**. Links below deliberately use that revision rather than today's `main`.

| Source | What it supports / recorded scope |
| :--- | :--- |
| [Official competition documentation](https://openkernelchallenge.dev/llms.txt), [rules](https://openkernelchallenge.dev/rules), [serving metadata](https://openkernelchallenge.dev/api/v1/serving) | Workload and evaluation contract. Live pages may change; the article reports saved November evaluations, not current quotas or rankings. |
| [Muse Spark Open NVFP4 model card](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B-NVFP4) | Model identity. The original audit had only partial metadata access; it did not read gated weights/configuration. Model dimensions used in the article were checked in source and instantiated probes. |
| [Muse DFlash2 card](https://huggingface.co/RadixArk/Muse-Spark-Open-DFlash2) | Model-specific draft configuration. The initial register records a full README/config review at revision prefix `46c90f41`. Its launch examples are not the competition launch. |
| [Muse model](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/models/muse_spark.py) | Expert dimensions, local expert ownership, attention dispatch and integration points. Initial register: entire model file; later targeted attention/caller reviews. |
| [NVFP4 Marlin expert module](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/muse_spark/muse_spark_fp4_moe.py) | Stock tile heuristic, alignment, two projections and finalization. Initial register: complete module. |
| [Marlin MoE wrapper](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/ops/moe/moe_wna16_marlin.py) and [kernel template](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h) | Existing packed-weight computation, scheduling and scratch contracts. Wrapper read completely; kernel template read selectively, not every inner-pipeline line. |
| [QK normalization / RoPE / store kernel](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/ops/attention/muse_spark_qk_norm_rope_store.py) | Existing fusion machinery, row-count dispatch and materialization boundaries. The final work extends this code, rather than inventing QK/RoPE fusion. |
| [FA3 backend](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/layers/attention/flashattention_backend.py) | Query cast, cache handoff and eligible model paths. Read during later A/B implementation reviews; it was an explicit gap in the initial proposal. |
| [SGLang's attention dependency pin](https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/kernels/aot/CMakeLists.txt) | Pins `sgl-project/sgl-attn` to `f89bc2306632d1ec5f97b014dded4254f5b4a907`. Source provenance is distinct from proving which binary ran on the judge. |

The ordering review followed five files at the **sgl-attn `f89bc230`** pin: [API](https://raw.githubusercontent.com/sgl-project/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_api.cpp), [forward launcher](https://raw.githubusercontent.com/sgl-project/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_fwd_launch_template.h), [SM90 forward kernel](https://raw.githubusercontent.com/sgl-project/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_fwd_kernel_sm90.h), [scheduler preparation](https://raw.githubusercontent.com/sgl-project/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_prepare_scheduler.cu), and [combine launcher](https://raw.githubusercontent.com/sgl-project/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/hopper/flash_fwd_combine_launch_template.h). The project preserved full hashes and verified downloads. This was a source-level ordering argument; compilation alone was not an H200 execution test.

The [archived feasibility review](evidence/fa3-feasibility-review.md) retains its original wording. Its COUNT discussion was later corrected: the verify counts overlap the prefill counts, giving **17 unique captured counts and 380 remaining eligible counts**, not 26 and 371. COUNT specialization was not changed in the final patch. Its pending-validation statements describe when it was written, before the final repaired run completed.

## Direct prior art & public experiment reports {#prior-art}

| Author / source | Revision and reading scope | Relationship to this project |
| :--- | :--- | :--- |
| **georgxzh**, [Muse technical write-up](https://github.com/georgxzh/muse-kernel-challenge/blob/890fa976c031b2433a5826d02c9355c25fa21efe/README.md) | `890fa976`; initial register: almost all the README, excluding the final rule-change-table tail | Marlin-preserving scheduling and large-prefill hypotheses. Mechanism attribution, not copied code or an independently reproduced serving result. |
| **alygg77**, [results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/RESULTS.md), [H200 measured facts](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/H200_MEASURED_FACTS.md) | `6cae49a6`; README, results and measured-facts report read in the initial proposal | Model-specific component bottleneck evidence and counterarguments to custom kernels. Single-GPU kernel-track measurements are not distributed serving measurements. |
| **alygg77**, [Negative results](https://github.com/alygg77/okc-muse-spark-kernels/blob/6cae49a639ee6d14b929a5fd4a63c627f96d7dc0/docs/NEGATIVE_RESULTS.md) | Same pin; initially unread, then read completely in the November 16 X supplement | Scoped warnings about overlap, prefetching, allocation layout, selective adoption and misleading proxies. No universal prohibition on those techniques. |
| **hariappi**, [kernel/serving README](https://github.com/hariappi/okc-muse-spark-kernels/blob/a3ebb0f2f027f07f38b063c6a74ac355a6b38848/README.md) | `a3ebb0f2`; README, serving patch, block-size/sweep data and selected routing lines | A simpler block-size alternative and comparison data. The accepted adaptive candidate is not the earlier one-line fix. |
| **GEMM People / Jerry2423**, [MoE Kernel Agent](https://github.com/Jerry2423/MoE-Kernel-Agent/tree/86c1e3f96d6478c3feed5b4f04f76810446618b8) | `86c1e3f9`; the four-page companion report in Nice Solutions was read; initial audit did not read the solution code | The audit explicitly credits route/dispatch fusion as a transferred principle. Their B200/DeepSeek implementation and speedups were not transplanted into this H200/NVFP4 result. |
| **NKI team 7013 / charleshong3**, [report and repository](https://github.com/charleshong3/nki-moe-7013/tree/751a7bf592a319600f6de679f354462eec74d704) | `751a7bf5`; report abstract, not a complete code audit | The audit records a warning about intermediate BF16 rounding boundaries as a constraint for the earlier prefill investigation. It did not diagnose the later Triton bug. |
| **Nathen Fernandes**, [dense-projection README](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75e6aca72511bf9b41b6dc190bf1e763373/README.md), [result](https://github.com/NathenFernandes/muse-bf16-proj/blob/86aff75e6aca72511bf9b41b6dc190bf1e763373/results/v18.json) | `86aff75e`; supplement: full README and result metadata/timing entries, not the full expanded implementation | Lossless packing and selective kernel reuse were considered. No dense-projection replacement was included in the final patch. |
| **operatorstack / bigboateng**, [MoE](https://github.com/operatorstack/open-kernel-challenge-solutions/blob/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_moe_w4a16/README.md), [dense projection](https://github.com/operatorstack/open-kernel-challenge-solutions/blob/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_bf16_proj/README.md), [attention](https://github.com/operatorstack/open-kernel-challenge-solutions/blob/6d9b77656b6c295e421be9e1424b0f8f99299b0a/muse_attention_decode/README.md) | `6d9b7765`; component READMEs and result extracts reviewed in the supplement | Hopper routing, layouts and fusion references. No claim that the complete foreign implementation was audited or imported. |
| **thethings.ai**, [dense-projection experiment report](https://thethings.ai/p/okc/muse-bf16-proj-report) | November 16 supplement: six-round report, caveats and artifact list read | Useful measurement and graph/workspace-lifetime lessons. Listed local artifacts were not verified as a public source package. |
| **vjeux**, [public attention implementation](https://github.com/vjeux/okc-2026/tree/f35a41ee1c13e316dabbf772830d47b23b453393) | `f35a41ee`; later attention-portability review examined the released CUDA implementation and related alygg77/hariappi variants | Candidate for a much larger port, deliberately not implemented. None of its kernel-board speedups are claimed for this serving patch. |

## X posts, blogs & discovery leads {#social-trail}

The November 16 supplement used indexed X mirrors because direct X access returned HTTP 403. A citation refresh still could not retrieve the Rohan Arun post directly. Mirrors establish a **discovery trail**, not an authenticated full thread. No tweet quotation, guessed post ID, current ranking or social speedup claim is used as proof in the article.

| Discovery trail | Primary material followed | Treatment |
| :--- | :--- | :--- |
| **@pranjalssh / @lmsysorg**, November 11; saved [indexed author profile](https://www.teapawt.net/pranjalssh) | Pranjal Shankhdhar / LMSYS, [Muse Spark Open Ultrafast: Pushing beyond 1,000 Tokens per second](https://www.lmsys.org/blog/2026-11-05-muse-spark-open-ultrafast); [Meta repository at `0775fc62`](https://github.com/facebookresearch/sgl-muse-mega/tree/0775fc62f768a781b74114afb6b4f075d31f73f5) | Full article, README and kernel-map README reviewed in the supplement. GB300/SM103 megakernel design was a mechanism reference, not a compatible H200 drop-in. |
| **@NathenFernandes**, October 24; saved [indexed profile](https://www.sotwe.com/NathenFernandes?lang=en) | Pinned projection README and published result linked above | Attribution follows through to inspectable artifacts. The historical ranking claim is not repeated as a win. |
| **@alspinu**, October 23–25; saved [indexed profile](https://teapawt.org/alspinu) | No attributable implementation verified in that pass | Screened lead only; no technical claim in this article relies on it. |
| **Rohan Arun**, [original X post](https://x.com/RohanArun/status/2114830856393027974) | [Auto-research-agent PR #6](https://github.com/rohanarun/Auto-research-agent/pull/6) | Supplement reviewed the PR description/merge metadata, not the full harness. Process reference; no measured contribution to the final candidate. |
| **@ryanli / @zhimao_liu**, November 12; saved [indexed announcement](https://nitter.rdrama.net/ryanli) | A proposed compute sponsorship | Availability was not verified and no sponsorship was obtained through this lead. Not performance evidence. |

Two other screened leads were [Pareton](https://github.com/Pareton-ai/pareton), whose separate campaign claims were not verified as OKC serving results, and [Archeon's challenge report](https://www.archeon.solutions/projects/open-kernel-challenge), which did not supply a new reproducible patch for this investigation. Neither contributes a speedup claim to the article.

## Other repository families assessed {#screened}

These entries complete the relevant repository trail without implying adoption. The initial Nice Solutions audit was a **report-level survey**, not a full implementation review. Prefixes identify the recorded revisions; the original origin/revision inventory is hashed in the evidence pack.

| Family / source | Recorded scope | What transferred, if anything |
| :--- | :--- | :--- |
| [Mixture of Kittens](https://github.com/cursor/mixture-of-kittens/tree/f3a8bd2432457c5a2a2ea13915f0dcd0224581e4) | README sections and beginnings of forward/scheduler source; paper, full runtime and missing submodule not read | Scheduling comparison only; Blackwell implementation not ported. |
| [DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) | README, deployment Q&A and outlines of policy/performance-cache code | Expert-load balancing and scheduling comparison; no runtime transplant. |
| [KernelEvolve / QasimKhan5d](https://github.com/QasimKhan5d/fused-moe/tree/338b470) | Three-page report | Per-expert work sizing; different hardware and arithmetic. |
| [Team Wombat / cheshire](https://github.com/cheshire/flashinfer-challenge/tree/e00c834) | Six-page report, nearly complete | Regime-specific scheduling and synchronization caution; no new barrier adopted. |
| [LLM-CUDA / syhya](https://github.com/syhya/mlsys26-flashinfer-contest/tree/e82b041) | Abstracts/introductions of two reports | Evaluator and experiment-process ideas. |
| [HAN Lab Kernel Mafia](https://github.com/mit-han-lab/mlsys2026-flashinfer-contest/tree/d61573f) | Abstract/introduction | Agent/verifier workflow background only. |
| [Insider / MayankSuthar1](https://github.com/MayankSuthar1/mlsys_2026_contest/tree/4f40b17) | Six-page report | GPU block maps, dispatch keys and accumulator initialization. |
| [UW fully agent](https://github.com/kamahori/mlsys-contest-syfi-fully-agent/tree/a4addd9) | Abstract | Workflow background. |
| [UW agent-assisted](https://github.com/kamahori/mlsys-contest-syfi-agent-assisted/tree/96aed88) | Abstract | Numerical-contract caution. |
| [Dogacel auto-kernel](https://github.com/Dogacel/auto-gpu-kernel/tree/db02153) and [sparse attention](https://github.com/Dogacel/DeepSeek-Sparse-Attention-Kernels/tree/9734cf8) | Abstracts | Sparse-attention mechanisms were not treated as valid dense-Muse shortcuts. |
| [Cong / learn-cutedsl](https://github.com/luongthecong123/learn-cutedsl/tree/5fa97b2) | First report page | CuTe/layout background; not a consumed complete course. |
| [Wombat sparse attention](https://github.com/ykirpichev/dsa_sparse_attention_h16_ckv512_kpe64_topk2048_ps64/tree/e33a4bf) and [indexer](https://github.com/ykirpichev/dsa_topk_indexer_fp8_h64_d128_topk2048_ps64/tree/54935bb) | Abstracts | No sparse/indexer algorithm adopted. |
| [Kachua](https://github.com/romitjain/kachua-mlsys/tree/34fac5f) | First report page | Different recurrence; background only. |
| [THU Storage NKI-MoE](https://github.com/thustorage/NKI-MoE/tree/217759f) | Abstract/problem setting | End-to-end MoE perspective, not a CUDA port. |
| NKI team 9376 archive | Abstract; canonical public origin not established | No invented URL or implementation attribution. |
| [Wenqi graph scheduler](https://github.com/wenqi-deng/mlsys-2026-graph-scheduler/tree/dde1a9f) and [Ryang graph scheduler](https://github.com/sohnryang/mlsys2026-graph-scheduling/tree/3c46795) | First report pages | Abstract cost-model / retain-versus-recompute background, not measured serving evidence. |
| [Huawei VNRC](https://github.com/huawei-csl/mlsys-2026-graph-scheduling-vnrc-submission/tree/1f19c2f) | Inventoried, unread | No influence claimed. |

GEMM People and NKI 7013 are credited separately under direct prior art because the audit explicitly records transferred principles. Listing a repository here does not mean its code is redistributed or endorsed as compatible.

## Documentation, lectures & teaching references {#learning}

| Material | Specific use in this article |
| :--- | :--- |
| NVIDIA, [Floating Point and IEEE 754 Compliance](https://docs.nvidia.com/cuda/floating-point/index.html), especially “Operations and Accuracy” and “Example: Dot Product” | Background for non-associativity and reduction order. The actual bug evidence is the preserved project probe, not this general white paper. |
| NVIDIA, [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) | Producer/consumer ordering concepts; paired with the pinned FA3 launchers rather than treated as proof of a particular call path. |
| [Triton fused-softmax tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) | Teaching example for avoiding intermediate traffic and launches. No softmax kernel was copied into the patch. |
| Si Böhm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM) (2022) | Tiling/performance intuition and a presentation reference. The prose and diagrams here are original. |
| Aleksa Gordić, [matmul article](https://www.aleksagordic.com/blog/matmul) | Visual teaching and structure reference for the November 22 article redesign. Not presented as a source consulted during the submitted implementation. No diagram or prose copied. |
| CMU 15-442/642, [GPU Architecture & CUDA Programming slides](https://mlsyscourse.org/slides/06-CUDA-programming.pdf) | Suggested thread/warp/reduction background. Not a claim that the complete course or lecture recording was consumed. |
| CMU 15-442/642, [Data Layouts](https://mlsyscourse.org/slides/data_layout/) | Suggested follow-up for logical-to-thread mappings. |
| CMU 15-442/642, [LLM Serving, Part 1](https://mlsyscourse.org/slides/14-LLM-serving-part1.pdf) | Suggested batching/cache background. |
| CMU 15-779, [Warp Specialization slides](https://www.cs.cmu.edu/~zhihaoj2/15-779/slides/06-warp-specialization.pdf) | Further study; this patch did not implement warp specialization. |
| humans&, [GPU-accelerated Weight Teleportation](https://humansand.ai/blog/gpu-accelerated-weight-teleportation) | Separate RL systems example and presentation reference. No compression or training-to-rollout weight-sync system was implemented here. |

The PyTorch FP8 conversion claim is version-specific: the independent A review inspected installed **PyTorch 2.14.1**, `torch/headeronly/util/Float8_e4m3fn.h`, `fp8e4m3fn_from_fp32_value`, and the project tested actual conversions. Its header hash and correction are preserved in the [archived A review](evidence/variant-a-review.md). A generic current PyTorch documentation page would not establish those exact bytes. That historical review's “no blocker” conclusion was subsequently superseded by the boundary failures and repair; only its conversion-source evidence is cited here.

The longer preparation catalog also listed vLLM, Orca, Sarathi-Serve, NanoFlow, DistServe, GPU MODE, Harvard/MIT courses, MLC and other resources. Being present in a catalog is not evidence of use. They are not presented as sources of this implementation or as completed study in this article.

## Project evidence & citation scope {#evidence}

- [Claim-to-artifact map](evidence/CLAIM_MAP.md): which saved result or probe supports each project claim.
- [Source inventory](evidence/source-inventory.json): original project paths, sizes and full SHA256 hashes, including the initial reading register, later addendum and implementation reviews.
- [Best completed evaluation](evidence/fused-result.md) and [final repaired evaluation](evidence/repaired-result.md): the source of the article's reported serving ratios.
- [Repair source/local review](evidence/repair-review.md), [independent dense-sweep addendum](evidence/dense-sweep-review.md), and [repair delta](evidence/repair-delta.patch): final numerical-repair evidence.
- [Initial mapping patch](evidence/mapping.patch): the first global-to-local GPU route lookup.
- [Final-selection extract](evidence/submission-decision.json) and [submission technical note](evidence/submission-method.md): what was frozen, what remained queued and which risks were recorded before the scored run. These are historical decision records, not current status.
- [Chart data](evidence/results.csv): the plotted values, with no invented uncertainty estimates.

All conceptual diagrams are original; editable Excalidraw scenes accompany them. No public benchmark table is relabeled as this project's result. Public reports are credited for the ideas they supplied, while unsuccessful, unimplemented and unread branches remain identified as such.

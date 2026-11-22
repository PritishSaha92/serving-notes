# Repaired-B final scored result — completed and passed

Owner review 2026-11-20T17:42:19.431865+00:00; official status fetched once at17:40:39UTC November20. Job **cfa908f197674235** finished 2026-11-20T10:55:18.422838+00:00. Status done, result status ok, passed true, aggregate **1.0224**. This is completed integrated H200 evidence for the exact candidate, replacing the earlier pending statement. No candidate, flags, entry, notes or monitor was changed.

Identity verified: candidate **adaptive32fused-qkvstore-repair-04ec69722720**; official patch SHA256 `04ec6972272094554f554bc864d08a872588e5e982acc8b1cbb87f447bef5fe2` equals frozen patch and original receipt. Frozen flags SHA256 `17501b5dcb9ff415c56c83f5d720521b47c4c7f59de9c49c1d9a7a4115d261c4` rehashed; content hash `b1907b064c96ad59089d34545fd8702710823c590681a020a5a0611c8e0097c5` matches accepted submission. Four source-file paths and pinned image match. The official result does not expose a separate flags-file SHA; flags consistency is established by the immutable local file/receipt/content identity and official stock-equivalent effective launch report (flags=[], launch_flags=[], denied=[], new_defaults=[], env_not_logged=false, unavailable=null). No mixed chunking or changed precision setting reported.

## Performance

| Users | Repair frontier ratio | Fused-stock ratio | Repair frontier tok/s/GPU | Paired stock tok/s/GPU |
|---:|---:|---:|---:|---:|
| 1 | 1.0318 | 1.0387 | 6710.8 | 6503.9 |
| 4 | 1.0097 | 1.0091 | 10061 | 9964.1 |
| 16 | 1.0217 | 1.0217 | 12584.5 | 12317.8 |
| 32 | 1.0265 | 1.0311 | 13032.1 | 12695.9 |

Aggregate is0.0027 below fused-stock1.0251, approximately0.2634% lower relative to that score. Ratios lower at1/32, slightly higher at4, identical at16 to reported precision. These are separate runs with their own stock fields, not a controlled paired comparison between candidates. No statistical significance or causal regression is established. No loads excluded; frontier paired_reps=3, TTFT ratio limit2.

| Users | Raw tok/s/GPU | Raw tok/s/user E2E | TTFT p90 first/later seconds | Reps | Reported spread | Later-turn pass |
|---:|---:|---:|---|---:|---:|---|
| 1 | 6505.9 | 143.3 | 8.971/0.224 | 3 | 0.069 | yes |
| 4 | 10061.6 | 56.2 | 21.38/0.254 | 3 | 0.0588 | yes |
| 16 | 12570 | 17.7 | 83.297/0.335 | 3 | 0.0007 | yes |
| 32 | 13032.1 | 9.3 | 184.25/3.041 | 1 | 0 | yes |

Raw rungs and interpolated frontier throughput are different returned measurement fields and are not substituted for one another. The32-user raw rung reports one repetition despite the frontier-level paired_reps=3. Its zero spread is not evidence of zero variability. Reported spreads are not confidence intervals. Against fused-stock raw rungs, first/later TTFT is8.971/.224 vs8.956/.225 at1;21.380/.254 vs21.383/.260 at4;83.297/.335 vs84.042/.346 at16;184.250/3.041 vs183.701/2.915 at32. Overall TTFT p90 at4 is.418 vs.330 and at32 is3.728 vs3.394: observed latency regressions, not statistically established causal effects. This experiment does not demonstrate an incremental aggregate gain over fused-stock.

## Correctness and later turns

All176 reference turns passed: prefill KL.054 (limit.11), decode KL.012 (limit.035), matching-prefix agreement.15 (minimum.05), prefill top1.81 (minimum.62), invalid0, short0, reasons[]. Fused-stock had.052/.012/.14/.83 respectively; both pass, no claim of equivalent population distributions.

Later-turn stock rescoring passed at every load:16conversations,45154servedpositions (3378first+41776later), first agreement.95, later.97, lead1.00 over192leadpositions. First/laterNLL.27/.20. No unmatched positions, insufficient sample or incomplete flag; reasons[]. Per-load first/later agreement and later positions:1=.96/.96/9924;4=.93/.97/10716;16=.96/.98/10758;32=.94/.97/10378. These satisfy the returned official gates. Original A/B remain locally blocked; this pass is not transferred to their different patches.

## Memory, settings and logs

Official memory is0.2GB/GPU below stock, within+2GB limit; fused-stock reported0.1GB below its stock. Derived prefill graph15.756vs15.791GB, target verify.248vs.254GB, decode graph.328vs.334GB; startup available20.719vs20.678GB. Full token capacity1891781vs1889034; SWA1513424vs1511227. These are the reported run's sizes, not a universal memory bound. Startup331.3s; later-turn rescore startup121.1s. No official liveness, memory or settings failure.

Read the complete available log, which contains a shortened startup excerpt and result summary, not a full unabridged runtime trace. Startup metadata says271lines,135797originalbytes, cut at firstrequest, truncated=true;299lines elided and server_args line shortened. It shows stock FA3/FP8-cache/TF32 defaults, successful multi-rank initialization, target/draft graph captures, KV allocation, startup and health requests. Target verify capture lists bs1,2,3,4,5,6,7,8,10,12,14,16,18,20,22,24,26,28,30,32 with draftwidth8. This proves capture occurred, not every runtime branch was exercised. Nonfatal notices: launch-entrypoint and server-info deprecations; NUMA host-memory policy unavailable so CPU-only binding; draft metadata glue graph disabled because backend declares no host-free draft prep. No exposed traceback/watchdog/OOM. Sandbox PermissionError/TimeoutError fields are isolation-probe outcomes, not candidate failures.

No per-kernel activation counts or profiler trace are returned. Exact patched H200 execution passed, but the fraction of requests taking fused preparation versus fallback and a causal fusion speedup cannot be established from these logs. General numerical/graph coverage remains bounded by local tests and this workload. No universal every-bucket guarantee; no confidence interval from historical runs.

## Disposition and artifacts

**Highest-scoring completed submission remains fused-stock68283766046b49f0,1.0251.** No promotion or artifact replacement. Preserved completed ranking: [{"job_id": "68283766046b49f0", "score": 1.0251}, {"job_id": "4ef0f823b01b406a", "score": 1.0247}, {"job_id": "ec8ff3db2e334729", "score": 1.0241}, {"job_id": "cfa908f197674235", "score": 1.0224}, {"job_id": "12d8a875dba14bc4", "score": 1.0149}]. Raw official result `research/results/h200/cfa908f197674235/status-20261120T174039Z.json`, SHA256 `912665cd6fba05ae225c233b6f3938ba0a16ab2ace837912dd4e4bc71a76ae71`; available log `research/results/h200/cfa908f197674235/log-20261120T174039Z.txt`, SHA256 `5b7ffff6d8cfd75cf8176d01ca63402b33d688eba8e4fd4674d8b20609c37f51`. Frozen bundles, selection/receipt and full reproduction archives unchanged.

Old monitor PID2450571 is no longer present; host process check found no matching watch_jobs.py process. No polling loop or replacement monitor started. Other check outcomes were outside this focused review and have not been freshly reconciled. No new request, withdrawal or notes mutation. Owner review complete; no automatic further action scheduled.

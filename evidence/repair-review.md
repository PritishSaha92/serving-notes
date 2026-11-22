# Minimal B repair — owner review, November20

Exact proposed patch SHA256: 04ec6972272094554f554bc864d08a872588e5e982acc8b1cbb87f447bef5fe2 (21,493bytes). Final QK source SHA256: 80384367c2a48b4e1e21df25cf9fbf693e27e68330aea95ae7034fe417950775. Stock flags17501b5dcb9ff415c56c83f5d720521b47c4c7f59de9c49c1d9a7a4115d261c4; overrides{}. Separate worktree; old A/B frozen candidates, failing fixtures and all official requests preserved and blocked unchanged. No mixed chunking, COUNT, precision-flag, logging or attention-port change.

## Cause and limits of the diagnosis

The54-run controlled matrix (`cause-v1.json`, `probe_cause.py`, hashedTTIR/TTGIR/PTX in cause-asm) uses thethree original minimizedfinitefixtures, independentstock/A/B productionkernels, BLOCK_M4/8/16 and contractionon/off. With the sameblock andcontraction setting eachcandidate agrees with appropriate stockquerydtype andK. Changingstockto theoldfusedblock reproduces differences; matchingstockdefaultblock restores allthree fixtures. Turningcontractionoff does notremove block-dependent differences and canintroduce extraRoPE differences fromstockdefault. No diagnosticcompileroption enters thecandidate.

Compiler evidence shows reductionlayout changes: BLOCK4 has sizePerThread[1,2]/threadsPerWarp[1,32]/warpsPerCTA[4,1]; BLOCK8 [1,4]/[2,16]/[4,1]; BLOCK16 [1,8]/[4,8]/[4,1]. PTX local square/reductiongrouping andshuffle widths change. Contractionpairings vary too, but are notnecessary for the observed mismatch. This supports layout-dependent finite-precision reduction/rounding as the cause for the original smallertile fixtures. Exact intermediate FP32 sums were not dumped; this is not a general proof of equivalence across compilers/hardware.

The first tiling-only repair deliberately tested the next boundary and FAILED: T3641stockBLOCK_M32 has finiteQ/K differences despite matchedtile. That v1source/patch, allwidefailures and two additionalfixtures are preserved. Exact compiler cause of this additionalfamily failure is not established. No claim that matchingBLOCK_M is universally sufficient. KNOWN_CANDIDATE_BLOCKERS.json now additionally rejects the tiling-only patch.

## One principled repair

For the FA3 FP8 path, select the tile from stock Q+K rows, while retaining Q+2K grid coverage for Vstorage. All arithmetic/contraction and conversion code stays unchanged. Declare the entire stockBLOCK_M32 compilationfamily unsupported using actualtokenmetadata andactualheadcounts. The model's existing eligibility branch then clearskv_store and uses ordinary QK plus backend cache/query preparation. The optimizedhelper rejects unsupported directcalls beforelaunch orcachewrite. This is a general compilation-family fallback, not a listof failingtoken counts or a workloadexception. Non-FA3, BF16, TRTLLM andHelion paths remainunchanged.

Independent review caught an unintended broad-edit Helion condition duringv2; it was removed beforefinalv3validation. Supersededv2source/results remainpreserved. The finaldiff fromfrozenB is only the tile selection, family eligibility and directhelper assertion.

## Final source validation

Allgates below ran on source803843... with unchanged independent pinnedstock reference; no tolerance/fixture/exception gate was relaxed.

- OriginalB410/noRoPE and416/RoPE finitefixtures: originalfrozenB reproduces oneFP8querybytefailureeach; repairedB exactQ/K/fullcache afterward.
- Boundarysuite:100eager and160changinggraph comparisons atT51/52/56/57/409/410/416/448/455/456. BothRoPE modes, stride2304, threefresh finite seeds plus adversarial andrawBF16bits, padding/untouchedslots/stableaddresses. PASS.
- Widertransition:24eager and32changinggraph cases atT3276/3277/3640/3641, bothRoPE, finite/adversarial/rawbits. Actualeligibility choosesfallback at3641. PASS. Oldtiling-onlysamefinitefixtures reproduce1Kbyte(nonRoPE),3Q/4K/1cacheKbyte(RoPE); newfallback exactallbytes. Failurefixtures retained.
- Production48bytecases; two65,536-pattern productionquery sweeps;72changinggraph cases;18synthetic62-layer/retainedbucket cases;18ordinaryfallback bytecases. AllPASS.
- Registered24tests andmodel/eligibility24tests PASS. Additionaltilefallback module runs35tests:24importedbaseline+11new. Eightnewactualmodelcases cross3640/3641, RoPE andfull/SWA cachelocations, eachwithfourchanginggraph replays includingallpadding; threeheadcountmetadatatests establish generalfamilyboundary. Unsupportedhelper raisesbeforecachewrites; modelusesBF16query andsave_kv_cache=True, thenactualquerypreparation andordinarycachewriter. Optimizedadjacentfamily emitsFP8 andsave_kv_cache=False.
- SixeligibleSM90PDL variants (BLOCK4/8/16×RoPE) compile. The earlierBLOCK32 compileis preserved asdiagnostic only; it is nowineligible. Compilation isnot execution. Registeredtests retainexistingunknownasyncio-optionwarning.

Seven-requestverify canreachT56; T416/448 andthirdboundary arekernel-level tests, notordinaryverify undermax32/draft8. No failure iswaived forunreachableworkloadcounts. Modeltests execute actualdispatch/querypreparation/cachewriter but a stubFA3consumer; noattentionmath. L40 graphreplay doesnot executeH200PDL, multirankcollectives orwholemodel. Syntheticpoolmemory isnot serverpeak/KVcapacity. Actualbinaryidentity andgraphdependencyedges remainunverified.

## Complete preparation timing

`timing-v3.json`: matchedL40 stock/blockedA/blockedB/repairedB; T8/52/56/128/256/410/416, bothRoPE, samehotbuffers,100calls/capturedgraph,20replays/sample,sevenalternatingmeasurements. CoversQK/RoPE, KVcast/store andquerypreparation. Repair saves4.56–5.21us/layer versusstock inthismicrobenchmark. Matchedstocktiling costsabout0.06–0.28us versusblockedB atchangedboundaries. OldA/B aretimingcontrols, noteligiblecandidates.

`fallback-timing-v3.json`: T3640/3641bothRoPE, stockandrepair only. At3641 stock17.1785/19.8922us, repair17.1858/19.8902us(noRoPE/RoPE). This supports no material GPU preparationoverhead atthestockfallback inthissample, notequivalenceofCPUdispatchcost. Timings excludeprojection, FA3, whole-server, CPUeligibilitychecks andH200. No scoreprojection or statisticalconfidenceclaim. Cachefusion has concretepotential tosave launches/temporaryconversions; its end-to-end gain remainsunmeasured.

## Ordering and integration

The repair doesnot alter B's ordering/handoff: exactstockBF16-rounding thenFP8 conversion is fused intoQ; K remainsBF16; ordinarycachebytes preserved. Reuse prior independentlyreviewed pinnedFA3 producer/consumer ordering argument andactualmodel/fallback tests. H200PDL andofficialFA3binary execution remain missing. OldA/B officialpasses, ifreturned, cannot validate thischangedpatch. Sharedmechanism failures require review before selection. A missingcheck is anexplicitrisk; anunresolvedhardfailure blocksselection.

Focused independentreview: INDEPENDENT_REVIEW.md (separate reviewer, exactpatch). Freeze onlyafteritsfinal assessment andallv3gates arechecked. Finalselection03:15UTC/target03:30/cutoff04 unchanged; preservefusedstock1.0251 regardlessofexperimentoutcome.

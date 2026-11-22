# Completed fused result review — November 18, 2026

Owner refreshed all14 jobs at11:24:08UTC. This replaces the missed01:30 planned review; no automatic owner review occurred in the gap. Public contract/updates unchanged. DeadlineNov20 08UTC, quota0scored/1check, resetNov19 00UTC. Three requests remainqueued: mixed32 73d99c440c874622, draft4 11c16403f6fd40dc, prefixadaptive e914b209d6cf4c39. None has a result to evaluate. Prefixfamily stayswithheld.

## Identity and review scope

Fusedscored68283766046b49f0 and fused32check7fc753bca4d447e0 are bothDONE/PASSED. Both returned patchSHA a2d84a8290b0c7d3a56c554bafb08643a8671a3486b1f2e287ed49db23677f6c matches frozen adaptive32fused-a2d84a8290b0. FrozenstockflagsSHA17501b5dcb9ff415c56c83f5d720521b47c4c7f59de9c49c1d9a7a4115d261c4, overrides{}. Returnedcontenthashes match their separate receipts: scored157d98e9a905616e39a20fc41a65e7ea2b331efaaf459933c7beb7989466e00f andcheck6f9c3487e19337ab0f5ff108a3b7bcaccb43bbee48788b534fd77090c4086c77. Pinnedimagef832b5c3 and Muse NVFP4 weights match. Resultpatchfiles lists only muse_spark_fp4_moe.py. No flags, denied settings, new defaults or unavailable setting evidence. Derivedgraph/capacity differences only. All returned structured fields and all available startup/CLI text reviewed; CLI startup content matches JSON, adds no full log.

The scored pre-request comparison/risks in risk-round/selection-fused-20261118T0002Z.json and FUSED_NEXT_DECISION.md were reconciled with these results. No numerical/precision/watchdog or promotion standard changed. This is the exact child, not inheritance from parent validation and not a mixed/draft4 combination.

## Correctness, later turns and resources

Scored:176reference turns, prefillKL.052<=.11, decodeKL.012<=.035, top1.83>=.62, matchingprefix.14>=.05, invalid0/short0, no reasons. Later47232servedpositions:3748first/43484later, agreement.96/.97, NLL.25/.16, lead192positions agreement1. Allfourper-rungs checked/passed; laterpositions10963/11292/10132/11097 andlateragreements.97/.97/.97/.98. Firstagreements.96/.96/.96/.95. incomplete/insufficientfalse, unmatched0. Per-rung first counts/NLL are not returned individually; aggregateandofficialper-rungpass preserved, noinventedvalues.

Check32:32references, KL.050/.015, top1.82, prefix.17, invalid0/short0. Later28323servedpositions:1848first/26475later, agreement.96/.97,NLL.26/.17,lead120agreement1. No incomplete/insufficient/unmatched, reasons[]. per_rung=[] for thischeck; don't manufacture separatecoverage.

Scoredmemory-.1GB/GPU andcheck-.2GB within+2GB. Bothderivedtargetverify.252vs.254GB,draftdecode.078vs.080,prefill15.756vs15.791,startupavailable20.717vs20.678, token_capacity1891651vs1889034,SWAcapacity1513320vs1511227. These are actualwhole-serverreportedvalues, not L40extrapolations. No scoredexclusions, shortoutputs0, allrungtruncatedfalse.

## Performance and comparison

| Users | Parent frontier ratio | Fused frontier ratio | Fused raw tok/s/GPU | Fused first/later TTFT seconds | Raw reps / spread |
|---|---:|---:|---:|---|---|
|1|1.0338|1.0387|6664.3|8.956 / .225|3 / .0408|
|4|1.0183|1.0091|10061|21.383 / .260|3 / .0207|
|16|1.0149|1.0217|12507.3|84.042 / .346|3 / .005|
|32|1.0319|1.0311|13067.8|183.701 / 2.915|1 / 0|

Fusedaggregate1.0251 versusparent1.0247, nominaldifference.0004 (~.039%). This is highestcompletedrecordedscore, not statisticalsuperiority oruniformdominance. Parentandchild are different runs/nodes; one32repwithspread0 is not zerovariance. Scoredfrontierpaired_reps3 is a top-levelfield while raw32has1rep; bothretained, do not rewriteorconflate. Frontierthroughputs6679.9/10080.2/12529.3/13067.8 againststock6431/9989.5/12262.8/12673.4. Rawrungs differfromfrontierinteractivitypoints. AllTTFTgatespassed/noexclusions, butscoredJSONdoesnotexposerawstockTTFTforeachrung; can'tindependentlyrecalculatethoseratios.

Separatecheck32: at_stock_point1.031(indicative,notaggregate); raw13047.7vs12653.2tok/s/GPU,9.3vs9.0/user; firstTTFT183.942vs184.482,later3.061vs3.332; returnedratios1.031throughput/1.033peruser/1.00first/.92later. One repetition,spread0,acceptance4.2,cache.92. Parentcheck32at_stock_point1.022,raw12896vs12620.3,TTFT186.342/3.058vs185.18/3.440 iscontextonly. Fusedcheckandscore32 are consistentnominally butnotcontrolledreplicates. Aggregate1.0251mustnotbecomparedtoindicative1.031 as equivalentstatistics.

## Startup and limitations

Scoredstartup331.3s,pod2898.4s; checkstartup321.3s,pod1009.8s. Laterturnvalidationseparatestartup111.1/141.2s. Engine scoredload23.80s,scheduler284.29s,tokenizer301.97s,graphprefill81.06/verify5.91/draft3.91s; checkload19.81,scheduler276.35,tokenizer294.07,prefill81.44/verify6.03/draft3.84. Not a controlledstartupcomparison. Target/draftfullgraphs use8tokens/request,20batchsizes1..32 listed; allrankscompletecapture/KVsizing. Graphborrow8376MiBhighwater/10062.3available,8374precarved; samplingprewarm8780775424B within8987672576Bheadroom. TargetKV23.96GB,draftKV4.51GB, hostpoolallocations48.48+111.52+30.30GBperrank arevisible, nozeroresourcecostclaim.

Availablelogsreportstandardlaunchdeprecation,NUMAmemorypolicyCPUonlybinding,defaultTF32/fp8KV/fa3,tokenizerreloadandserverinfoendpointdeprecation. Draftmetadatagluegraphdisabledbecausebackenddeclaresnohost-freeprep. These are not changedprecisionflags ordemonstratedfailures. No traceback/watchdog/OOM ispresentinreturnedtext; absenceinshortenedlogsdoesnotproveabsenceelsewhere. Providerstartupmetadata135207/135497bytes,270/269lines,truncatedtrue,300/302lineselided; server_argsitselfclipped. Entirepost-startupper-requestlogsarenotavailable. Settingscompliancecomesfromstructuredlaunchcomparison, notreadingmissingserverargs. Warmupchatdoesnotestablishbenchmarkendpoint. Capturedvariantwarmingobserved; uncapturedCOUNTcompilationfrequency/costintrafficunmeasured, noCOUNTchangejustified.

## Decision and next check

Promote exactcompletedfusedstockflagsbundle as localpreferredreadyentry; preserveparent1.0247 andearliermapping1.0149/1.0088withfullbundles. This performs noofficialrequest/withdrawal/final-selectionoperation. Rootwillmatchfusedbyte-for-byteafterguardedbackupandhashverification. Thequeuedmixed32bundle remainsparent-based766f1a86..., not silentlyrebased.

Do not spendfinalchecknow. Fusedall-loadresultalreadyvalidates16M48; nofailureornewinformationneedjustifiesunchanged16. Highestremainingdecisionisparent+mixedall-loadvalue, pendingits32check. Ifpassing32iscrediblebutlow-loadregressionisdecisioncritical, fouruserscouldbeinformativebecausehistoricalmixedregressedthere; definecriteriaandtimingafteritsactualresult, notautomaticsweep. Meaningfulpositivedraft4resultcanreopenit; weak/netregressingorfailingresultclosesdirectionunderexistingcriteria. Prefixcombinedpasscannotrepairunexplainedhang. No combinationofindividuallypassingchangeswithoutitsowninteractionvalidation/freeze. No checksneededjusttospendquota; noimprovementinferredfromqueuedwork.

NextownerreviewNov18 12:00UTC orimmediatelyonresumeifoverdue. Noautomatedreview/selection. Nov19slotonlyactivatesafterfreshUTCreset/quota/jobs; earlyresult-drivenselectionpermitted, latest12decision/14hard. FinalpartialwindowNov20 00–08UTC,hard04marginpreserved. Allpendingjobsretained.

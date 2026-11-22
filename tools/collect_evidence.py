"""Copy a small, explicit allowlist of reports; never package raw logs or credentials."""
from pathlib import Path
import hashlib, json, csv, re
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT.parent/'serving'
FILES={
 'mapping.patch':'builder-artifacts/candidates/mapping-e8e1a61c6e0d/patch.diff',
 'submission-method.md':'builder-artifacts/final-submit-20261120/TECHNICAL_NOTE_REPAIR.md',
 'fused-result.md':'builder-artifacts/fused-results-20261118/FUSED_COMPLETED_REVIEW.md',
 'repaired-result.md':'builder-artifacts/final-submit-20261120/REPAIRED_B_COMPLETED_REVIEW.md',
 'repair-review.md':'builder-artifacts/variant-repair/SOURCE_AND_LOCAL_REVIEW.md',
 'dense-sweep-review.md':'builder-artifacts/variant-repair/INDEPENDENT_SWEEP_ADDENDUM.md',
 'repair-delta.patch':'builder-artifacts/variant-repair/delta-from-blocked-B.patch',
 'variant-a-review.md':'research/final-opportunity-audit/VARIANT_A_INDEPENDENT_REVIEW.md',
 'fa3-feasibility-review.md':'research/final-opportunity-audit/VARIANT_B_AND_COUNT_FEASIBILITY.md',
}
AUDIT=[
 'builder-artifacts/final-submit-20261120/ACTUAL_FINAL_SELECTION.json',
 'CURRENT_RESUME.md','research/CONTRACT.md','research/SOURCE_REGISTER.md','research/NICE_SOLUTIONS_AUDIT.md',
 'X_RESEARCH_ADDENDUM.md','research/REPOSITORIES.json',
 'research/final-opportunity-audit/ATTENTION_PORTABILITY_REVIEW.md',
 'research/references/sgl-attn/f89bc2306632d1ec5f97b014dded4254f5b4a907/PROVENANCE.json',
 'research/R1_LIFECYCLE_REVIEW.md',
 'research/ROUND2_C1_DIAGNOSIS.md','research/C3E_REVIEW.md',
 'research/parallel-round3/R1_INDEPENDENT_REVIEW.md',
 'research/prefix-liveness/PASS_RESULT_REVIEW.md',
 'research/final-opportunity-audit/VARIANT_B_IMPLEMENTATION_REVIEW.md',
 'builder-artifacts/variant-repair/INDEPENDENT_REVIEW.md',
 'builder-artifacts/variant-repair/probe_dense_sweep.py',
 'builder-artifacts/variant-repair/qk-tiling-only-blocked.py',
 'builder-artifacts/fused-mixed-final/FINAL_COMPARISON_AFTER_PARENT_SCORE.md',
 'builder-artifacts/candidates/adaptive32fused-a2d84a8290b0/patch.diff',
 'builder-artifacts/candidates/adaptive32fused-qkvstore-repair-04ec69722720/patch.diff',
 'builder-artifacts/final-submit-20261120/TECHNICAL_NOTE_REPAIR.md',
]
records=[]
for relative in dict.fromkeys(list(FILES.values())+AUDIT):
    p=SRC/relative
    if not p.is_file(): raise RuntimeError('Missing evidence: '+relative)
    raw=p.read_bytes()
    if not raw: raise RuntimeError('Empty evidence: '+relative)
    records.append({'source':'serving/'+relative,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
for dest,relative in FILES.items():
    text=(SRC/relative).read_text(encoding='utf-8')
    # The final selected reports are plain research text, not account/config dumps.
    if re.search(r'Bearer\s+[A-Za-z0-9]|kc_[A-Za-z0-9]{12,}|ghp_[A-Za-z0-9]+',text):
        raise RuntimeError('Credential-like value in selected report')
    (ROOT/'evidence'/dest).write_text(text,encoding='utf-8')
(ROOT/'evidence'/'source-inventory.json').write_text(json.dumps({
    'scope':'Claim-bearing reports, complete final patches, targeted implementation/probe sources. Not a line-by-line audit of all vendored code.',
    'as_of':'2026-11-22','sources':records},indent=2),encoding='utf-8')
print('Verified and indexed',len(records),'evidence sources; copied',len(FILES),'allowlisted text artifacts.')

# A small explicitly selected extract, rather than a full launch/account dump.
selection_path=SRC/'builder-artifacts/final-submit-20261120/ACTUAL_FINAL_SELECTION.json'
selection=json.loads(selection_path.read_text(encoding='utf-8'))
checks=['1ca49f01cff9467a','f8b1e9877ac54ef9','70bd81d1439e4a56','3c9f2dbf76334fa3']
extract={
 'scope':'Editorial extract of the recorded selection; not a live job-status report.',
 'source':'serving/'+selection_path.relative_to(SRC).as_posix(),
 'source_sha256':hashlib.sha256(selection_path.read_bytes()).hexdigest(),
 **{key:selection[key] for key in ['utc','decision','candidate','patch_sha256','flags_sha256','cli_overrides']},
 'checks_at_selection':{job:{key:selection['reviewed_jobs'][job][key] for key in ['status','passed','score']} for job in checks},
}
(ROOT/'evidence/submission-decision.json').write_text(json.dumps(extract,indent=2)+'\n',encoding='utf-8')

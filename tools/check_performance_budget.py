"""Verify the article's conditional budget, not an H200 speedup prediction."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]

def main():
    record = json.loads((ROOT/'evidence/preparation-timings.json').read_text(encoding='utf-8'))
    rows = record['cases']
    assert {(r['T'], r['rope']) for r in rows} == {
        (t, rope) for t in (8, 52, 56, 128, 256, 410, 416) for rope in (False, True)
    }
    for row in rows:
        assert math.isclose(row['saving_us'], row['median_stock_us']-row['median_repair_us'], abs_tol=1e-9)
    lo, hi = min(r['saving_us'] for r in rows), max(r['saving_us'] for r in rows)
    assert (round(lo, 2), round(hi, 2)) == (4.56, 5.21)
    # Use the rounded numbers quoted in the article.
    milliseconds = [62*x/1000 for x in (4.56, 5.21)]
    fractions = [x/20 for x in milliseconds]
    speedups = [1/(1-x) for x in fractions]
    assert [round(x,3) for x in milliseconds] == [0.283,0.323]
    assert [round(100*x,2) for x in fractions] == [1.41,1.62]
    assert [round(x,3) for x in speedups] == [1.014,1.016]
    for source in record['sources']:
        path = ROOT.parent/source['path']
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source['sha256'], source['path']
    print('PASS: 14 timing cases and conditional 62-layer / illustrative 20-ms arithmetic; no H200 inference.')

if __name__ == '__main__':
    main()

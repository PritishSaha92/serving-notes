"""Package the standalone notebook, excluding local tooling state and previews."""
from pathlib import Path
import hashlib, json, zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'muse-serving-notebook-private.zip'
SKIP = {'node_modules', '.preview', '.git', '__pycache__'}

sources = json.loads((ROOT/'evidence/source-inventory.json').read_text(encoding='utf-8'))
for record in sources['sources']:
    source = ROOT.parent / record['source']
    if source.exists():
        assert hashlib.sha256(source.read_bytes()).hexdigest() == record['sha256'], record['source']

qa = json.loads((ROOT/'.preview/qa.json').read_text(encoding='utf-8'))
assert qa['status'] == 'passed'
(ROOT/'VERIFICATION.json').write_text(json.dumps(qa, indent=2)+'\n', encoding='utf-8')

files = sorted(p for p in ROOT.rglob('*') if p.is_file()
               and not any(part in SKIP for part in p.relative_to(ROOT).parts)
               and p.name != 'FILE_MANIFEST.json')
manifest = [{'path':p.relative_to(ROOT).as_posix(), 'bytes':p.stat().st_size,
             'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
(ROOT/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
files.append(ROOT/'FILE_MANIFEST.json')
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(OUT) as z:
    assert z.testzip() is None
    for item in manifest:
        data = z.read(ROOT.name+'/'+item['path'])
        assert hashlib.sha256(data).hexdigest() == item['sha256']
print(json.dumps({'archive':str(OUT),'files':len(files),'bytes':OUT.stat().st_size,
                  'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest()}))

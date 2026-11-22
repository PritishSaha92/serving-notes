"""Create static-upload and harmless bootstrap ZIPs; does not deploy anything."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
files = [ROOT / name for name in (
    'index.html', 'sources.html', 'article.md', 'sources.md', 'robots.txt',
    'LICENSE', 'THIRD_PARTY_NOTICES.md'
)]
for directory in ('assets', 'evidence', 'licenses'):
    files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file())
assert all(p.is_file() for p in files)
out = ROOT.parent / 'muse-serving-site.zip'
with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(files):
        archive.write(path, path.relative_to(ROOT).as_posix())
with zipfile.ZipFile(out) as archive:
    assert archive.testzip() is None
    assert 'index.html' in archive.namelist()
    for path in files:
        assert archive.read(path.relative_to(ROOT).as_posix()) == path.read_bytes()

placeholder = ROOT.parent / 'muse-serving-placeholder.zip'
with zipfile.ZipFile(placeholder, 'w', zipfile.ZIP_DEFLATED) as archive:
    archive.writestr('index.html', '<!doctype html><html lang="en"><meta charset="utf-8">'
                      '<meta name="robots" content="noindex,nofollow">'
                      '<title>Site setup</title><h1>Site setup</h1>'
                      '<p>No article has been uploaded.</p></html>')
    archive.writestr('robots.txt', 'User-agent: *\nDisallow: /\n')
for path in (out, placeholder):
    print(json.dumps({'archive': str(path), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}))

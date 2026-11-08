"""Build a compact submission without env, QA, caches or regenerateable floats."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDE={'.venv','.git','__pycache__','.pytest_cache'}
PREFIXES=['results/cache/','results/outputs/','demo/raw/','demo/voiceover/','report/qa/','report/qa_polished/','report/backup/']
def included(p):
    rel=p.relative_to(ROOT)
    if any(x in EXCLUDE for x in rel.parts):return False
    if any(rel.as_posix().startswith(x) for x in PREFIXES):return False
    if p.suffix in ['.tmp','.pyc','.webm'] or p.name.startswith('~$'):return False
    if p.name in ['交付清单.json','download_verified.zip','ui_initial.png','video_contact_sheet.jpg','render_log.json','analysis_console.json','encode_log.txt']:return False
    return True

def main():
    archive=ROOT.parent/'混合噪声去噪大作业提交包.zip'
    files=[p for p in sorted(ROOT.rglob('*')) if p.is_file() and included(p)]
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in files:z.write(p,Path('大作业成果')/p.relative_to(ROOT))
    # Reopen after the central directory and buffered bytes have been flushed.
    with zipfile.ZipFile(archive,'r') as z:
        assert z.testzip() is None
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    result={'archive':str(archive),'file_count':len(files),'bytes':archive.stat().st_size,'sha256':digest,
            'excluded':'virtual environment, git internals, exact float caches/outputs, QA and raw browser videos; retained in original directory'}
    (ROOT/'交付清单.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

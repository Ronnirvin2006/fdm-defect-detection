"""Consolidate verified release evidence and remove superseded generated documents only."""
import hashlib
import json
import re
from pathlib import Path
import shutil
import zipfile

import pymupdf

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"outputs/paper/IAME2026_Final"


def main():
    with pymupdf.open(PAPER/"IAME2026_Paper54_Camera_Ready.pdf") as pdf:
        assert len(pdf)==6
        text="\n".join(page.get_text() for page in pdf)
        assert all(word in text for word in ["83.83%","96.52%","TABLE VI","0.0635"])
    with zipfile.ZipFile(PAPER/"IAME2026_Paper54_Submission_Package.zip") as package:
        assert package.testzip() is None
    assert (ROOT/"models/historical_20260727.keras").exists()
    model=json.loads((ROOT/"models/deployment.json").read_text())
    assert hashlib.sha256((ROOT/"models/best_model.keras").read_bytes()).hexdigest()==model["sha256"]
    local=json.loads((ROOT/"logs/final_qa/validation.json").read_text())
    public=json.loads((ROOT/"logs/public_qa/validation.json").read_text())
    unit_log=(ROOT/"logs/release_unit_tests.log").read_text()
    count=re.search(r"Ran (\d+) tests",unit_log)
    assert count and "\nOK\n" in unit_log and "FAILED (" not in unit_log
    release={"date":"2026-10-29","model":model,"local_browser":local,"public_browser":public,"unit_tests_passed":int(count.group(1)),"physical_printer_validated":False,"real_android_ios_hardware_validated":False,"machine_control":"excluded by user","similarity_percentage":None,"similarity_note":"Requires an institution-approved external report; no under-10% certification.","manuscript":json.loads((PAPER/"validation.json").read_text())}
    (ROOT/"outputs/reports/release_validation.json").write_text(json.dumps(release,indent=2))
    # This is a deliberately closed list: never traverse into data/ or models/.
    obsolete=[
        "outputs/paper/review_revision",
        "outputs/paper/camera_ready_v1_2026-10-29",
        "outputs/paper/FDM_Defect_Detection_Conference_Paper.md",
        "outputs/paper/IAME2026_FDM_Defect_Detection_Full_Paper.docx",
        "outputs/paper/IAME2026_FDM_Defect_Detection_Full_Paper.pdf",
        "outputs/paper/SUBMISSION_CHECKLIST.md",
        "outputs/reports/project_completion_audit.md",
        "outputs/reports/camera_ready_and_video_status_2026-10-29.md",
        "outputs/reports/project_status_2026-10-29.md",
        "outputs/reports/final_project_summary.md",
    ]
    removed=[]
    for name in obsolete:
        path=ROOT/name
        if not path.exists():continue
        assert path.resolve().is_relative_to(ROOT/"outputs")
        size=sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.is_dir() else path.stat().st_size
        removed.append({"path":name,"bytes":size})
        if path.is_dir():shutil.rmtree(path)
        else:path.unlink()
    report={"removed":removed,"bytes_freed":sum(item["bytes"] for item in removed),"protected":["data/","models/","all training run evidence","latest IAME submission package","original supplied templates outside this folder"]}
    cleanup_file=ROOT/"outputs/reports/document_cleanup.json"
    if removed or not cleanup_file.exists():
        cleanup_file.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()

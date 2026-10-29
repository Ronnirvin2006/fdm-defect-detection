"""Select a completed continuation checkpoint and preserve the historical model."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
run = ROOT / "outputs/reports/local_20261029_183228"
metrics = json.loads((run / "metrics.json").read_text())
source = ROOT / metrics["model"]
names = json.loads((source.with_suffix(".classes.json")).read_text())
assert names == json.loads((ROOT / "models/class_names.json").read_text())
assert source.is_file()
active = ROOT / "models/best_model.keras"
historical = ROOT / "models/historical_20260727.keras"
if not historical.exists():
    shutil.copy2(active, historical)
temporary = ROOT / "models/selected.saving.keras"
shutil.copy2(source, temporary)
temporary.replace(active)
metadata = {
    "selected_checkpoint":str(source.relative_to(ROOT)),
    "sha256":hashlib.sha256(active.read_bytes()).hexdigest(),
    "historical_backup":str(historical.relative_to(ROOT)),
    "report":str(run.relative_to(ROOT)),
    "class_names":names,
    "grouped_validation_accuracy":metrics["validation_accuracy"],
    "evaluation_note":metrics["evaluation_note"],
    "decision":"Selected for demonstration after continuation training; independent physical-printer validation remains pending. Not a merged ensemble.",
}
(ROOT / "models/deployment.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))

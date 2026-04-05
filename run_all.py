"""
Nepal MICS6 ECDI2030 — Run Full Pipeline
==========================================
Run this file to execute all steps in order.
"""

import subprocess
import sys
from pathlib import Path

scripts = [
    ("01_data_loading.py",  "Step 1: Loading and merging MICS6 modules"),
    ("02_ecdi_scoring.py",  "Step 2: Computing ECDI2030 domain scores"),
    ("03_eda.py",           "Step 3: Exploratory data analysis"),
    ("04_supervised_ml.py", "Step 4: Supervised ML + SHAP"),
    ("05_clustering.py",    "Step 5: Unsupervised clustering"),
    ("06_geospatial.py",    "Step 6: Geospatial analysis"),
]

base = Path(__file__).parent

for script, label in scripts:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(base / script)],
                            capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] {script} failed with return code {result.returncode}")
        sys.exit(1)

print("\n" + "="*60)
print("  Pipeline complete.")
print(f"  Figures saved to: {base / 'figures'}")
print("="*60)

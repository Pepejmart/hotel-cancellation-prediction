"""Create the synthetic demo data on first run (data/ is not versioned)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
REQUIRED = ["df_unificado_completo_v8.csv", "reservas_2026_full.csv", "reservas_web_2026.csv", "precios_adr.json"]


def ensure_data():
    if all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED):
        return
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from generate_synthetic_data import generate
    generate(DATA_DIR, verbose=False)


ensure_data()

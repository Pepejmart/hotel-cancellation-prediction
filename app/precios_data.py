# =============================================================================
# PRECIOS_DATA.PY - Diccionario de precios ADR por habitación/PAX/semana
# =============================================================================
#
# ESTRUCTURA:
#   PRECIOS_ADR[hotel_code][habitacion][pax]["YYYY-WW"] = adr_por_persona
#
# La tabla original (derivada de los datos reales del cliente) no se
# publica. Se carga desde data/precios_adr.json, generado de forma sintética con:
#   python src/generate_synthetic_data.py
# =============================================================================
import json
import os

PRECIOS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "precios_adr.json"
)

try:
    with open(PRECIOS_PATH, encoding="utf-8") as f:
        PRECIOS_ADR = json.load(f)
except FileNotFoundError:
    print(f"[precios_data] {PRECIOS_PATH} no encontrado. Ejecuta: python src/generate_synthetic_data.py")
    PRECIOS_ADR = {}

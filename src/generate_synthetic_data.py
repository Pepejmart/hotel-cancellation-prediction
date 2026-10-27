#!/usr/bin/env python3
"""
Generate synthetic datasets with the same schema as the (confidential) project data.

The real client data cannot be published. This script creates
fully synthetic files so that the Streamlit app and the notebooks can run
end to end. No value is sampled from the real data: distributions are
hand-tuned approximations and the cancellation label is drawn from a simple
logistic rule, so model metrics on this data are NOT comparable with the
results reported in the README.

Outputs (written to ``data/`` by default):
    df_unificado_completo_v8.csv  Historical bookings 2022-2023 (training schema, 61 cols)
    reservas_2026_full.csv        2026 booking book scored by the model (app "Intranet")
    reservas_web_2026.csv         Bookings made through the demo web (app "Cliente")
    precios_adr.json              ADR table hotel -> room -> pax -> "YYYY-WW"

Usage:
    python src/generate_synthetic_data.py                 # default sizes
    python src/generate_synthetic_data.py --n-hist 50000 --n-2026 20000 --seed 7
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# Catalog (public product names: hotels, complexes and room types)
# -----------------------------------------------------------------------------
HOTELS = {
    # code: (hotel name, complex, base nightly price per person in USD)
    "CANA_BAV": ("Aurea Select Bavaro", "Complejo Punta Cana", 200),
    "CANA_PAL": ("Aurea Palace Resort & Spa", "Complejo Punta Cana", 190),
    "CANA_PC": ("Aurea Punta Cana Resort & Spa", "Complejo Punta Cana", 180),
    "CANA_SIG": ("Aurea Signature Turquesa", "Complejo Punta Cana", 290),
    "MAYA_COL": ("Aurea Hacienda Resort & Spa", "Complejo Riviera Maya", 175),
    "MAYA_KAN": ("Aurea Jungle Resort & Spa", "Complejo Riviera Maya", 190),
    "MAYA_SIG": ("Aurea Signature Yucatan", "Complejo Riviera Maya", 275),
    "MAYA_WS": ("Aurea Select White Sand", "Complejo Riviera Maya", 210),
    "MUJE_CMU": ("Aurea Select Costa Mujeres", "Complejo Costa Mujeres", 225),
    "MUJE_CMU_FS": ("Aurea Family Costa Mujeres", "Complejo Costa Mujeres", 325),
    "MUJE_SIGC": ("Aurea Signature Coral", "Complejo Costa Mujeres", 350),
}

ROOMS = {
    "CANA_BAV": ["BAV JUNIOR SUITE GV", "BAV JUNIOR SUITE BS", "BAV SUPERIOR JS GV", "BAV SUPERIOR JS BS",
                 "BAV PREMIUM JS GV", "BAV ROMANCE SUITE GV", "BAV ROOFTOP JT JS", "BAV AMBASSADOR SUITE POV"],
    "CANA_PAL": ["PAL DELUXE GARDEN VIEW", "PAL DELUXE BEACHSIDE", "PAL JUNIOR SUITE GV", "PAL JUNIOR SUITE BS",
                 "PAL JUNIOR SUITE SW BS", "PAL LOFT SUITE GV", "PAL LOFT SUITE BS"],
    "CANA_PC": ["PC DELUXE GARDEN VIEW", "PC DELUXE POOLSIDE", "PC JUNIOR SUITE GV", "PC JUNIOR SUITE POOLSIDE",
                "PC PREMIUM JS PS", "PC LOFT SUITE GV", "PC LOFT FAMILY SUITE"],
    "CANA_SIG": ["SIG JUNIOR SUITE GV/PV", "SIG JUNIOR SUITE PS", "SIG JUNIOR SUITE SWIM UP", "SIG ROMANCE SUITE PS",
                 "SIG VILLA SUITE POOLSIDE", "SIG AMBASSADOR SUITE"],
    "MAYA_COL": ["COL DELUXE GARDEN VIEW", "COL JUNIOR SUITE GV", "COL JUNIOR SUITE PS", "COL ROMANCE VILLA SUI PS",
                 "COL AMBASSADOR SUITE PS"],
    "MAYA_KAN": ["KAN DELUXE GARDEN VIEW", "KAN JUNIOR SUITE GV", "KAN FS JUNIOR SUITE GV", "KAN FS JS SWIM UP",
                 "KAN ROMANCE VILLA SUI BS", "KAN AMBASSADOR SUITE"],
    "MAYA_SIG": ["SIG JUNIOR SUITE GV", "SIG JUNIOR SUITE PS", "SIG JUNIOR SUITE PP GV", "SIG SUITE GARDEN VIEW",
                 "SIG SUITE POOLSIDE", "SIG ROMANCE BW BYTHE LAKE", "SIG AMBASSADOR SUITE POV"],
    "MAYA_WS": ["WS JUNIOR SUITE GV", "WS JUNIOR SUITE BS", "WS SUITE GARDEN VIEW", "WS SUITE BEACHSIDE",
                "WS ROMANCE BW BYTHE LAKE"],
    "MUJE_CMU": ["CMU JUNIOR SUITE GV", "CMU JUNIOR SUITE PS", "CMU JUNIOR SUITE BS", "CMU JUNIOR SUITE SW BS",
                 "CMU LOFT SUITE JT", "CMU AMBASSADOR ST BS", "CMU RESIDENCE SUI BS POV"],
    "MUJE_CMU_FS": ["CMU FS JUNIOR SUITE BS", "CMU FS JUNIOR SUITE SW BS", "CMU FS LOFT SUITE JT",
                    "CMU FS AMBASSADOR SUI BS"],
    "MUJE_SIGC": ["SIGC JUNIOR SUITE GV", "SIGC JUNIOR SUITE BS", "SIGC JUNIOR SUITE SW", "SIGC LOFT SUITE JT",
                  "SIGC AMBASSADOR ST BS OV", "SIGC SIGNATURE ST BS POV"],
}

HOTEL_WEIGHTS = {"MUJE_CMU": 15, "CANA_BAV": 14, "MUJE_SIGC": 13, "CANA_PC": 12, "MAYA_SIG": 10, "MAYA_COL": 10,
                 "CANA_SIG": 8, "CANA_PAL": 7, "MAYA_WS": 6, "MUJE_CMU_FS": 4, "MAYA_KAN": 2}

# Generic distribution channels (placeholders, not real partners)
CLIENTES = {
    "DIRECT SALES": ["WEB PROPIA", "CALL CENTER", "AUREA TRAVEL CLUB_SOCIOS", "MOTOR RESERVAS GLOBAL"],
    "E-COMMERCE": ["OTA EUROPE BV", "OTA AMERICAS INC", "OTA LATAM S.A.", "METABUSCADOR B2C"],
    "T.O. / T.A.": ["TOUR OPERADOR NORTEAMERICA", "TOUR OPERADOR ESPANA", "AGENCIA MAYORISTA MX",
                    "TOUR OPERADOR UK", "AGENCIA CANADA"],
    "CORPORATE": ["CUENTA CORPORATIVA A", "CUENTA CORPORATIVA B"],
    "OTHERS": ["OTROS CANALES"],
}
FUENTE_W = {"DIRECT SALES": 0.30, "E-COMMERCE": 0.28, "T.O. / T.A.": 0.36, "CORPORATE": 0.02, "OTHERS": 0.04}
SEGMENTOS = {"DIRECT SALES": ["BAR", "Loyalty", "Fixed Rates"], "E-COMMERCE": ["BAR", "Package"],
             "T.O. / T.A.": ["Fixed Rates", "Package", "Group Leisure"], "CORPORATE": ["Corporate"],
             "OTHERS": ["Others"]}
PAISES = ["ESTADOS UNIDOS", "CANADA", "MEXICO", "ESPAÑA", "ARGENTINA", "REINO UNIDO", "COLOMBIA", "BRASIL",
          "CHILE", "ALEMANIA", "FRANCIA", "ITALIA", "PERU", "SIN PAIS"]
PAIS_W = [0.38, 0.14, 0.12, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02]
FIDELIDADES = ["Aurea Rewards", "Partner Rewards", "Aurea Connect"]

NOCHES_P = {1: 3, 2: 12, 3: 12, 4: 14, 5: 11, 6: 7, 7: 26, 8: 4, 9: 3, 10: 3, 11: 1, 14: 3, 21: 1}
PAX_P = {1: 8, 2: 70, 3: 14, 4: 6.5, 5: 1, 6: 0.5}

HIST_COLUMNS = [
    "ID_RESERVA", "LLEGADA", "SALIDA", "NOCHES", "REGIMEN", "USO", "PAX", "ADULTOS", "NENES", "BEBES",
    "TIPO_CLIENTE", "CLIENTE", "MONEDA", "STATUS", "ID_MOTIVO", "SUPLETORIA", "CUNAS", "FECHA_TOMA", "FECHA_MOD",
    "FECHA_CANCELACION", "FIDELIDAD", "VALHAB", "VALPEN", "VALSERV", "VALFIJOS", "COMERCIALIZADORA", "CMVALHAB",
    "CMVALPEN", "CMCVALSERV", "GRATIS", "PAIS", "SEGMENTO", "FUENTE_NEGOCIO", "TIPO_NORMALIZADO", "NOMBRE_HOTEL",
    "NOMBRE_HABITACION", "Categoria", "NUMSOCIO", "CANCELADO", "VALOR_RESERVA", "ADR", "REV_PAX", "LEAD_TIME",
    "HORA_TOMA", "HORA_TOMA_SIN", "HORA_TOMA_COS", "MES_LLEGADA", "MES_LLEGADA_SIN", "MES_LLEGADA_COS",
    "TIPO_VIAJERO", "TIENE_FIDELIDAD", "LOYALTY_REWARDS", "ES_GRUPO", "PAIS_TOP_200",
    "FUENTE_NEGOCIO_SEGMENTO_CLIENTE", "PREFIJO_HAB", "NOMBRE_COMPLEJO", "HOTEL_COMPLEJO", "COMPLEJO_RESERVA",
    "HABITACION_TOP", "HOTEL_HABITACION_TOP",
]
FULL_2026_COLUMNS = [
    "ID_RESERVA", "LLEGADA", "SALIDA", "NOCHES", "PAX", "NOMBRE_HABITACION", "VALOR_RESERVA", "HOTEL_COMPLEJO",
    "NOMBRE_HOTEL_REAL", "COMPLEJO_REAL", "PROBABILIDAD_CANCELACION", "oferta_retencion", "fecha_envio_oferta",
    "estado_oferta",
]
WEB_COLUMNS = [
    "ID_RESERVA", "LLEGADA", "SALIDA", "NOCHES", "PAX", "VALOR_RESERVA", "NOMBRE_HABITACION", "CANAL", "MERCADO",
    "AGENCIA", "NOMBRE_HOTEL_REAL", "COMPLEJO_REAL", "PROBABILIDAD_CANCELACION", "HOTEL_COMPLEJO",
    "CLIENTE_NOMBRE", "CLIENTE_EMAIL", "CLIENTE_TELEFONO", "TELEGRAM_CHAT", "SEGMENTO", "FIDELIDAD",
    "FUENTE_NEGOCIO", "FECHA_CREACION", "ESTADO",
]


def _choice(rng, options, n, weights=None):
    keys = list(options)
    p = None if weights is None else np.asarray(weights, float) / np.sum(weights)
    return np.asarray(keys, dtype=object)[rng.choice(len(keys), size=n, p=p)]


def _unique_ids(rng, n):
    """12-digit unique booking ids."""
    return rng.permutation(10**11 + np.cumsum(rng.integers(1, 5000, n)))


def _room_factor(room):
    r = room.upper()
    if any(k in r for k in ("AMBASSADOR", "SIGNATURE", "RESIDENCE", "VILLA")):
        return 1.9
    if any(k in r for k in ("LOFT", "ROMANCE", "SW", "SWIM", "ROOFTOP", "PREMIUM")):
        return 1.4
    if "DELUXE" in r:
        return 0.85
    return 1.0


def _season_factor(month):
    # High season: winter + Easter + summer
    return 1.0 + 0.25 * np.cos(2 * np.pi * (np.asarray(month) - 2) / 12) + 0.08 * (np.asarray(month) == 7)


def build_price_table(rng):
    """PRECIOS_ADR[hotel][room][pax]['YYYY-WW'] = ADR per person (USD)."""
    table = {}
    for hotel, rooms in ROOMS.items():
        base = HOTELS[hotel][2]
        table[hotel] = {}
        for room in rooms:
            table[hotel][room] = {}
            rf = _room_factor(room) * rng.uniform(0.92, 1.08)
            for pax in range(1, 7):
                pf = {1: 1.6, 2: 1.0}.get(pax, 1.0 - 0.07 * (pax - 2))
                weeks = {}
                for year in range(2022, 2028):
                    infl = 1.03 ** (year - 2022)
                    for week in range(1, 53):
                        month = min(12, 1 + (week - 1) * 12 // 52)
                        adr = base * rf * pf * infl * _season_factor(month) * rng.normal(1, 0.03)
                        weeks[f"{year}-{week:02d}"] = round(float(adr), 2)
                table[hotel][room][str(pax)] = weeks
    # The app maps "Aurea Signature Coral" to the legacy code MUJE_SIG
    table["MUJE_SIG"] = table["MUJE_SIGC"]
    return table


def _bookings_core(rng, n, arrival_start, arrival_days):
    hotel = _choice(rng, HOTEL_WEIGHTS, n, list(HOTEL_WEIGHTS.values()))
    room = np.array([rng.choice(ROOMS[h]) for h in hotel], dtype=object)
    noches = _choice(rng, NOCHES_P, n, list(NOCHES_P.values())).astype(int)
    pax = _choice(rng, PAX_P, n, list(PAX_P.values())).astype(int)
    llegada = pd.to_datetime(arrival_start) + pd.to_timedelta(rng.integers(0, arrival_days, n), unit="D")
    salida = llegada + pd.to_timedelta(noches, unit="D")
    base = np.array([HOTELS[h][2] for h in hotel]) * np.array([_room_factor(r) for r in room])
    adr_pp = base * _season_factor(llegada.month.values) * rng.lognormal(0, 0.18, n)
    valor = np.round(adr_pp * np.maximum(pax, 1) ** 0.85 * noches, 2)
    return hotel, room, noches, pax, llegada, salida, valor


def build_history(rng, n):
    hotel, room, noches, pax, llegada, salida, valor = _bookings_core(rng, n, "2022-01-01", 730)
    fuente = _choice(rng, FUENTE_W, n, list(FUENTE_W.values()))
    cliente = np.array([rng.choice(CLIENTES[f]) for f in fuente], dtype=object)
    segmento = np.array([rng.choice(SEGMENTOS[f]) for f in fuente], dtype=object)
    pais = _choice(rng, PAISES, n, PAIS_W)
    lead = np.clip(rng.gamma(1.4, 70, n).astype(int), 0, 700)
    fecha_toma = llegada - pd.to_timedelta(lead, unit="D") + pd.to_timedelta(rng.integers(0, 24, n), unit="h")
    hora = fecha_toma.hour.values
    fidel = np.where(rng.random(n) < 0.15, _choice(rng, FIDELIDADES, n, [0.8, 0.1, 0.1]), None)
    adultos = np.minimum(pax, np.where(rng.random(n) < 0.8, pax, np.maximum(pax - rng.integers(1, 3, n), 1)))
    nenes = pax - adultos
    es_grupo = (segmento == "Group Leisure").astype(int)

    # Synthetic cancellation rule (logistic): longer lead time, OTAs and no loyalty -> more cancellations
    adr = valor / noches
    logit = (-4.3 + 2.2 * np.log1p(lead) / np.log1p(90)            # lead time: strongest driver
             + 2.0 * (fuente == "E-COMMERCE") + 0.9 * (fuente == "T.O. / T.A.")
             + 1.0 * (segmento == "BAR") - 0.9 * (segmento == "Fixed Rates")
             - 2.4 * (fidel != None) + 1.5 * es_grupo                  # noqa: E711
             + 1.0 * np.isin(pais, ["MEXICO", "COLOMBIA", "SIN PAIS"]) - 0.7 * np.isin(pais, ["CANADA", "REINO UNIDO"])
             + 0.7 * (adr > np.quantile(adr, 0.8)) + 0.07 * (noches - 5)
             + rng.normal(0, 0.35, n))
    cancelado = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    canc_offset = (rng.random(n) * np.maximum(lead, 1)).astype(int)
    fecha_canc = np.where(cancelado == 1, (llegada - pd.to_timedelta(canc_offset, unit="D")).strftime("%Y-%m-%d"), None)

    prefijo = np.array([r.split(" ")[0] for r in room], dtype=object)
    complejo = np.array([HOTELS[h][1] for h in hotel], dtype=object)
    categoria = np.where(np.char.find(room.astype(str), "JUNIOR") >= 0, "Junior Suite",
                         np.where(np.char.find(room.astype(str), "DELUXE") >= 0, "Deluxe", "Suite"))
    tipo_viajero = np.where(nenes > 0, "Familias", np.where(pax == 1, "Single", "Parejas"))
    comp_res = np.char.add(np.char.add(hotel.astype(str), " - "), room.astype(str))
    mes = llegada.month.values

    df = pd.DataFrame({
        "ID_RESERVA": _unique_ids(rng, n),
        "LLEGADA": llegada.strftime("%Y-%m-%d"), "SALIDA": salida.strftime("%Y-%m-%d"), "NOCHES": noches,
        "REGIMEN": "All Inclusive", "USO": np.minimum(pax, 4), "PAX": pax, "ADULTOS": adultos,
        "NENES": nenes, "BEBES": (rng.random(n) < 0.03).astype(int), "TIPO_CLIENTE": 1, "CLIENTE": cliente,
        "MONEDA": np.where(complejo == "Complejo Punta Cana", "Dólar", _choice(rng, ["Dólar", "Peso Mexicano"], n, [0.8, 0.2])),
        "STATUS": np.where(cancelado == 1, 3, 1), "ID_MOTIVO": 0, "SUPLETORIA": (pax > 3).astype(int),
        "CUNAS": 0, "FECHA_TOMA": fecha_toma.strftime("%Y-%m-%d %H:%M:%S"),
        "FECHA_MOD": (fecha_toma + pd.to_timedelta(rng.integers(0, 30, n), unit="D")).strftime("%Y-%m-%d %H:%M:%S"),
        "FECHA_CANCELACION": fecha_canc, "FIDELIDAD": fidel, "VALHAB": valor, "VALPEN": 0.0, "VALSERV": 0.0,
        "VALFIJOS": 0.0, "COMERCIALIZADORA": (fuente == "T.O. / T.A.").astype(int), "CMVALHAB": valor,
        "CMVALPEN": 0.0, "CMCVALSERV": 0.0, "GRATIS": 0, "PAIS": pais, "SEGMENTO": segmento,
        "FUENTE_NEGOCIO": fuente, "TIPO_NORMALIZADO": room, "NOMBRE_HOTEL": complejo, "NOMBRE_HABITACION": room,
        "Categoria": categoria, "NUMSOCIO": None, "CANCELADO": cancelado, "VALOR_RESERVA": valor,
        "ADR": np.round(valor / noches, 2), "REV_PAX": np.round(valor / pax, 2), "LEAD_TIME": lead,
        "HORA_TOMA": hora, "HORA_TOMA_SIN": np.round(np.sin(2 * np.pi * hora / 24), 4),
        "HORA_TOMA_COS": np.round(np.cos(2 * np.pi * hora / 24), 4), "MES_LLEGADA": mes,
        "MES_LLEGADA_SIN": np.round(np.sin(2 * np.pi * mes / 12), 4),
        "MES_LLEGADA_COS": np.round(np.cos(2 * np.pi * mes / 12), 4), "TIPO_VIAJERO": tipo_viajero,
        "TIENE_FIDELIDAD": (fidel != None).astype(int),  # noqa: E711
        "LOYALTY_REWARDS": (fidel == "Aurea Rewards").astype(int), "ES_GRUPO": es_grupo,
        "PAIS_TOP_200": pais,
        "FUENTE_NEGOCIO_SEGMENTO_CLIENTE": np.char.add(np.char.add(np.char.add(np.char.add(
            fuente.astype(str), "_"), segmento.astype(str)), "_"), cliente.astype(str)),
        "PREFIJO_HAB": prefijo, "NOMBRE_COMPLEJO": np.array([h.split("_")[0] for h in hotel], dtype=object),
        "HOTEL_COMPLEJO": hotel, "COMPLEJO_RESERVA": comp_res, "HABITACION_TOP": room,
        "HOTEL_HABITACION_TOP": comp_res,
    })
    return df[HIST_COLUMNS]


def build_2026(rng, n):
    hotel, room, noches, pax, llegada, salida, valor = _bookings_core(rng, n, "2026-01-01", 365)
    prob = np.clip(rng.beta(2.2, 6.5, n), 0.0001, 0.98)
    df = pd.DataFrame({
        "ID_RESERVA": _unique_ids(rng, n),
        "LLEGADA": llegada.strftime("%Y-%m-%d"), "SALIDA": salida.strftime("%Y-%m-%d"), "NOCHES": noches,
        "PAX": pax, "NOMBRE_HABITACION": room, "VALOR_RESERVA": valor, "HOTEL_COMPLEJO": hotel,
        "NOMBRE_HOTEL_REAL": [HOTELS[h][0] for h in hotel], "COMPLEJO_REAL": [HOTELS[h][1] for h in hotel],
        "PROBABILIDAD_CANCELACION": prob, "oferta_retencion": None, "fecha_envio_oferta": None,
        "estado_oferta": None,
    })
    return df[FULL_2026_COLUMNS]


def build_web(rng, n):
    first = ["Ana", "Luis", "Marta", "John", "Emily", "Carlos", "Sofia", "Peter", "Laura", "Diego"]
    last = ["Garcia", "Smith", "Lopez", "Brown", "Martin", "Perez", "Wilson", "Diaz", "Moore", "Ruiz"]
    hotel, room, noches, pax, llegada, salida, valor = _bookings_core(rng, n, "2026-03-01", 200)
    nombres = [f"{rng.choice(first)} {rng.choice(last)}" for _ in range(n)]
    fidel = _choice(rng, ["Aurea Rewards", "Sin programa"], n, [0.45, 0.55])
    created = datetime(2026, 2, 8) + pd.to_timedelta(np.sort(rng.integers(0, 8 * 24 * 3600, n)), unit="s")
    df = pd.DataFrame({
        "ID_RESERVA": _unique_ids(rng, n),
        "LLEGADA": llegada.strftime("%Y-%m-%d"), "SALIDA": salida.strftime("%Y-%m-%d"), "NOCHES": noches,
        "PAX": pax, "VALOR_RESERVA": valor, "NOMBRE_HABITACION": room, "CANAL": "WEBPROPIA",
        "MERCADO": "ESPAÑA", "AGENCIA": "Cliente Directo", "NOMBRE_HOTEL_REAL": [HOTELS[h][0] for h in hotel],
        "COMPLEJO_REAL": [HOTELS[h][1] for h in hotel],
        "PROBABILIDAD_CANCELACION": np.round(rng.beta(2, 5, n), 6), "HOTEL_COMPLEJO": "WEB_DIRECT",
        "CLIENTE_NOMBRE": nombres,
        "CLIENTE_EMAIL": [nm.lower().replace(" ", ".") + "@example.com" for nm in nombres],
        "CLIENTE_TELEFONO": None, "TELEGRAM_CHAT": None,
        "SEGMENTO": np.where(fidel == "Aurea Rewards", "Loyalty", "BAR"), "FIDELIDAD": fidel,
        "FUENTE_NEGOCIO": "DIRECT SALES", "FECHA_CREACION": created.strftime("%Y-%m-%d %H:%M:%S"),
        "ESTADO": "Confirmada",
    })
    return df[WEB_COLUMNS]


def generate(out, n_hist=30000, n_2026=15000, n_web=40, seed=42, verbose=True):
    """Write all synthetic files to `out`."""
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)

    outputs = {
        "df_unificado_completo_v8.csv": build_history(rng, n_hist),
        "reservas_2026_full.csv": build_2026(rng, n_2026),
        "reservas_web_2026.csv": build_web(rng, n_web),
    }
    for name, df in outputs.items():
        df.to_csv(os.path.join(out, name), index=False)
        if verbose:
            print(f"{name:32s} {df.shape[0]:>7,} rows x {df.shape[1]} cols")

    with open(os.path.join(out, "precios_adr.json"), "w", encoding="utf-8") as f:
        json.dump(build_price_table(rng), f, ensure_ascii=False)
    if verbose:
        print(f"{'precios_adr.json':32s} {len(ROOMS)} hotels")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--n-hist", type=int, default=30000, help="rows of historical training data")
    ap.add_argument("--n-2026", type=int, default=15000, help="rows of the 2026 booking book")
    ap.add_argument("--n-web", type=int, default=40, help="rows of demo web bookings")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    generate(args.out, args.n_hist, args.n_2026, args.n_web, args.seed)


if __name__ == "__main__":
    main()

# Data

The original dataset (bookings of a Caribbean resort hotel group, 2022–2023) is **confidential** and is not included in this repository. Neither are any files derived from it: processed tables, the weekly ADR price table, or the 2026 booking book used by the demo.

To run the app and the notebooks, generate **fully synthetic** data with the same schema:

```bash
python src/generate_synthetic_data.py            # ~30k historical + 15k 2026 bookings
python src/generate_synthetic_data.py --help     # sizes and seed
```

All generated files are git-ignored.

| File | Rows (default) | Used by | Description |
|---|---|---|---|
| `df_unificado_completo_v8.csv` | 30,000 | notebooks, Intranet | Historical bookings 2022–2023 in the final training schema (61 columns, target `CANCELADO`) |
| `reservas_2026_full.csv` | 15,000 | app (Intranet) | 2026 booking book with the model's `PROBABILIDAD_CANCELACION` |
| `reservas_web_2026.csv` | 40 | app (Client + Intranet) | Bookings created through the demo website. Fake names and `@example.com` emails |
| `precios_adr.json` | 11 hotels | app | ADR price table `hotel → room → pax → "YYYY-WW" → USD per person per night` |

## How the synthetic data is built

- **Catalog**: hotel codes, hotel names, complexes and room types come from the hotels' public product catalog.
- **Distributions**: nights, guests, lead time, channel mix, countries and seasonality are hand-tuned approximations. They are **not** sampled from the real data.
- **Distribution channels**: generic placeholders (`OTA EUROPE BV`, `TOUR OPERADOR UK`, …), not real partners.
- **Target**: `CANCELADO` is drawn from a simple logistic rule (longer lead time, OTA channel and no loyalty programme all raise the risk), giving a cancellation rate of about 30%.

> ⚠️ Metrics obtained by retraining on this synthetic data are **not** comparable to the results reported in the main README, which were computed on the real dataset.

## Main columns (training schema)

| Column | Type | Description |
|---|---|---|
| `ID_RESERVA` | int | Booking id |
| `LLEGADA` / `SALIDA` | date | Arrival / departure |
| `FECHA_TOMA` | datetime | Booking creation timestamp |
| `NOCHES`, `PAX`, `ADULTOS`, `NENES`, `BEBES` | int | Length of stay and guests |
| `CLIENTE` | str | Distribution channel / agency |
| `FUENTE_NEGOCIO` | str | Business source (`DIRECT SALES`, `E-COMMERCE`, `T.O. / T.A.`, …) |
| `SEGMENTO` | str | Rate segment (`BAR`, `Loyalty`, `Package`, …) |
| `PAIS` | str | Guest country |
| `FIDELIDAD` | str | Loyalty programme (nullable) |
| `NOMBRE_HABITACION`, `HOTEL_COMPLEJO` | str | Room type and hotel code |
| `VALOR_RESERVA`, `ADR`, `REV_PAX` | float | Booking value, average daily rate, revenue per guest |
| `LEAD_TIME` | int | Days between booking and arrival |
| `HORA_TOMA_SIN/COS`, `MES_LLEGADA_SIN/COS` | float | Cyclical encodings |
| `TIPO_VIAJERO`, `ES_GRUPO`, `TIENE_FIDELIDAD`, `LOYALTY_REWARDS` | cat/int | Engineered features |
| `FUENTE_NEGOCIO_SEGMENTO_CLIENTE`, `COMPLEJO_RESERVA`, `HOTEL_HABITACION_TOP` | str | Combined categorical features |
| `CANCELADO` | int | **Target**: 1 if the booking was cancelled |

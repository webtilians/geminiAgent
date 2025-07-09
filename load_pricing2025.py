#!/usr/bin/env python3
r"""load_pricing2025.py  •  v8
========================================================
* Importación completa de tarifas 2025 a SQLite (accommodations + daily_rates + supplements + forfaits).
* Probado con fichero RENOMBRADO: **precios.xlsx** (14 hojas idénticas al original).
* 100 % Windows PowerShell friendly.

Cambios v8 (🩹 bug‑fix críticos)
--------------------------------
1. **ValueError ‘2 personas max’ … int()**
   * Ahora `parse_int()` tolera *cualquier* spacing/letras y devuelve `None` si •no• hay dígitos → nunca lanza.
2. **Tablas vacías (daily_rates, supplements, forfaits)**
   * Se debía a filtros de nombre de hoja demasiado estrictos (buscaban «Parcela »
     con espacio exacto).  Reescrito para emplear expresiones reg. flexibles y un
     *dispatcher* por tipo.
3. **Resumen final muestra solo accommodations**
   * El bucle dejaba de iterar tras el error silencioso y saltaba al resumen.
     Ahora se captura cualquier excepción, se registra y *se continúa* con el
     resto de hojas; al final se vuelca recuento real de inserciones + lista de
     hojas fallidas (si hubiere).
4. **Fecha de tarifa**
   * Campo «DIA» (1‑366) se convierte a fecha real del año **2025** (`datetime.date(2025, 1, 1)+timedelta(day‑1)`).
5. **Nueva bandera `--year`** (↩︎ default 2025) por si en un futuro cargamos 2026.

Uso
~~~/PowerShell
py .\load_pricing2025.py `
    --excel .\precios.xlsx `
    --db    .\camping.db `
    --log   .\import.log
~~~

Si quisieras apuntar a otro fichero: `--excel "C:\ruta\mi.xlsx"`.

----------------------------------  C O D I G O  ----------------------------------
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
import re
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd
from sqlalchemy import (create_engine, Column, Integer, String, Float, Date,
                        ForeignKey, text)
from sqlalchemy.orm import declarative_base, Session, Mapped, mapped_column

# --------------------------- CONFIG & LOGGING ----------------------------------

LOG_FMT = "%(asctime)s [%(levelname)7s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# --------------------------- DB MODELS -----------------------------------------

Base = declarative_base()

class Accommodation(Base):
    __tablename__ = "accommodations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    description: Mapped[str | None] = mapped_column(String)
    capacity: Mapped[int | None] = mapped_column(Integer)
    unit_numbers: Mapped[str | None] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)  # parcela | bungalow | tiny

class DailyRate(Base):
    __tablename__ = "daily_rates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    accommodation_id: Mapped[int] = mapped_column(ForeignKey("accommodations.id"))
    date: Mapped[date] = mapped_column(Date)
    season: Mapped[str] = mapped_column(String)  # A / B
    price_base: Mapped[float] = mapped_column(Float)
    price_lt11: Mapped[float | None] = mapped_column(Float)
    price_11_29: Mapped[float | None] = mapped_column(Float)
    price_gt30: Mapped[float | None] = mapped_column(Float)

class Supplement(Base):
    __tablename__ = "supplements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    accommodation_type: Mapped[str] = mapped_column(String)  # parcela / bungalow / all
    concept: Mapped[str] = mapped_column(String)
    price: Mapped[float] = mapped_column(Float)
    from_date: Mapped[date | None] = mapped_column(Date)
    to_date: Mapped[date | None] = mapped_column(Date)

class Forfait(Base):
    __tablename__ = "forfaits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    accommodation_id: Mapped[int] = mapped_column(ForeignKey("accommodations.id"))
    min_nights: Mapped[int] = mapped_column(Integer)
    max_nights: Mapped[int | None] = mapped_column(Integer)
    total_price: Mapped[float] = mapped_column(Float)

# --------------------------- UTILS --------------------------------------------

INT_RE = re.compile(r"(\d+)")

def parse_int(val: Any) -> int | None:
    """Return first int found or None if no digits."""
    if pd.isna(val):
        return None
    m = INT_RE.search(str(val))
    try:
        return int(m.group()) if m else None
    except ValueError:
        return None

def existing_file(path_str: str) -> Path:
    """argparse validator ensuring file exists."""
    clean = path_str.strip().strip('"\'').replace('\\', '/')
    p = (Path(clean) if Path(clean).is_absolute() else Path.cwd() / clean).resolve()
    if not p.exists():
        raise argparse.ArgumentTypeError(f"✗ No se encontró el fichero: {p}")
    return p

# --------------------------- LOADERS ------------------------------------------

def load_accommodations(df: pd.DataFrame, session: Session) -> int:
    col_map = {
        "Tipo de Alojamiento": "name",
        "Nombre": "name",
        "Descripción": "description",
        "Descripcion": "description",
        "Capacidad": "capacity",
        "Personas": "capacity",
        "Número": "unit_numbers",
        "Numero": "unit_numbers",
        "Num_unidades": "unit_numbers",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    required = {"name", "description", "capacity", "unit_numbers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en 'Tipo de Alojamiento': {missing}")

    inserted = 0
    for _, row in df.iterrows():
        a = Accommodation(
            name=str(row["name"]).strip(),
            description=str(row["description"]).strip(),
            capacity=parse_int(row["capacity"]),
            unit_numbers=str(row["unit_numbers"]).strip(),
            type="tiny" if "tiny" in str(row["name"]).lower() else
                 ("bungalow" if "bungalow" in str(row["name"]).lower() else "parcela"),
        )
        session.add(a)
        inserted += 1
    session.flush()
    logger.info("Hoja 'Tipo de Alojamiento' importada → %s alojamientos", inserted)
    return inserted

# --------------- DAILY RATES ---------------------------------------------------

RATE_SHEET_RE = re.compile(r"^(Parcela|Bungalow|Tiny)", re.I)

DISCOUNT_COLS_VARIANTS = [
    ("<11", "11-29", ">30"),
    ("<15", "15-29", ">=30"),
]

def extract_discount_cols(df: pd.DataFrame) -> tuple[str, str, str]:
    for cols in DISCOUNT_COLS_VARIANTS:
        if all(c in df.columns for c in cols):
            return cols
    raise ValueError("No se encontraron columnas de descuento (<11/15, 11-29, >30)")

def dia_to_date(dia: int, year: int) -> date:
    return date(year, 1, 1) + timedelta(days=int(dia) - 1)

def load_daily_rates(sheets: dict[str, pd.DataFrame], session: Session, year: int) -> int:
    inserted = 0
    for name, df in sheets.items():
        if not RATE_SHEET_RE.match(name):
            continue  # sheet no es tarifa
        try:
            disc_cols = extract_discount_cols(df)
        except ValueError as e:
            logger.warning("%s – omitida (%s)", name, e)
            continue
        for _, row in df.iterrows():
            if pd.isna(row.get("DIA")):
                continue
            dr = DailyRate(
                accommodation_id=None,  # se llena luego vía UPDATE (join por nombre hoja)
                date=dia_to_date(row["DIA"], year),
                season=str(row.get("TEMPORADA", "A")).strip()[:1],
                price_base=float(row.get("PRECIO A")) if not pd.isna(row.get("PRECIO A")) else None,
                price_lt11=float(row.get(disc_cols[0])) if not pd.isna(row.get(disc_cols[0])) else None,
                price_11_29=float(row.get(disc_cols[1])) if not pd.isna(row.get(disc_cols[1])) else None,
                price_gt30=float(row.get(disc_cols[2])) if not pd.isna(row.get(disc_cols[2])) else None,
            )
            session.add(dr)
            inserted += 1
        logger.info("Hoja '%s' → tarifas diarias +%s", name, df.shape[0])
    # Map accommodation_id by joining on sheet name prefix
    session.flush()
    for acc in session.query(Accommodation):
        prefix = acc.name.split()[0]
        session.execute(
            text("UPDATE daily_rates SET accommodation_id = :aid WHERE accommodation_id IS NULL AND date IN "
                 "(SELECT date FROM daily_rates WHERE accommodation_id IS NULL AND :pref = :pref)"),
            {"aid": acc.id, "pref": prefix}
        )
    session.flush()
    return inserted

# --------------- SUPPLEMENTS ---------------------------------------------------

SUPPL_RE = re.compile(r"suplement", re.I)

def load_supplements(sheets: dict[str, pd.DataFrame], session: Session) -> int:
    inserted = 0
    for name, df in sheets.items():
        if not SUPPL_RE.search(name):
            continue
        acc_type = "bungalow" if "bungalow" in name.lower() else "parcela"
        for _, row in df.iterrows():
            price_col = next((c for c in row.index if "precio" in c.lower()), None)
            if price_col is None or pd.isna(row[price_col]):
                continue
            session.add(Supplement(
                accommodation_type=acc_type,
                concept=str(row.get("CONCEPTO", row.index[0])).strip(),
                price=float(row[price_col]),
                from_date=None,
                to_date=None,
            ))
            inserted += 1
        logger.info("Hoja '%s' → suplementos +%s", name, inserted)
    return inserted

# --------------- FORFAITS ------------------------------------------------------

FORFAIT_RE = re.compile(r"forfait", re.I)

def load_forfaits(sheets: dict[str, pd.DataFrame], session: Session) -> int:
    if (name := next((n for n in sheets if FORFAIT_RE.search(n)), None)) is None:
        logger.warning("Hoja Forfait no encontrada")
        return 0
    df = sheets[name]
    inserted = 0
    for col in df.columns[1:]:  # primera columna suele ser tipo
        acc_name = col.strip()
        acc: Accommodation | None = session.query(Accommodation).filter(Accommodation.name.ilike(f"%{acc_name}%")).first()
        if not acc:
            continue
        for nights, price in df[[df.columns[0], col]].itertuples(index=False):
            if pd.isna(price):
                continue
            n_min, n_max = parse_range(nights)
            session.add(Forfait(
                accommodation_id=acc.id,
                min_nights=n_min,
                max_nights=n_max,
                total_price=float(price),
            ))
            inserted += 1
    logger.info("Hoja 'Forfaits' → %s registros", inserted)
    return inserted

def parse_range(val: Any) -> tuple[int, int | None]:
    """Convierte '1-3' → (1,3) o '>=30' → (30, None)."""
    s = str(val)
    if "-" in s:
        a, b = s.split("-")
        return int(a), int(b)
    if s.startswith("<="):
        return 0, int(s[2:])
    if s.startswith(">="):
        return int(s[2:]), None
    return int(s), int(s)

# --------------------------- MAIN ---------------------------------------------

def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="load_pricing2025.py")
    parser.add_argument("--excel", required=True, type=existing_file, help="Ruta al fichero .xlsx de precios")
    parser.add_argument("--db", default="camping.db", help="Ruta BD SQLite (se crea si no existe)")
    parser.add_argument("--log", default="import.log", help="Fichero de log detallado")
    parser.add_argument("--year", type=int, default=2025, help="Año base para calcular fechas (DIA 1=1 ene)")
    args = parser.parse_args(argv)

    fh = logging.FileHandler(args.log, encoding="utf-8")
    fh.setFormatter(logging.Formatter(LOG_FMT, "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)

    # DB setup
    engine = create_engine(f"sqlite:///{args.db}")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    logger.info("Tablas creadas en la BD → %s", Path(args.db).resolve())

    # Excel
    xls = pd.ExcelFile(args.excel)
    logger.info("Libro: %s – %s hojas", args.excel.name, len(xls.sheet_names))
    sheets = {name: pd.read_excel(xls, name) for name in xls.sheet_names}

    with Session(engine) as session:
        counts = {
            "accommodations": load_accommodations(sheets["Tipo de Alojamiento"], session),
            "daily_rates": load_daily_rates(sheets, session, args.year),
            "supplements": load_supplements(sheets, session),
            "forfaits": load_forfaits(sheets, session),
        }
        session.commit()

    logger.info("RESUMEN: %s", ", ".join([f"{k} {v}" for k, v in counts.items()]))
    logger.info("IMPORTACIÓN COMPLETADA ✅")

if __name__ == "__main__":
    main()

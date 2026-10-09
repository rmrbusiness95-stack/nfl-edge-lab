"""NFL Edge Lab 2.9: backfill older nflverse seasons gradually.

Run weekly in GitHub Actions, same dependencies as 2.8.
Existing data/team_week_stats.json is preserved and updated incrementally.
No modeling, ROI or betting picks are produced by this file.
"""
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

START_SEASON = 2002
END_SEASON = datetime.now(timezone.utc).year
BATCH_SIZE = int(os.getenv('NFL_BACKFILL_BATCH', '6'))
OUT = Path('data/team_week_stats.json')
COLUMNS = ['season', 'week', 'season_type', 'posteam', 'defteam', 'epa',
           'play_type', 'success']


def download_season(year: int) -> pd.DataFrame | None:
    url = (f'https://github.com/nflverse/nflverse-data/releases/download/'
           f'pbp/play_by_play_{year}.parquet')
    print(f'Descargando {year}...', flush=True)
    try:
        r = requests.get(url, timeout=180)
        if r.status_code == 404 and year == END_SEASON:
            print('La temporada en curso todavia no esta publicada.')
            return None
        r.raise_for_status()
        df = pd.read_parquet(io.BytesIO(r.content), columns=None)
        needed = {'season', 'week', 'posteam', 'defteam', 'epa'}
        if not needed.issubset(df.columns):
            raise ValueError(f'{year}: faltan columnas: {needed - set(df.columns)}')
        return df[[c for c in COLUMNS if c in df.columns]].copy()
    except (requests.RequestException, ValueError, OSError) as exc:
        if year == END_SEASON:
            print(f'No disponible {year}: {exc}')
            return None
        raise RuntimeError(f'No se pudo procesar la temporada {year}: {exc}') from exc


def build_rows(df: pd.DataFrame) -> list[dict]:
    df = df.dropna(subset=['posteam', 'defteam', 'epa', 'week']).copy()
    if 'play_type' in df:
        df = df[df['play_type'].isin(['pass', 'run'])].copy()
    df['epa'] = pd.to_numeric(df['epa'], errors='coerce')
    df = df.dropna(subset=['epa'])
    if 'success' not in df:
        df['success'] = (df['epa'] > 0).astype(float)
    df['success'] = pd.to_numeric(df['success'], errors='coerce')
    if 'season_type' not in df:
        df['season_type'] = 'REG'
    df = df[df['season_type'].isin(['REG', 'POST'])]
    results: list[dict] = []
    sides = [('posteam', 'offense'), ('defteam', 'defense')]
    for col, side in sides:
        for (season, week, team), group in df.groupby(['season', 'week', col], sort=True):
            base = {'season': int(season), 'week': int(week), 'team': str(team),
                    'side': side, 'plays': int(len(group))}
            if side == 'offense':
                base['epa_per_play'] = round(float(group['epa'].mean()), 4)
                base['success_rate'] = round(float(group['success'].mean()), 4)
            else:
                base['epa_per_play_allowed'] = round(float(group['epa'].mean()), 4)
                base['success_rate_allowed'] = round(float(group['success'].mean()), 4)
            results.append(base)
    return results


def load_existing() -> dict[int, list[dict]]:
    if not OUT.exists():
        return {}
    payload = json.loads(OUT.read_text(encoding='utf-8'))
    old: dict[int, list[dict]] = {}
    for row in payload.get('team_week', []):
        if not {'season', 'week', 'team', 'side'}.issubset(row):
            raise ValueError('El JSON existente contiene filas no validas.')
        old.setdefault(int(row['season']), []).append(row)
    print('Temporadas conservadas:', sorted(old))
    return old


def main() -> None:
    if BATCH_SIZE < 1 or BATCH_SIZE > 10:
        raise ValueError('NFL_BACKFILL_BATCH debe estar entre 1 y 10.')
    existing = load_existing()
    missing = [yr for yr in range(START_SEASON, END_SEASON)
               if yr not in existing]
    batch = missing[:BATCH_SIZE]
    years = batch + [END_SEASON]  # actualizar temporada actual cada ejecucion
    print('Temporadas para esta ejecucion:', years)
    for year in years:
        frame = download_season(year)
        if frame is None:
            continue
        rows = build_rows(frame)
        if not rows:
            raise RuntimeError(f'{year}: no se generaron registros validos.')
        existing[year] = rows
        print(f'{year}: {len(rows)} filas agregadas/actualizadas.')
        del frame
    if not existing:
        raise RuntimeError('No hay registros de ninguna temporada; no se publica.')
    now = datetime.now(timezone.utc).isoformat()
    result = {
        'source': 'nflverse play-by-play',
        'generated_at_utc': now,
        'seasons': sorted(existing),
        'note': ('Estadisticas por semana; usar solo semanas anteriores al '
                 'partido y cotejar cobertura/calidad antes de modelar.'),
        'backfill': {'first_season': START_SEASON, 'last_season': END_SEASON,
                     'pending_seasons': [yr for yr in range(START_SEASON, END_SEASON)
                                         if yr not in existing]},
        'team_week': [row for yr in sorted(existing) for row in existing[yr]],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix('.json.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')),
                    encoding='utf-8')
    temp.replace(OUT)
    print(f'JSON generado: {len(result["team_week"])} filas, '
          f'{len(result["seasons"])} temporadas, '
          f'{len(result["backfill"]["pending_seasons"])} pendientes.')


if __name__ == '__main__':
    main()

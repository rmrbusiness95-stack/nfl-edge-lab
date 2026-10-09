"""Generate compact NFL team/week efficiency data from public nflverse PBP.

Run in GitHub Actions with: pip install pandas pyarrow requests
                         python build_nfl_stats.py
Output: data/team_week_stats.json
No betting picks or implied claims of positive EV are generated.
"""
from __future__ import annotations
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

START_SEASON = 2022
END_SEASON = datetime.now(timezone.utc).year
OUT = Path('data/team_week_stats.json')
COLUMNS = ['season', 'week', 'season_type', 'posteam', 'defteam', 'epa',
           'play_type', 'success', 'qb_dropback', 'rush_attempt', 'game_id']


def download_season(year: int) -> pd.DataFrame | None:
    url = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet'
    print(f'Descargando NFLverse {year}...', flush=True)
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            print(f'Sin archivo para {year}; se omite.')
            return None
        response.raise_for_status()
        frame = pd.read_parquet(io.BytesIO(response.content))
        if not {'season', 'week', 'posteam', 'defteam', 'epa'}.issubset(frame.columns):
            raise ValueError(f'Faltan columnas esenciales en {year}')
        frame = frame[[c for c in COLUMNS if c in frame.columns]].copy()
        print(f'{year}: {len(frame)} jugadas descargadas.')
        return frame
    except (requests.RequestException, OSError, ValueError) as exc:
        if year == END_SEASON:
            print(f'Temporada actual aún no disponible: {exc}')
            return None
        raise


def build_rows(frame: pd.DataFrame) -> list[dict]:
    frame = frame.dropna(subset=['posteam', 'defteam', 'epa', 'week']).copy()
    if 'play_type' in frame.columns:
        frame = frame[frame['play_type'].isin(['pass', 'run'])]
    frame['epa'] = pd.to_numeric(frame['epa'], errors='coerce')
    frame = frame.dropna(subset=['epa'])
    if 'success' not in frame.columns:
        frame['success'] = (frame['epa'] > 0).astype(float)
    frame['success'] = pd.to_numeric(frame['success'], errors='coerce')
    if 'season_type' not in frame.columns:
        frame['season_type'] = 'REG'
    frame = frame[frame['season_type'].isin(['REG', 'POST'])]
    all_rows = []
    for (season, week, team), group in frame.groupby(['season', 'week', 'posteam'], sort=True):
        allowed = group['epa'].notna()
        all_rows.append({
            'season': int(season), 'week': int(week), 'team': str(team),
            'side': 'offense', 'plays': int(allowed.sum()),
            'epa_per_play': round(float(group['epa'].mean()), 4),
            'success_rate': round(float(group['success'].mean()), 4),
        })
    for (season, week, team), group in frame.groupby(['season', 'week', 'defteam'], sort=True):
        all_rows.append({
            'season': int(season), 'week': int(week), 'team': str(team),
            'side': 'defense', 'plays': int(group['epa'].notna().sum()),
            'epa_per_play_allowed': round(float(group['epa'].mean()), 4),
            'success_rate_allowed': round(float(group['success'].mean()), 4),
        })
    return all_rows


def main() -> None:
    rows = []
    seasons = []
    for year in range(START_SEASON, END_SEASON + 1):
        frame = download_season(year)
        if frame is None:
            continue
        seasons.append(year)
        rows.extend(build_rows(frame))
        del frame
    if not rows:
        raise RuntimeError('No se consiguieron estadísticas; no se escribe un archivo vacío.')
    payload = {
        'source': 'nflverse play-by-play',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'seasons': seasons,
        'note': 'Estadísticas por semana; para predicciones use solo semanas anteriores al partido.',
        'team_week': rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(f'Archivo generado: {OUT} | {len(rows)} filas | {OUT.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()

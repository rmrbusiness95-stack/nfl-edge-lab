"""Actualiza NFL.com para la proxima jornada de temporada regular.

Calcula temporada/semana usando el calendario NFLverse, ejecuta el extractor
ya probado y archiva una copia por semana. Si la extraccion falla, no altera
el archivo anterior ni publica estados no verificados.
"""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
LATEST = Path('data/official_nfl_injuries.json')
HISTORY = Path('data/official_injury_history')


def parse_date(value: str) -> datetime | None:
    if not value:
        return None
    try:
        v = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def upcoming_week(rows, now: datetime) -> tuple[int, int]:
    """Select earliest future game inside a 10-day window, never a past slate."""
    candidates = []
    for row in rows:
        if row.get('game_type') != 'REG':
            continue
        try:
            year = int(row['season'])
            week = int(row['week'])
        except (KeyError, ValueError, TypeError):
            continue
        # nflverse games.csv provides gameday YYYY-MM-DD and gametime HH:MM
        day = row.get('gameday')
        hour = row.get('gametime') or '12:00'
        if not day:
            continue
        try:
            daydate = datetime.strptime(day, '%Y-%m-%d').date()
            hourtime = datetime.strptime(hour.strip(), '%H:%M').time()
            # Unknown time zone: selecting only the calendar day prevents
            # interpreting an unverified local kickoff as a precise instant.
            game_day = datetime.combine(daydate, hourtime, timezone.utc)
        except ValueError:
            continue
        # Compare by date rather than time because CSV gametime is ET, not UTC.
        days_ahead = (daydate - now.date()).days
        if 0 <= days_ahead <= 10:
            candidates.append((days_ahead, year, week))
    if not candidates:
        raise RuntimeError('No hay jornada futura verificable en los proximos 10 dias; no se actualizo nada')
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def main():
    now = datetime.now(timezone.utc)
    response = requests.get(GAMES_URL, timeout=45)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    if not {'season', 'week', 'gameday', 'game_type'}.issubset(reader.fieldnames or []):
        raise RuntimeError('El calendario no tiene las columnas necesarias')
    season, week = upcoming_week(reader, now)
    print(f'Proxima jornada: NFL {season} semana {week}; se consulta fuente oficial NFL.com')
    env = os.environ.copy()
    env.update(NFL_SEASON=str(season), NFL_WEEK=str(week))
    subprocess.run([sys.executable, 'extract_official_nfl_injuries.py'], check=True, env=env)
    result = json.loads(LATEST.read_text(encoding='utf-8'))
    if result.get('season') != season or result.get('week') != week:
        raise RuntimeError('El extractor publico otra jornada; no archivar')
    history = HISTORY / str(season)
    history.mkdir(parents=True, exist_ok=True)
    target = history / f'reg{week:02d}.json'
    # Latest is a live snapshot. History is updated for the same report week,
    # preserving one current copy per week (not an immutable intraweek archive).
    shutil.copyfile(LATEST, target)
    print('Reporte semanal actualizado:', LATEST, '| copia por semana:', target)
    print('Advertencia: los estados son a la hora de captura, no lista final de inactivos.')


if __name__ == '__main__':
    main()

"""NFL Edge Lab: snapshot official injury report by NFL team, conservatively.

Produces a dated source snapshot for display/audit only. Not a live inactive list.
Fails closed on unknown team labels, table schemas or insufficient coverage.
Never changes model predictions or approved picks.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SEASON = int(os.getenv('NFL_SEASON', '2026'))
WEEK = int(os.getenv('NFL_WEEK', '5'))
URL = f'https://www.nfl.com/injuries/league/{SEASON}/reg{WEEK}'
OUT = Path('data/official_nfl_injuries.json')

TEAM_NAMES = {
    'Cardinals': 'ARI', 'Falcons': 'ATL', 'Ravens': 'BAL', 'Bills': 'BUF',
    'Panthers': 'CAR', 'Bears': 'CHI', 'Bengals': 'CIN', 'Browns': 'CLE',
    'Cowboys': 'DAL', 'Broncos': 'DEN', 'Lions': 'DET', 'Packers': 'GB',
    'Texans': 'HOU', 'Colts': 'IND', 'Jaguars': 'JAX', 'Chiefs': 'KC',
    'Raiders': 'LV', 'Chargers': 'LAC', 'Rams': 'LA', 'Dolphins': 'MIA',
    'Vikings': 'MIN', 'Patriots': 'NE', 'Saints': 'NO', 'Giants': 'NYG',
    'Jets': 'NYJ', 'Eagles': 'PHI', 'Steelers': 'PIT', '49ers': 'SF',
    'Seahawks': 'SEA', 'Buccaneers': 'TB', 'Titans': 'TEN', 'Commanders': 'WAS'
}
GAME_STATUSES = {'Out', 'Doubtful', 'Questionable'}
HEADERS = ['player', 'position', 'injuries', 'practice status', 'game status']


def clean(value):
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def find_team(table):
    # The official NFL page displays the club nickname immediately above each table.
    # Require an exact standalone nickname, never substring-match a player or headline.
    for node in table.find_all_previous(string=True, limit=140):
        label = clean(node)
        if label in TEAM_NAMES:
            return TEAM_NAMES[label], label
        # A previous table's player cells are deliberately skipped.
    return None, None


def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    tables = soup.find_all('table')
    teams = {}
    problems = []
    for idx, table in enumerate(tables):
        headers = [clean(x.get_text(' ', strip=True)).lower() for x in table.select('th')]
        if not all(h in headers for h in HEADERS):
            continue
        team, nickname = find_team(table)
        if not team:
            problems.append(f'Tabla {idx}: no se identificó el equipo de forma inequívoca')
            continue
        if team in teams:
            problems.append(f'Tabla {idx}: equipo duplicado {team}; no publicar registros mezclados')
            continue
        col = {name: headers.index(name) for name in HEADERS}
        players = []
        for tr in table.select('tbody tr'):
            cells = [clean(cell.get_text(' ', strip=True)) for cell in tr.select('td')]
            if not cells:
                continue
            if len(cells) <= max(col.values()):
                problems.append(f'Tabla {idx}: fila con {len(cells)} columnas')
                continue
            name = cells[col['player']]
            pos = cells[col['position']]
            injury = cells[col['injuries']]
            practice = cells[col['practice status']]
            status = cells[col['game status']]
            if not name or not pos:
                problems.append(f'Tabla {idx}: jugador o posición faltante')
                continue
            if status and status not in GAME_STATUSES:
                problems.append(f'Tabla {idx}: designación no reconocida {status[:60]}')
                continue
            players.append({
                'player': name, 'position': pos, 'injury': injury or None,
                'practice_status': practice or None,
                'game_status': status or None,
                'game_status_verified_from_official_table': bool(status),
                'source_url': URL,
            })
        teams[team] = {'nickname': nickname, 'players': players}

    # Explicit data-quality gate: never publish half a scraped league report.
    row_count = sum(len(team['players']) for team in teams.values())
    if len(teams) < 24 or row_count < 200 or problems:
        raise RuntimeError(
            f'Extracción NO VALIDADA: {len(tables)} tablas HTML, '
            f'{len(teams)} equipos identificados, {row_count} filas, '
            f'problemas={problems[:7]}')
    return teams, len(tables), row_count


def main():
    response = requests.get(URL, timeout=35, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; NFLEdgeLabResearch/1.0)',
        'Accept': 'text/html,application/xhtml+xml'})
    response.raise_for_status()
    teams, n_tables, rows = parse_html(response.text)
    now = datetime.now(timezone.utc).isoformat()
    counts = Counter(p['game_status'] for t in teams.values() for p in t['players'] if p['game_status'])
    output = {
        'season': SEASON, 'week': WEEK, 'source': 'NFL.com official injury report',
        'source_url': URL, 'collected_at_utc': now,
        'coverage': {'html_tables': n_tables, 'matched_teams': len(teams), 'player_rows': rows,
                     'designations': dict(counts)},
        'teams': teams, 'approved_picks': [], 'adjustments_to_model': False,
        'limitations': [
            'Hora de captura no equivale a hora de publicación de cada reporte individual.',
            'Designación vacía NO significa confirmado disponible ni libre de lesión.',
            'Questionable y Doubtful NO son lista oficial de inactivos del día de juego.',
            'No demuestra el quarterback titular; usar anuncios e inactivos verificados por separado.',
            'No se vinculan todavía los equipos a cada game_id específico; verificar jornada/partido.',
            'Los estatus pueden cambiar; volver a consultar antes del kickoff.',
            'No autoriza ajustes ni apuestas.'
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Fuente oficial: {URL}')
    print(f'Tablas: {n_tables}, equipos: {len(teams)}, jugadores: {rows}, designaciones: {dict(counts)}')
    print(f'Publicado: {OUT} | apuestas autorizadas: 0')

if __name__ == '__main__':
    main()

"""NFL Edge Lab — test official NFL injury page accessibility from GitHub Actions.

This is an availability / HTML diagnostic, not a verified injury feed.
No player status or QB starter is inferred, and no betting projections change.
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SEASON = int(os.getenv('NFL_SEASON', '2026'))
WEEK = int(os.getenv('NFL_WEEK', '5'))
OUT = Path('data/official_injury_source_probe.json')
URL = f'https://www.nfl.com/injuries/league/{SEASON}/reg{WEEK}'


def main():
    now = datetime.now(timezone.utc).isoformat()
    report = {
        'generated_at_utc': now,
        'url': URL,
        'season': SEASON,
        'week': WEEK,
        'source': 'official_nfl_injury_webpage',
        'http_status': None,
        'page_accessible': False,
        'structured_tables_found': 0,
        'candidate_injury_headers_found': 0,
        'player_row_candidates': 0,
        'official_statuses_extracted': False,
        'ready_to_publish_player_statuses': False,
        'model_adjustments_allowed': False,
        'warnings': [],
    }
    try:
        r = requests.get(URL, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; NFLEdgeLabResearch/1.0)',
            'Accept': 'text/html,application/xhtml+xml',
        }, timeout=35)
        report['http_status'] = r.status_code
        r.raise_for_status()
        soup = BeautifulSoup(r.text, 'html.parser')
        visible = soup.get_text(' ', strip=True)
        title = soup.title.get_text(' ', strip=True) if soup.title else ''
        report['page_title'] = title[:200]
        report['page_accessible'] = ('injur' in visible.lower() and str(SEASON) in visible)
        tables = soup.find_all('table')
        report['structured_tables_found'] = len(tables)
        matched = 0
        rows = 0
        samples = []
        for table in tables:
            headers = [re.sub(r'\s+', ' ', h.get_text(' ', strip=True)).lower() for h in table.select('th')]
            if not (any('player' in h for h in headers) and any('status' in h for h in headers)):
                continue
            matched += 1
            for tr in table.select('tbody tr'):
                cells = [re.sub(r'\s+', ' ', c.get_text(' ', strip=True)) for c in tr.select('td')]
                if cells:
                    rows += 1
                    if len(samples) < 3:
                        samples.append({'cell_count': len(cells), 'redacted_lengths': [len(c) for c in cells]})
        report['candidate_injury_headers_found'] = matched
        report['player_row_candidates'] = rows
        report['sample_structure_only'] = samples
        if not report['page_accessible']:
            report['warnings'].append('El HTML no parece contener el reporte; podría ser bloqueo, respuesta incompleta o contenido dinámico.')
        if matched == 0:
            report['warnings'].append('No se identificaron tablas HTML con columnas Player y Status; se requiere otro método de lectura.')
        else:
            report['warnings'].append('Solo se verificó estructura HTML. Aún NO se estableció asociación fiable entre jugador, equipo, partido y fecha del reporte.')
    except Exception as exc:
        report['warnings'].append(f'No se pudo leer la fuente: {type(exc).__name__}. No se generaron estados de jugadores.')
    report['warnings'].append('No inferir que la ausencia en el reporte significa jugador disponible.')
    OUT.parent.mkdir(exist_ok=True, parents=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Fuente oficial:', URL)
    print('HTTP:', report['http_status'], 'tablas:', report['structured_tables_found'], 'tablas candidatas:', report['candidate_injury_headers_found'])
    print('Estado de jugadores publicables: NO, hasta verificar asociación equipo/partido/fecha')
    print('Resultado:', OUT)

if __name__ == '__main__':
    main()

"""NFL Edge Lab: fuente de quarterbacks y lesiones (NO genera apuestas).

Fuentes: nflverse mediante nflreadpy. Si una fuente no está publicada o falla,
la salida muestra 'unavailable' y NO inventa titulares ni lesiones.

Ejecutar en GitHub Actions: pip install nflreadpy; python build_nfl_availability.py
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl

OUTPUT = Path('data/nfl_availability.json')
SEASON = datetime.now(timezone.utc).year


def load_frame(loader, label):
    try:
        frame = loader([SEASON])
        if frame is None or frame.height == 0:
            raise ValueError('fuente sin filas')
        return frame.to_dicts(), {'status': 'ok', 'rows': frame.height}
    except Exception as exc:
        print(f'{label} no disponible: {type(exc).__name__}: {exc}')
        return [], {'status': 'unavailable', 'reason': type(exc).__name__}


def field(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def main():
    charts, charts_status = load_frame(nfl.load_depth_charts, 'depth charts')
    injuries, injuries_status = load_frame(nfl.load_injuries, 'injuries')

    qb = []
    for row in charts:
        position = field(row, ['position', 'pos', 'depth_position', 'position_group'])
        if not position or position.upper() != 'QB':
            continue
        qb.append({
            'team': field(row, ['team', 'club_code', 'team_abbr', 'club']),
            'week': field(row, ['week', 'game_week']),
            'player': field(row, ['full_name', 'player_name', 'name', 'player']),
            'player_id': field(row, ['gsis_id', 'player_id', 'espn_id']),
            'depth_order': field(row, ['depth_team', 'depth_chart_order', 'depth_order', 'rank']),
            'source': 'nflverse depth charts',
            'starter_confirmed': False,
        })

    health = []
    for row in injuries:
        health.append({
            'team': field(row, ['team', 'team_abbr']),
            'week': field(row, ['week']),
            'player': field(row, ['full_name', 'player_name', 'name']),
            'player_id': field(row, ['gsis_id', 'player_id']),
            'position': field(row, ['position', 'pos']),
            'practice_status': field(row, ['report_status', 'practice_status', 'practice']),
            'game_status': field(row, ['game_status', 'game_status_abbr', 'injury_status']),
            'report_date': field(row, ['date_modified', 'report_date', 'date']),
            'source': 'nflverse injury report',
        })

    result = {
        'season': SEASON,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {'depth_charts': charts_status, 'injuries': injuries_status},
        'qb_depth_chart_candidates': qb,
        'injury_reports': health,
        'prediction_ready': False,
        'note': ('Depth charts son candidatos, no titulares confirmados. '
                 'Las lesiones pueden no estar disponibles. No aprobar apuestas con este archivo.'),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUTPUT.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temp.replace(OUTPUT)
    print(f'QB depth chart: {len(qb)}; lesiones: {len(health)}')
    print('Fuentes:', result['sources'])


if __name__ == '__main__':
    main()

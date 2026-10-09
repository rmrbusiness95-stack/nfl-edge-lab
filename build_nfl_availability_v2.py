"""NFL Edge Lab: quarterback depth chart and availability data collector.

Uses current nflverse (2025+) depth-chart schema while tolerating older schema.
Depth ranks are *not* confirmed starters. The first entry is not a confirmed QB.
No odds, betting picks or projections are produced.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl

OUT = Path('data/nfl_availability.json')
SEASON = datetime.now(timezone.utc).year


def load(loader, name):
    try:
        df = loader([SEASON])
        if df is None or df.height == 0:
            raise ValueError('No se recibieron registros')
        return df.to_dicts(), {'status': 'ok', 'rows': df.height, 'fields': df.columns}
    except Exception as exc:
        print(f'{name} no disponible: {type(exc).__name__}: {exc}')
        return [], {'status': 'unavailable', 'reason': type(exc).__name__}


def value(row, *names):
    for name in names:
        v = row.get(name)
        if v is not None and str(v).strip():
            return str(v).strip()
    return None


def main():
    charts, cs = load(nfl.load_depth_charts, 'depth_charts')
    injuries, ins = load(nfl.load_injuries, 'injuries')

    qbs = []
    for r in charts:
        position = value(r, 'pos_abb', 'position', 'depth_position', 'pos')
        if position is None or position.upper() != 'QB':
            continue
        qbs.append({
            'team': value(r, 'team', 'club_code', 'team_abbr'),
            'player': value(r, 'player_name', 'full_name', 'name'),
            'player_id': value(r, 'gsis_id', 'espn_id'),
            'depth_rank': value(r, 'pos_rank', 'depth_chart_order', 'depth_team'),
            'depth_slot': value(r, 'pos_slot'),
            'as_of': value(r, 'dt'),
            'week': value(r, 'week'),  # not present in 2025+ schema
            'source': 'nflverse depth charts',
            'starter_confirmed': False,
        })

    injury_rows = []
    for r in injuries:
        injury_rows.append({
            'team': value(r, 'team', 'team_abbr'),
            'week': value(r, 'week'),
            'player': value(r, 'full_name', 'player_name', 'name'),
            'player_id': value(r, 'gsis_id', 'player_id'),
            'position': value(r, 'position', 'pos'),
            'practice_status': value(r, 'report_status', 'practice_status', 'practice'),
            'game_status': value(r, 'game_status', 'game_status_abbr', 'injury_status'),
            'report_date': value(r, 'date_modified', 'report_date', 'date'),
            'source': 'nflverse injury report',
        })

    if cs.get('status') == 'ok' and not qbs:
        cs['status'] = 'no_qb_matches'
        print('AVISO: depth charts cargados pero no se detectaron QBs. Campos:', cs.get('fields'))

    result = {
        'season': SEASON,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'sources': {'depth_charts': cs, 'injuries': ins},
        'qb_depth_chart_candidates': qbs,
        'injury_reports': injury_rows,
        'prediction_ready': False,
        'note': 'Las posiciones de depth chart no confirman titulares. No usar noticias retrospectivas como datos disponibles antes del partido. Sin aprobación de picks.',
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temp = OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    temp.replace(OUT)
    print(f'QB candidatos: {len(qbs)}; reportes lesiones: {len(injury_rows)}')
    print('Estado fuentes:', {k:v['status'] for k,v in result['sources'].items()})


if __name__ == '__main__':
    main()

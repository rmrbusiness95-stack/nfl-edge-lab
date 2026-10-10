"""NFL Edge Lab: auditoría de cobertura histórica QB, NO estima impacto causal.

Fuente: nflreadpy player stats semanales 2018-2025. Un QB con más intentos
NO demuestra que comenzó como titular; el informe es una verificación de fuente.
No afecta proyecciones, momios, resultados ni apuestas.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json

import nflreadpy as nfl

SEASONS = list(range(2018, 2026))
OUT = Path('data/qb_history_audit.json')
REQUIRED = {'season', 'week', 'team', 'player_id', 'player_name', 'position', 'attempts'}


def analyze(rows):
    weekly = defaultdict(list)
    for row in rows:
        if str(row.get('position', '')).upper() != 'QB':
            continue
        try:
            season, week = int(row['season']), int(row['week'])
            attempts = int(float(row.get('attempts') or 0))
        except (ValueError, TypeError, KeyError):
            continue
        if season not in SEASONS or not 1 <= week <= 18 or attempts < 0:
            continue
        team, pid = str(row.get('team') or ''), str(row.get('player_id') or '')
        if not team or pid in ('', 'None', 'null'):
            continue
        weekly[(season, week, team)].append((pid, str(row.get('player_name') or ''), attempts))

    season_counts = defaultdict(lambda: {'team_weeks': 0, 'dominant_qb_weeks': 0, 'consecutive_dominant_qb_changes': 0})
    previous = {}
    examples = []
    for (season, week, team), candidates in sorted(weekly.items()):
        total = sum(x[2] for x in candidates)
        s = season_counts[season]
        s['team_weeks'] += 1
        # Dominant passer is a transparent proxy, not a verified starter.
        top = max(candidates, key=lambda x: x[2])
        if total < 10 or top[2] < 10 or top[2] / total < 0.65:
            previous[(season, team)] = (week, None)
            continue
        s['dominant_qb_weeks'] += 1
        prev_week, prev_id = previous.get((season, team), (None, None))
        if prev_week == week - 1 and prev_id is not None and prev_id != top[0]:
            s['consecutive_dominant_qb_changes'] += 1
            if len(examples) < 15:
                examples.append({'season': season, 'week': week, 'team': team, 'dominant_passer': top[1],
                                 'pass_attempts': top[2], 'team_pass_attempts': total})
        previous[(season, team)] = (week, top[0])

    return {'seasons': {str(y): season_counts[y] for y in SEASONS},
            'total_team_weeks': sum(v['team_weeks'] for v in season_counts.values()),
            'dominant_qb_weeks': sum(v['dominant_qb_weeks'] for v in season_counts.values()),
            'consecutive_dominant_qb_changes': sum(v['consecutive_dominant_qb_changes'] for v in season_counts.values()),
            'examples': examples}


def main():
    frames = []
    columns_seen = None
    for season in SEASONS:
        df = nfl.load_player_stats([season], summary_level='week')
        cols = set(df.columns)
        if not REQUIRED.issubset(cols):
            raise RuntimeError(f'Campos faltantes en {season}: {sorted(REQUIRED - cols)}. Campos disponibles: {sorted(cols)}')
        columns_seen = sorted(cols)
        frames.extend(df.select(sorted(REQUIRED)).to_dicts())
        print(f'{season}: {df.height} filas de estadísticas de jugadores')
    result = analyze(frames)
    if result['total_team_weeks'] < 1200 or result['dominant_qb_weeks'] < 1000:
        raise RuntimeError(f'Cobertura insuficiente: {result["total_team_weeks"]} team-weeks, {result["dominant_qb_weeks"]} con QB dominante')
    report = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'source': 'nflreadpy.load_player_stats(summary_level=week)',
              'season_range': [SEASONS[0], SEASONS[-1]],
              'selection_rule': 'QB con al menos 10 pases y >=65% de intentos del equipo por semana',
              'metrics': result,
              'schema_verified': True,
              'causal_effect_estimated': False,
              'qb_starter_verified': False,
              'model_modified': False,
              'approved_picks': [],
              'warnings': [
                  'Un pasador dominante NO equivale necesariamente al titular de inicio.',
                  'Los cambios observados no miden el impacto causal de lesiones o cambios de QB.',
                  'Faltan controlar rivales, localía, estado de partido, EPA, y disponibilidad previa.',
                  'Esta auditoría NO autoriza ajustar spreads, totals o probabilidades.'
              ]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Cobertura:', result['total_team_weeks'], 'equipo-semanas; cambios de pasador dominante:', result['consecutive_dominant_qb_changes'])
    print('Publicado:', OUT, 'Picks aprobados: 0')


if __name__ == '__main__':
    main()

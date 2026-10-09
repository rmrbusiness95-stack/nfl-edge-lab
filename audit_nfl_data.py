#!/usr/bin/env python3
"""NFL Edge Lab 3.0: audit of weekly NFLverse statistics.

Read-only diagnostic. Run from repository root after build_nfl_stats.py.
Writes data/quality_report.json. Does not change predictions or picks.
"""
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('data/team_week_stats.json')
OUTPUT = Path('data/quality_report.json')


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def audit(payload):
    records = payload.get('team_week')
    if not isinstance(records, list) or not records:
        raise ValueError('No se encontraron registros team_week.')
    byyear = defaultdict(list)
    duplicates = set()
    seen = set()
    for idx, record in enumerate(records):
        required = ('season', 'week', 'team', 'side', 'plays')
        if any(field not in record for field in required):
            raise ValueError(f'Registro {idx}: faltan campos requeridos.')
        season, week = int(record['season']), int(record['week'])
        team, side = str(record['team']), str(record['side'])
        if side not in ('offense', 'defense') or not (1 <= week <= 25):
            raise ValueError(f'Registro {idx}: semana o lado no válido.')
        key = (season, week, team, side)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
        byyear[season].append(record)
    summaries = []
    for season in sorted(byyear):
        rows = byyear[season]
        weeks = sorted({int(r['week']) for r in rows})
        offense = [r for r in rows if r['side'] == 'offense']
        defense = [r for r in rows if r['side'] == 'defense']
        bad_plays = sum(not finite(r.get('plays')) or float(r['plays']) <= 0 for r in rows)
        bad_epa = sum(not finite(r.get('epa_per_play') if r['side'] == 'offense' else r.get('epa_per_play_allowed')) for r in rows)
        bad_success = sum(not finite(r.get('success_rate') if r['side'] == 'offense' else r.get('success_rate_allowed')) or (finite(r.get('success_rate') if r['side'] == 'offense' else r.get('success_rate_allowed')) and not (0 <= float(r.get('success_rate') if r['side'] == 'offense' else r.get('success_rate_allowed')) <= 1)) for r in rows)
        teams = {r['team'] for r in rows}
        summaries.append({
            'season': season,
            'rows': len(rows),
            'weeks': len(weeks),
            'first_week': min(weeks),
            'last_week': max(weeks),
            'teams': len(teams),
            'offense_rows': len(offense),
            'defense_rows': len(defense),
            'bad_play_counts': bad_plays,
            'bad_epa_values': bad_epa,
            'bad_success_rates': bad_success,
            'duplicate_keys': sum(k[0] == season for k in duplicates),
        })
    return {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_generated_at_utc': payload.get('generated_at_utc'),
        'season_count': len(summaries),
        'weekly_records': len(records),
        'seasons': summaries,
        'warning': 'Cobertura y rangos son controles básicos; no validan fuentes, cuotas ni ausencia de fuga temporal. No genera picks.'
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'No existe {SOURCE}')
    payload = json.loads(SOURCE.read_text(encoding='utf-8'))
    report = audit(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Temporadas:', report['season_count'], '| Registros:', report['weekly_records'])
    for s in report['seasons']:
        print(f"{s['season']}: {s['rows']} registros, {s['weeks']} semanas, "
              f"{s['teams']} equipos, {s['bad_epa_values']} EPA inválidos, "
              f"{s['duplicate_keys']} duplicados")
    print('Informe:', OUTPUT)


if __name__ == '__main__':
    main()

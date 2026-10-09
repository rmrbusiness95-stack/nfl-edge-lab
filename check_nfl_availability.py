"""NFL Edge Lab 3.0: conservatively validate current QB and injury feeds.

Read data/nfl_availability.json and produce data/availability_quality.json.
Never infer starters or current game statuses from historical reports.
No model/picks are modified.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('data/nfl_availability.json')
DEST = Path('data/availability_quality.json')


def utc(value):
    try:
        dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except (TypeError, AttributeError, ValueError):
        return None


def inspect(payload, now=None):
    now = now or datetime.now(timezone.utc)
    qbs = payload.get('qb_depth_chart_candidates', [])
    injuries = payload.get('injury_reports', [])
    season = payload.get('season')
    generated = utc(payload.get('generated_at_utc'))
    qb_by_team = defaultdict(list)
    for qb in qbs:
        if qb.get('team'):
            qb_by_team[qb['team']].append(qb)
    week_nums = [int(row['week']) for row in injuries if str(row.get('week', '')).isdigit()]
    latest_week = max(week_nums) if week_nums else None
    injuries_latest = [row for row in injuries if latest_week is not None and str(row.get('week')) == str(latest_week)]
    injury_by_team = defaultdict(list)
    for row in injuries_latest:
        if row.get('team'):
            injury_by_team[row['team']].append(row)
    report_dates_missing = sum(not utc(row.get('report_date')) for row in injuries_latest)
    teams = {}
    for team in sorted(set(qb_by_team) | set(injury_by_team)):
        q = qb_by_team[team]
        i = injury_by_team[team]
        snapshots = [utc(row.get('snapshot_at_utc')) for row in q]
        dates = [d for d in snapshots if d]
        latest_snapshot = max(dates) if dates else None
        q_age_hours = round((now - latest_snapshot).total_seconds()/3600, 1) if latest_snapshot else None
        game_designations = Counter(str(r.get('game_designation') or 'no designation') for r in i)
        teams[team] = {
            'qb_candidates': len(q),
            'depth_chart_snapshot_at_utc': latest_snapshot.isoformat() if latest_snapshot else None,
            'depth_chart_age_hours': q_age_hours,
            'qb_starter_confirmed': False,
            'injury_report_week': latest_week if i else None,
            'injury_report_rows_in_latest_dataset_week': len(i),
            'game_designations_in_latest_dataset_week': dict(game_designations),
            'injury_report_date_known': bool(i) and all(utc(r.get('report_date')) for r in i),
            'current_game_status_confirmed': False,
            'safe_to_adjust_betting_lines': False,
        }
    warnings = []
    if not generated or (now - generated).total_seconds() > 36*3600:
        warnings.append('El archivo fuente no tiene fecha válida o supera 36 horas de antigüedad.')
    if latest_week is None:
        warnings.append('No existen semanas utilizables de reportes de lesiones.')
    if report_dates_missing:
        warnings.append(f'{report_dates_missing} reportes de la última semana disponible no tienen fecha verificable.')
    if not qbs:
        warnings.append('No se identificaron candidatos QB.')
    return {
        'season': season,
        'checked_at_utc': now.isoformat(),
        'input_generated_at_utc': payload.get('generated_at_utc'),
        'qb_teams': len(qb_by_team),
        'qb_candidate_records': len(qbs),
        'latest_injury_week_in_dataset': latest_week,
        'latest_week_injury_records': len(injuries_latest),
        'latest_week_injury_teams': len(injury_by_team),
        'latest_week_rows_without_verifiable_report_date': report_dates_missing,
        'injury_dataset_week_is_next_game_week': 'unknown',
        'starter_confirmation_available': False,
        'ready_for_betting_model': False,
        'warnings': warnings,
        'teams': teams,
        'note': 'La última semana disponible NO implica reportes vigentes para el próximo partido. No se interpreta depth_rank como titular confirmado.'
    }


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f'Falta {SOURCE}. Ejecutar después de publicar disponibilidad.')
    payload = json.loads(SOURCE.read_text(encoding='utf-8'))
    report = inspect(payload)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Equipos con QB:', report['qb_teams'])
    print('Última semana de lesiones en fuente:', report['latest_injury_week_in_dataset'])
    print('Equipos con lesiones en esa semana:', report['latest_week_injury_teams'])
    print('Estado para apuestas: NO APROBADO')
    for warning in report['warnings']:
        print('AVISO:', warning)


if __name__ == '__main__':
    main()

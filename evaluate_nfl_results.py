"""NFL Edge Lab — settle archived pregame forecasts against verified final scores.

Reads immutable data/prediction_snapshots/*.json and NFLverse games.csv.
Does not change archived snapshots, forecast models or betting selections.
Requires complete score AND kickoff at least 24h ago (avoid provisional scores).
The game data provider is not an official final-score certification; verify discrepancies.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

SNAPSHOTS = Path('data/prediction_snapshots')
REPORT = Path('data/prospective_results.json')
GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
MIN_HOURS_SINCE_KICKOFF = 24


def date_utc(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Datetime without timezone')
    return dt.astimezone(timezone.utc)


def parse_score(value):
    try:
        return int(value) if pd.notna(value) and float(value).is_integer() else None
    except (TypeError, ValueError):
        return None


def main():
    files = sorted(SNAPSHOTS.glob('**/*.json'))
    if not files:
        raise RuntimeError('No hay instantaneas archivadas. No se genera un informe vacio.')
    response = requests.get(GAMES_URL, timeout=60)
    response.raise_for_status()
    schedule = pd.read_csv(StringIO(response.text), low_memory=False)
    required = {'game_id', 'home_score', 'away_score', 'game_type'}
    if not required.issubset(schedule.columns):
        raise ValueError('Faltan columnas del calendario: ' + str(sorted(required-set(schedule.columns))))
    known = {}
    for row in schedule.itertuples(index=False):
        if row.game_type != 'REG':
            continue
        game_id = str(row.game_id)
        h, a = parse_score(row.home_score), parse_score(row.away_score)
        if h is not None and a is not None:
            known[game_id] = (h, a)
    now = datetime.now(timezone.utc)
    snapshots = []
    total_settled = 0
    for path in files:
        saved = json.loads(path.read_text(encoding='utf-8'))
        if not saved.get('games'):
            continue
        created = date_utc(saved['archived_at_utc'])
        model_at = date_utc(saved['model_generated_at_utc'])
        odds_at = date_utc(saved['odds_generated_at_utc'])
        summary = {'snapshot_file': str(path), 'archived_at_utc': created.isoformat(),
                   'model_generated_at_utc': model_at.isoformat(),
                   'odds_generated_at_utc': odds_at.isoformat(),
                   'settled_games': [], 'pending_games': 0, 'invalid_time_games': 0}
        for game in saved['games']:
            kickoff = date_utc(game['kickoff_utc'])
            # Do not treat a later-created forecast or quote as pre-game.
            if max(created, model_at, odds_at) >= kickoff:
                summary['invalid_time_games'] += 1
                continue
            if now < kickoff + timedelta(hours=MIN_HOURS_SINCE_KICKOFF):
                summary['pending_games'] += 1
                continue
            final = known.get(game['game_id'])
            if final is None:
                summary['pending_games'] += 1
                continue
            home, away = final
            actual_margin = home - away
            actual_total = home + away
            model_margin = float(game['model_home_margin_projection'])
            model_total = float(game['model_total_projection'])
            comparisons = []
            for book in game.get('books', []):
                spread = total_line = None
                for market in book.get('markets', []):
                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == game['home_team'] and outcome.get('point') is not None:
                                spread = float(outcome['point'])
                    if market.get('key') == 'totals':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == 'Over' and outcome.get('point') is not None:
                                total_line = float(outcome['point'])
                if spread is None and total_line is None:
                    continue
                comparisons.append({
                    'book': book['title'],
                    'home_spread': spread,
                    'total_line': total_line,
                    'market_margin_absolute_error': round(abs(actual_margin + spread), 3) if spread is not None else None,
                    'market_total_absolute_error': round(abs(actual_total - total_line), 3) if total_line is not None else None,
                    'home_ats_result': ('cover' if actual_margin + spread > 0 else 'no_cover' if actual_margin + spread < 0 else 'push') if spread is not None else None,
                    'over_result': ('over' if actual_total > total_line else 'under' if actual_total < total_line else 'push') if total_line is not None else None,
                })
            summary['settled_games'].append({
                'game_id': game['game_id'], 'home_team': game['home_team'], 'away_team': game['away_team'],
                'home_score': home, 'away_score': away,
                'actual_home_margin': actual_margin, 'actual_total': actual_total,
                'model_home_margin': model_margin, 'model_total': model_total,
                'model_margin_absolute_error': round(abs(actual_margin-model_margin), 3),
                'model_total_absolute_error': round(abs(actual_total-model_total), 3),
                'book_comparisons': comparisons,
            })
        total_settled += len(summary['settled_games'])
        snapshots.append(summary)
    result = {'generated_at_utc': now.isoformat(), 'source': GAMES_URL,
              'snapshots': snapshots, 'snapshot_count': len(snapshots),
              'settled_snapshot_game_records': total_settled,
              'approved_picks': [],
              'notes': [
                  'El informe es retrospectivo sobre pronosticos archivados; no demuestra EV o ROI.',
                  'El mismo encuentro puede aparecer en varias capturas, que NO son partidos independientes.',
                  'Las cuotas de cada casa se conservan como se registraron, no como cierre.',
                  'No se infiere que una discrepancia del modelo represente una apuesta realizada.',
                  'Partidos con menos de 24 horas desde kickoff o sin marcador disponible permanecen pendientes.',
                  'Marcadores segun calendario nflverse; discrepancias deben verificarse externamente.',
              ]}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Instantaneas:', len(snapshots), 'partidos liquidados por instantanea:', total_settled)
    print('Pendientes:', sum(s['pending_games'] for s in snapshots), 'salida:', REPORT)


if __name__ == '__main__':
    main()

"""NFL Edge Lab: independent error diagnosis, not an improved betting strategy.
Runs expanding-season training 2022-2025, computes descriptive residuals.
Market reference is historical nflverse published lines, NOT executable odds.
Does not modify the model, publish picks, or estimate returns.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_nfl_model import SCHEDULE_URL, finite, game_features, load_weekly, ridge_fit, predict

OUT = Path('data/model_error_diagnostics.json')
YEARS = (2022, 2023, 2024, 2025)


def summarize(records):
    if not records:
        return {'n': 0}
    a = np.asarray([v['actual'] for v in records], dtype=float)
    model = np.asarray([v['model'] for v in records], dtype=float)
    market = np.asarray([v['market'] for v in records], dtype=float)
    return {
        'n': len(records),
        'model_mae': round(float(np.abs(a - model).mean()), 3),
        'market_mae': round(float(np.abs(a - market).mean()), 3),
        'model_bias': round(float((model - a).mean()), 3),
        'market_bias': round(float((market - a).mean()), 3),
        'avg_abs_disagreement': round(float(np.abs(model - market).mean()), 3),
        'pct_model_better_than_market': round(float((np.abs(a-model) < np.abs(a-market)).mean() * 100), 1),
    }


def breakdown(rows, field):
    grouped = {}
    for r in rows:
        grouped.setdefault(str(r[field]), []).append(r)
    return {k: summarize(v) for k, v in sorted(grouped.items())}


def main():
    mapping, _ = load_weekly()
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)
    required = {'season','week','game_type','game_id','home_team','away_team',
                'home_score','away_score','spread_line','total_line'}
    if not required.issubset(games.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(games.columns))}')

    records = []
    for row in games.itertuples(index=False):
        if str(row.game_type) != 'REG':
            continue
        try:
            season, week = int(row.season), int(row.week)
        except (TypeError, ValueError):
            continue
        home_score, away_score = finite(row.home_score), finite(row.away_score)
        if home_score is None or away_score is None:
            continue
        f = game_features(mapping, season, week, str(row.home_team), str(row.away_team))
        if f is None:
            continue
        records.append({'year': season, 'week': week, 'features': f,
                        'game_id': str(row.game_id), 'home': str(row.home_team),
                        'away': str(row.away_team), 'margin': home_score - away_score,
                        'total': home_score + away_score,
                        'market_margin': finite(row.spread_line),
                        'market_total': finite(row.total_line)})

    evaluations = {'margin': [], 'total': []}
    year_counts = []
    for year in YEARS:
        train = [r for r in records if r['year'] < year]
        test = [r for r in records if r['year'] == year]
        if len(train) < 500 or len(test) < 100:
            raise ValueError(f'Insufficient data {year}: train={len(train)} test={len(test)}')
        xtrain = np.asarray([r['features'] for r in train], dtype=float)
        xtest = np.asarray([r['features'] for r in test], dtype=float)
        year_counts.append({'year': year, 'train_games': len(train), 'test_games': len(test)})
        for target, market_key in [('margin','market_margin'), ('total','market_total')]:
            fit = ridge_fit(xtrain, [r[target] for r in train])
            predicted = predict(fit, xtest)
            for row, p in zip(test, predicted):
                m = row[market_key]
                if m is None:
                    continue
                diff = abs(float(p) - m)
                bucket = '0-1.5' if diff < 1.5 else '1.5-3' if diff < 3 else '3-5' if diff < 5 else '5+'
                margin_class = ('market_home_favorite' if row['market_margin'] is not None and row['market_margin'] > 0
                                else 'market_away_favorite' if row['market_margin'] is not None and row['market_margin'] < 0
                                else 'market_pickem_or_unknown')
                evaluations[target].append({
                    'year': year, 'game_id': row['game_id'], 'week': row['week'],
                    'home': row['home'], 'away': row['away'],
                    'actual': float(row[target]), 'model': float(p), 'market': float(m),
                    'disagreement_bucket': bucket,
                    'market_favorite': margin_class,
                    'model_above_market': bool(p > m),
                })

    output = {'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'reference': 'nflverse games.csv historical spread_line and total_line',
              'seasons': list(YEARS), 'year_counts': year_counts,
              'approved_picks': [], 'analysis': {}}
    for target, items in evaluations.items():
        output['analysis'][target] = {
            'overall': summarize(items),
            'by_season': breakdown(items, 'year'),
            'by_disagreement': breakdown(items, 'disagreement_bucket'),
            'by_market_favorite': breakdown(items, 'market_favorite'),
            'by_model_above_market': breakdown(items, 'model_above_market'),
            'largest_disagreements': [
                {'year': r['year'], 'game_id': r['game_id'], 'week': r['week'],
                 'model': round(r['model'], 2), 'market': round(r['market'], 2),
                 'actual': round(r['actual'], 2),
                 'abs_line_difference': round(abs(r['model'] - r['market']), 2)}
                for r in sorted(items, key=lambda v: abs(v['model']-v['market']), reverse=True)[:20]
            ],
        }
    output['warnings'] = [
        'Historical market spreads are published references, not independently verified time-stamped executable odds.',
        'This diagnosis does not measure ROI, EV, calibrated cover probability or true betting edge.',
        'Years 2022-2025 were previously inspected; these are diagnostic observations, not an untouched holdout.',
        'Features exclude verified quarterback/injury information and early season games with insufficient prior weekly data.',
        'Large disagreement is NOT evidence for a profitable bet; no model parameters were optimized here.',
    ]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    for target in ('margin','total'):
        print(target, 'overall', output['analysis'][target]['overall'])
        print(target, '5+ point disagreement', output['analysis'][target]['by_disagreement'].get('5+'))
    print('Saved:', OUT)
    print('Approved picks: 0')


if __name__ == '__main__':
    main()

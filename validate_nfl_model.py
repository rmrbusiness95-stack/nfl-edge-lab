"""NFL Edge Lab: expanding-window, year-by-year evaluation.
Uses existing feature pipeline; never trains on the evaluated season.
No market odds, picks, or quarterback adjustments are inferred.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_nfl_model import (SCHEDULE_URL, finite, game_features, load_weekly,
                             ridge_fit, predict)

OUT = Path('data/model_multiyear_validation.json')
TEST_YEARS = (2022, 2023, 2024, 2025)
MIN_TRAIN = 500
MIN_TEST = 100


def evaluate():
    mapping, source_seasons = load_weekly()
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)
    required = {'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score'}
    if not required.issubset(games.columns):
        raise ValueError(f'Calendario sin columnas: {sorted(required - set(games.columns))}')

    # Form features strictly from earlier weeks of the SAME season.
    # Exclude incomplete matches and non-regular season games.
    observations = []
    for g in games.itertuples(index=False):
        try:
            year, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        if str(getattr(g, 'game_type', 'REG')) != 'REG':
            continue
        hs, aws = finite(g.home_score), finite(g.away_score)
        if hs is None or aws is None:
            continue
        features = game_features(mapping, year, week, str(g.home_team), str(g.away_team))
        if features is None:
            continue
        observations.append({'year': year, 'x': features, 'margin': hs - aws, 'total': hs + aws})

    reports = []
    for test_year in TEST_YEARS:
        # Expanding-window: pre-2022 for 2022, pre-2023 for 2023, etc.
        train = [g for g in observations if g['year'] < test_year]
        test = [g for g in observations if g['year'] == test_year]
        row = {'test_season': test_year, 'train_games': len(train), 'test_games': len(test),
               'train_seasons': sorted({g['year'] for g in train})}
        if len(train) < MIN_TRAIN or len(test) < MIN_TEST:
            row['status'] = 'insufficient_data'
            reports.append(row)
            continue
        xtr = np.asarray([g['x'] for g in train], dtype=float)
        xte = np.asarray([g['x'] for g in test], dtype=float)
        row['status'] = 'evaluated'
        for field in ('margin', 'total'):
            ytr = np.asarray([g[field] for g in train], dtype=float)
            yt = np.asarray([g[field] for g in test], dtype=float)
            fit = ridge_fit(xtr, ytr)
            yp = predict(fit, xte)
            # Global historical mean is intentionally a weak benchmark.
            base = np.full(len(yt), ytr.mean())
            row[f'{field}_mae'] = round(float(np.mean(np.abs(yt - yp))), 3)
            row[f'{field}_baseline_mae'] = round(float(np.mean(np.abs(yt - base))), 3)
            row[f'{field}_mean_bias'] = round(float(np.mean(yp - yt)), 3)
        reports.append(row)

    evaluated = [r for r in reports if r['status'] == 'evaluated']
    if not evaluated:
        raise RuntimeError('No hubo temporadas de prueba suficientes; no se publica un informe vacío.')
    result = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation': 'expanding_window_by_season',
        'feature_window_weeks': 8,
        'source_seasons': source_seasons,
        'results': reports,
        'approved_picks': [],
        'market_odds_validated': False,
        'notes': [
            'Las temporadas 2022-2025 ya se han examinado durante el desarrollo: esto es diagnóstico, no holdout nuevo.',
            'Cada temporada se evalúa usando exclusivamente partidos de temporadas previas para ajustar coeficientes.',
            'Las características de un partido usan semanas anteriores de la misma temporada.',
            'La referencia es una media global simple, no un spread de mercado.',
            'Sin cuotas históricas punto-en-tiempo no es posible calcular ROI, EV ni una ventaja sobre las casas.',
            'No se incorporan lesionados ni quarterbacks no confirmados.',
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in reports:
        print(r['test_season'], r['status'], 'partidos', r['test_games'],
              'MAE margen', r.get('margin_mae'), 'MAE total', r.get('total_mae'))
    print('Apuestas aprobadas: 0. Falta comparación con mercado y validación prospectiva.')


if __name__ == '__main__':
    evaluate()

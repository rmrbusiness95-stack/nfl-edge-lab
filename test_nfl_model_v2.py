"""NFL Edge Lab: prueba controlada de contexto prepartido.

Modelo vigente vs candidato con descanso y etapa de la temporada.
No altera pronósticos publicados, ni autoriza picks. Las líneas nflverse
son referencias históricas, no cuotas ejecutables punto-en-tiempo.
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

YEARS = (2022, 2023, 2024, 2025)
OUT = Path('data/model_v2_context_test.json')


def mae(a, b):
    return round(float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))), 3)


def summary(rows):
    result = {'games': len(rows)}
    if not rows:
        return result
    for target in ('margin', 'total'):
        result[target] = {
            'original_mae': mae([r[target] for r in rows], [r[f'original_{target}'] for r in rows]),
            'candidate_mae': mae([r[target] for r in rows], [r[f'candidate_{target}'] for r in rows]),
            'market_reference_mae': mae([r[target] for r in rows], [r[f'market_{target}'] for r in rows]),
        }
        result[target]['candidate_minus_original_mae'] = round(
            result[target]['candidate_mae'] - result[target]['original_mae'], 3)
    return result


def main():
    mapping, _ = load_weekly()
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)
    needed = {'season', 'week', 'game_type', 'home_team', 'away_team',
              'home_score', 'away_score', 'home_rest', 'away_rest',
              'spread_line', 'total_line'}
    if not needed.issubset(games.columns):
        raise ValueError('Columnas ausentes: ' + ', '.join(sorted(needed - set(games.columns))))

    observations = []
    for g in games.itertuples(index=False):
        if str(g.game_type) != 'REG':
            continue
        try:
            season, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        hs, aws = finite(g.home_score), finite(g.away_score)
        hr, ar = finite(g.home_rest), finite(g.away_rest)
        market_spread, market_total = finite(g.spread_line), finite(g.total_line)
        if any(v is None for v in (hs, aws, hr, ar, market_spread, market_total)):
            continue
        features = game_features(mapping, season, week, str(g.home_team), str(g.away_team))
        if features is None:
            continue
        # Contexto disponible antes del inicio del juego; no usar marcadores futuros.
        # Limites de descanso previenen que valores anomalos dominen el ajuste.
        context = [min(max(hr, 3), 16) - min(max(ar, 3), 16),
                   int(week >= 17),
                   min(week, 18) / 18.0]
        observations.append({'season': season, 'week': week,
                             'x': features, 'xc': features + context,
                             'margin': hs-aws, 'total': hs+aws,
                             'market_margin': market_spread,
                             'market_total': market_total})

    results, evaluated_rows = [], []
    for year in YEARS:
        train = [r for r in observations if r['season'] < year]
        test = [r for r in observations if r['season'] == year]
        if len(train) < 500 or len(test) < 100:
            raise RuntimeError(f'Datos insuficientes para {year}: train={len(train)}, test={len(test)}')
        xtr = np.asarray([r['x'] for r in train], dtype=float)
        xte = np.asarray([r['x'] for r in test], dtype=float)
        xctr = np.asarray([r['xc'] for r in train], dtype=float)
        xcte = np.asarray([r['xc'] for r in test], dtype=float)
        predictions = {}
        for target in ('margin', 'total'):
            ytr = np.asarray([r[target] for r in train], dtype=float)
            predictions[f'original_{target}'] = predict(ridge_fit(xtr, ytr), xte)
            predictions[f'candidate_{target}'] = predict(ridge_fit(xctr, ytr), xcte)
        yearly = []
        for i, r in enumerate(test):
            yearly.append({
                'season': year, 'week': r['week'],
                'margin': r['margin'], 'total': r['total'],
                'market_margin': r['market_margin'], 'market_total': r['market_total'],
                **{name: float(values[i]) for name, values in predictions.items()},
            })
        evaluated_rows.extend(yearly)
        results.append({'season': year, **summary(yearly),
                        'late_weeks_17_18': summary([r for r in yearly if r['week'] >= 17])})

    overall = summary(evaluated_rows)
    # Es una decisión experimental, no automática. Exigir mejoras por ambos objetivos
    # y al menos tres años por objetivo antes de siquiera considerar promoción.
    improved_years = {
        t: sum(r[t]['candidate_mae'] < r[t]['original_mae'] for r in results)
        for t in ('margin', 'total')
    }
    output = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'add_pregame_rest_difference_and_week_phase',
        'train_test_rule': 'train prior seasons, test current season',
        'historical_test_seasons': list(YEARS),
        'yearly': results, 'overall': overall, 'improved_years': improved_years,
        'automatic_promotion': False, 'approved_picks': [],
        'warnings': [
            'Investigacion retrospectiva, NO evaluacion completamente independiente: 2022-2025 ya se examinaron.',
            'La nueva seleccion de variables surge del diagnostico en esos anos; riesgo de sobreajuste por investigacion.',
            'Lineas historicas nflverse son referencias, NO cuotas verificadas punto-en-tiempo.',
            'No mide ROI, EV, probabilidades calibradas ni ventaja de apuestas.',
            'No incorpora datos historicos verificados de quarterback/lesiones.',
            'La prueba NO reemplaza build_nfl_model.py ni las proyecciones publicadas.',
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    for r in results:
        print(r['season'], 'juegos', r['games'],
              'margen original/candidato/mercado', *(r['margin'][k] for k in ('original_mae','candidate_mae','market_reference_mae')),
              'total original/candidato/mercado', *(r['total'][k] for k in ('original_mae','candidate_mae','market_reference_mae')))
    print('MEJORAS POR TEMPORADA:', improved_years)
    print('RESULTADO:', OUT, 'PICKS APROBADOS: 0')


if __name__ == '__main__':
    main()

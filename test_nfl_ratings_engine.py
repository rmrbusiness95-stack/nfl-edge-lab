"""NFL Edge Lab: prueba de motor alternativo de fuerza ofensiva/defensiva.

Ratings causales: cada pronostico usa exclusivamente partidos anteriores.
El mercado se usa SOLO para evaluar, nunca para aprender ratings.
Compara en la MISMA muestra contra el modelo original y mercado (2022-2025).
Exploratorio: no calibra probabilidades ni autoriza apuestas.
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

OUT = Path('data/ratings_engine_test.json')
YEARS = (2022, 2023, 2024, 2025)
BASE_POINTS = 22.0
HOME_ADVANTAGE_POINTS = 2.0
LEARNING_RATE = 0.12
OFFSEASON_KEEP = 0.65


def mae(actual, estimate):
    return round(float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(estimate, dtype=float)))), 3)


def describe(rows):
    result = {'games': len(rows)}
    for field in ('margin', 'total'):
        if not rows:
            result[field] = None
            continue
        y = [r[field] for r in rows]
        result[field] = {
            'original_mae': mae(y, [r['original_' + field] for r in rows]),
            'ratings_mae': mae(y, [r['ratings_' + field] for r in rows]),
            'market_mae': mae(y, [r['market_' + field] for r in rows]),
        }
        result[field]['ratings_minus_original'] = round(
            result[field]['ratings_mae'] - result[field]['original_mae'], 3)
    return result


def main():
    mapping, _ = load_weekly()
    resp = requests.get(SCHEDULE_URL, timeout=60)
    resp.raise_for_status()
    games = pd.read_csv(StringIO(resp.text), low_memory=False)
    required = {'season','week','gameday','game_id','game_type','home_team','away_team',
                'home_score','away_score','spread_line','total_line'}
    if not required.issubset(games.columns):
        raise ValueError('Columnas faltantes: ' + str(sorted(required - set(games.columns))))
    games = games[games['game_type'].eq('REG')].copy()
    games['date_sort'] = pd.to_datetime(games['gameday'], errors='coerce')
    games = games.sort_values(['season','week','date_sort','game_id'])

    # Calificaciones centradas en cero. Ofensiva mayor=mejor; defensa mayor=mejor.
    # Se calculan antes de observar el resultado de cada partido.
    ratings = {}
    last_season = None
    observations = []
    for g in games.itertuples(index=False):
        try:
            season, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        if season < 2002:
            continue
        if last_season != season:
            if last_season is not None:
                ratings = {team: [o * OFFSEASON_KEEP, d * OFFSEASON_KEEP]
                           for team, (o, d) in ratings.items()}
            last_season = season
        home, away = str(g.home_team), str(g.away_team)
        home_off, home_def = ratings.get(home, [0.0, 0.0])
        away_off, away_def = ratings.get(away, [0.0, 0.0])
        expected_home = BASE_POINTS + home_off - away_def + HOME_ADVANTAGE_POINTS/2
        expected_away = BASE_POINTS + away_off - home_def - HOME_ADVANTAGE_POINTS/2
        pred_margin = expected_home - expected_away
        pred_total = expected_home + expected_away
        hs, aws = finite(g.home_score), finite(g.away_score)
        if hs is None or aws is None:
            continue  # No actualizar ratings sin marcador final
        features = game_features(mapping, season, week, home, away)
        spread, total_line = finite(g.spread_line), finite(g.total_line)
        if season in YEARS and features is not None and spread is not None and total_line is not None:
            observations.append({
                'season': season, 'week': week, 'x': features,
                'margin': hs - aws, 'total': hs + aws,
                'ratings_margin': pred_margin, 'ratings_total': pred_total,
                'market_margin': spread, 'market_total': total_line,
            })
        # Actualizacion simetrica usando residuos que ajustan por rival y localia.
        err_home = hs - expected_home
        err_away = aws - expected_away
        ratings[home] = [home_off + LEARNING_RATE * err_home,
                         home_def - LEARNING_RATE * err_away]
        ratings[away] = [away_off + LEARNING_RATE * err_away,
                         away_def - LEARNING_RATE * err_home]

    # Construir tambien el historial previo necesario para entrenar modelo original.
    training_rows = []
    for g in games.itertuples(index=False):
        try:
            season, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        if season >= max(YEARS):
            # El entrenamiento se filtra por año posteriormente.
            pass
        hs, aws = finite(g.home_score), finite(g.away_score)
        if hs is None or aws is None:
            continue
        feature = game_features(mapping, season, week, str(g.home_team), str(g.away_team))
        if feature is not None:
            training_rows.append({'season': season, 'x': feature,
                                  'margin': hs - aws, 'total': hs + aws})

    reports, pooled = [], []
    for year in YEARS:
        train = [r for r in training_rows if r['season'] < year]
        test = [r for r in observations if r['season'] == year]
        if len(train) < 500 or len(test) < 100:
            raise RuntimeError(f'Muestra insuficiente {year}: train={len(train)}, test={len(test)}')
        for field in ('margin','total'):
            model = ridge_fit([r['x'] for r in train], [r[field] for r in train])
            preds = predict(model, [r['x'] for r in test])
            for r, p in zip(test, preds):
                r['original_' + field] = float(p)
        pooled.extend(test)
        reports.append({'season': year, **describe(test),
                        'weeks_17_18': describe([r for r in test if r['week'] >= 17])})
    out = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': 'causal_offense_defense_ratings_with_home_advantage',
        'parameters_fixed_in_advance': {
            'base_points': BASE_POINTS,
            'home_advantage_points': HOME_ADVANTAGE_POINTS,
            'learning_rate': LEARNING_RATE,
            'offseason_keep': OFFSEASON_KEEP,
        },
        'yearly': reports, 'overall': describe(pooled),
        'approved_picks': [], 'automatic_promotion': False,
        'warnings': [
            'Exploratorio: temporadas 2022-2025 ya fueron examinadas varias veces.',
            'Ratings se actualizan despues de cada marcador; nunca con juegos futuros.',
            'El modelo de ratings parte de parametros fijos no optimizados.',
            'Las lineas historicas del mercado no son precios apostables verificados punto-en-tiempo.',
            'No estima EV, rentabilidad, probabilidades calibradas ni ajustes de lesiones/QB.',
            'La muestra de evaluacion es comun entre ratings, modelo original y mercado.',
            'No modifica el motor original ni publica picks.',
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in reports:
        print(r['season'], r['games'], 'spread', r['margin'], 'total', r['total'])
    print('GLOBAL:', out['overall'])
    print('Archivo:', OUT, 'Picks aprobados: 0')

if __name__ == '__main__':
    main()

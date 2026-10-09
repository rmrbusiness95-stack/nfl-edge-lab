"""NFL Edge Lab: modelo exploratorio de margen y total sin fuga temporal.

Entrada: data/team_week_stats.json (generado por GitHub Actions)
Calendario: NFLverse nfldata games.csv (descarga automatica)
Salida: data/model_diagnostics.json (sin picks aprobados)
No usa lesiones historicas sin fecha, ni afirma obtener valor esperado.
"""
from __future__ import annotations
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = Path('data')
SOURCE = BASE / 'team_week_stats.json'
OUTPUT = BASE / 'model_diagnostics.json'
SCHEDULE_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
WINDOW = 8  # predeterminado antes de esta prueba; no optimizar con 2025
MIN_GAMES = 4
ALPHA = 20.0  # regularizacion fija: diagnostico, no parametros optimizados
FEATURES = ['home_off_epa', 'away_off_epa', 'home_def_epa_allowed', 'away_def_epa_allowed',
            'home_off_success', 'away_off_success', 'home_def_success_allowed',
            'away_def_success_allowed', 'off_epa_diff', 'def_epa_diff']


def finite(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (ValueError, TypeError):
        return None


def load_weekly():
    if not SOURCE.exists():
        raise FileNotFoundError('Falta data/team_week_stats.json. Ejecuta primero Actualizar datos NFL.')
    data = json.loads(SOURCE.read_text(encoding='utf-8'))
    rows = data.get('team_week', [])
    if not rows:
        raise ValueError('team_week vacio; no se puede entrenar.')
    by_team = defaultdict(dict)
    for r in rows:
        try:
            season, week = int(r['season']), int(r['week'])
            team, side = str(r['team']), str(r['side'])
        except (KeyError, ValueError, TypeError):
            continue
        if side not in ('offense', 'defense'):
            continue
        if finite(r.get('plays')) is None or float(r['plays']) <= 0:
            continue
        by_team[(season, team, side)][week] = r
    return by_team, data.get('seasons', [])


def prior_average(mapping, season, team, side, week, key):
    hist = [(w, r) for w, r in mapping.get((season, team, side), {}).items() if w < week]
    hist.sort(key=lambda x: x[0], reverse=True)
    values = [finite(r.get(key)) for _, r in hist[:WINDOW]]
    values = [v for v in values if v is not None]
    if len(values) < MIN_GAMES:
        return None
    return sum(values) / len(values)


def game_features(mapping, season, week, home, away):
    # Todas las semanas consultadas son estrictamente anteriores al encuentro.
    requests_ = [
        ('home_off_epa', home, 'offense', 'epa_per_play'),
        ('away_off_epa', away, 'offense', 'epa_per_play'),
        ('home_def_epa_allowed', home, 'defense', 'epa_per_play_allowed'),
        ('away_def_epa_allowed', away, 'defense', 'epa_per_play_allowed'),
        ('home_off_success', home, 'offense', 'success_rate'),
        ('away_off_success', away, 'offense', 'success_rate'),
        ('home_def_success_allowed', home, 'defense', 'success_rate_allowed'),
        ('away_def_success_allowed', away, 'defense', 'success_rate_allowed'),
    ]
    result = {}
    for name, team, side, metric in requests_:
        result[name] = prior_average(mapping, season, team, side, week, metric)
        if result[name] is None:
            return None
    result['off_epa_diff'] = result['home_off_epa'] - result['away_off_epa']
    result['def_epa_diff'] = result['away_def_epa_allowed'] - result['home_def_epa_allowed']
    return [result[name] for name in FEATURES]


def ridge_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    means = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - means) / std
    xm = np.column_stack([np.ones(len(xs)), xs])
    penalty = np.eye(xm.shape[1]) * ALPHA
    penalty[0, 0] = 0.0
    coeff = np.linalg.solve(xm.T @ xm + penalty, xm.T @ y)
    return {'mean': means, 'std': std, 'coeff': coeff}


def predict(fit, x):
    a = (np.asarray(x, dtype=float) - fit['mean']) / fit['std']
    if a.ndim == 1:
        a = a[None, :]
    return fit['coeff'][0] + a @ fit['coeff'][1:]


def mae(a, b):
    return round(float(np.abs(np.asarray(a) - np.asarray(b)).mean()), 3)


def main():
    mapping, seasons_available = load_weekly()
    r = requests.get(SCHEDULE_URL, timeout=60)
    r.raise_for_status()
    from io import StringIO
    games = pd.read_csv(StringIO(r.text), low_memory=False)
    fields = {'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score'}
    if not fields.issubset(games.columns):
        raise ValueError(f'Faltan campos en games.csv: {sorted(fields - set(games.columns))}')
    data = []
    prospective = []
    now = datetime.now(timezone.utc).isoformat()
    latest_season = max(int(s) for s in seasons_available)
    for row in games.itertuples(index=False):
        try:
            season, week = int(row.season), int(row.week)
            home, away = str(row.home_team), str(row.away_team)
        except (ValueError, TypeError, AttributeError):
            continue
        if getattr(row, 'game_type', 'REG') != 'REG':
            continue
        features = game_features(mapping, season, week, home, away)
        if features is None:
            continue
        hs, as_ = finite(row.home_score), finite(row.away_score)
        record = {'season': season, 'week': week, 'home': home, 'away': away, 'x': features,
                  'game_id': str(getattr(row, 'game_id', ''))}
        if hs is not None and as_ is not None:
            record['margin'] = hs - as_
            record['total'] = hs + as_
            data.append(record)
        elif season == latest_season:
            prospective.append(record)

    # TRAIN: 2002-2024. TEST: 2025, aunque 2025 ya fue examinado
    # en etapas exploratorias anteriores; NO es una validacion intocada.
    train = [v for v in data if v['season'] <= 2024]
    test = [v for v in data if v['season'] == 2025]
    if len(train) < 500:
        raise RuntimeError(f'Solo {len(train)} partidos historicos elegibles de entrenamiento (minimo 500).')
    if len(test) < 100:
        raise RuntimeError(f'Solo {len(test)} partidos de 2025 con variables previas (minimo 100).')
    xtr = [v['x'] for v in train]
    fm = ridge_fit(xtr, [v['margin'] for v in train])
    ft = ridge_fit(xtr, [v['total'] for v in train])
    xte = [v['x'] for v in test]
    margin_pred, total_pred = predict(fm, xte), predict(ft, xte)
    margin_true = np.array([v['margin'] for v in test])
    total_true = np.array([v['total'] for v in test])
    baseline_margin = np.repeat(np.mean([v['margin'] for v in train]), len(test))
    baseline_total = np.repeat(np.mean([v['total'] for v in train]), len(test))
    # Solo demostrar si hay mejora sobre una referencia ingenua.
    results = {'season': 2025, 'games': len(test),
               'model_margin_mae': mae(margin_true, margin_pred),
               'baseline_margin_mae': mae(margin_true, baseline_margin),
               'model_total_mae': mae(total_true, total_pred),
               'baseline_total_mae': mae(total_true, baseline_total)}
    upcoming = []
    if latest_season >= 2026:
        for game in prospective:
            if game['season'] != latest_season:
                continue
            pred_m = float(predict(fm, game['x'])[0])
            pred_t = float(predict(ft, game['x'])[0])
            upcoming.append({'game_id': game['game_id'], 'season': game['season'],
                             'week': game['week'], 'home_team': game['home'], 'away_team': game['away'],
                             'home_margin_projection': round(pred_m, 2),
                             'game_total_projection': round(pred_t, 2),
                             'status': 'experimental_unvalidated'})
        upcoming.sort(key=lambda x: (x['season'], x['week'], x['game_id']))
    result = {'generated_at_utc': now, 'source': 'team_week_stats + nflverse games.csv',
              'feature_window': WINDOW, 'min_prior_games': MIN_GAMES,
              'train_seasons': sorted({v['season'] for v in train}), 'train_games': len(train),
              'holdout': results, 'prospective': upcoming,
              'approved_picks': [], 'model_uses_qb_injuries': False,
              'warnings': [
                  '2025 ya fue examinado antes y NO cuenta como holdout independiente definitivo.',
                  'No hay momios punto-en-tiempo, EV, ROI ni validacion prospectiva.',
                  'Los reportes de lesiones carecen de fechas verificables: NO se aplican ajustes.',
                  'Las proyecciones sin confirmacion de QB son experimentales; no apostar por estas cifras.'
              ]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ENTRENAMIENTO:', len(train), 'partidos. EVALUACION 2025:', len(test), 'partidos.')
    print('MAE margen modelo/baseline:', results['model_margin_mae'], results['baseline_margin_mae'])
    print('MAE total modelo/baseline:', results['model_total_mae'], results['baseline_total_mae'])
    print('Proyecciones experimentales de partidos no jugados:', len(upcoming))
    print('0 picks aprobados: falta validacion frente al mercado.')


if __name__ == '__main__':
    main()

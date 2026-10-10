"""NFL Edge Lab: historical audit of extreme projected game margins.

Year-by-year expanding-window training on earlier seasons only (2022–2025
are retrospective checks, NOT untouched holdouts); actual features for each
match use only earlier weeks via build_nfl_model.game_features.

Does NOT use or infer betting-market lines, QB status, EV, or approved picks.
Results should not be used as pick recommendations or post-hoc tuning data.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_nfl_model import (load_weekly, game_features, ridge_fit, predict,
                             finite, SCHEDULE_URL)

OUT = Path('data/extreme_projection_audit.json')
EVAL_YEARS = (2022, 2023, 2024, 2025)
THRESHOLDS = (0, 7, 10, 14)


def stats(rows):
    if not rows:
        return {'n': 0}
    p = np.array([r['predicted_home_margin'] for r in rows], float)
    a = np.array([r['actual_home_margin'] for r in rows], float)
    error = p-a
    # Predicted team's actual winning margin (positive means selected side won).
    predicted_side_margin = np.sign(p)*a
    return {
        'n': len(rows),
        'margin_mae': round(float(np.mean(np.abs(error))), 3),
        'margin_mean_signed_error': round(float(np.mean(error)), 3),
        'mean_predicted_abs_margin': round(float(np.mean(np.abs(p))), 3),
        'mean_actual_margin_in_predicted_direction': round(float(np.mean(predicted_side_margin)), 3),
        'prediction_direction_correct_rate': round(float(np.mean(predicted_side_margin > 0)), 4),
        'actual_win_by_10_or_more_rate': round(float(np.mean(predicted_side_margin >= 10)), 4),
        'overstates_realized_directional_margin_by_points': round(float(np.mean(np.abs(p) - predicted_side_margin)), 3),
    }


def main():
    mapping, _ = load_weekly()
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)
    required = {'season','week','home_team','away_team','home_score','away_score','game_type'}
    if not required.issubset(games.columns):
        raise RuntimeError('Faltan columnas: '+', '.join(sorted(required-set(games.columns))))
    finished=[]
    for g in games.itertuples(index=False):
        try:
            y,w = int(g.season),int(g.week)
            home,away = str(g.home_team),str(g.away_team)
        except (ValueError,TypeError,AttributeError):
            continue
        if g.game_type != 'REG' or y > max(EVAL_YEARS):
            continue
        hs,ats=finite(g.home_score),finite(g.away_score)
        if hs is None or ats is None:
            continue
        x=game_features(mapping,y,w,home,away)
        if x is None:
            continue
        finished.append({'year':y,'week':w,'home':home,'away':away,
                         'x':x,'actual_home_margin':hs-ats})
    records=[]
    model_train_sizes={}
    for target_year in EVAL_YEARS:
        train=[g for g in finished if g['year'] < target_year]
        test=[g for g in finished if g['year'] == target_year]
        if len(train)<500 or len(test)<100:
            raise RuntimeError(f'Cobertura insuficiente {target_year}: entrenamiento {len(train)}, prueba {len(test)}')
        fitted=ridge_fit([g['x'] for g in train],[g['actual_home_margin'] for g in train])
        preds=predict(fitted,[g['x'] for g in test]).reshape(-1)
        model_train_sizes[str(target_year)]=len(train)
        for g,p in zip(test,preds):
            records.append({'year':target_year,'week':g['week'],'home':g['home'],
                            'away':g['away'],'predicted_home_margin':float(p),
                            'actual_home_margin':g['actual_home_margin']})

    overall={}
    for threshold in THRESHOLDS:
        eligible=[r for r in records if abs(r['predicted_home_margin'])>=threshold]
        overall[str(threshold)]=stats(eligible)
    by_year={}
    for y in EVAL_YEARS:
        yr=[r for r in records if r['year']==y]
        by_year[str(y)]={str(t):stats([r for r in yr if abs(r['predicted_home_margin'])>=t]) for t in THRESHOLDS}

    # Specific archived 2026 example; not a historical prediction error.
    future_example=None
    model_diag=Path('data/model_diagnostics.json')
    if model_diag.exists():
        diag=json.loads(model_diag.read_text(encoding='utf-8'))
        match=next((g for g in diag.get('prospective',[]) if g.get('game_id')=='2026_05_PHI_JAX'),None)
        if match:
            future_example={'game_id':match['game_id'],
                            'home_margin_projection':match.get('home_margin_projection'),
                            'evaluation_status':'prospective_unscored'}
    report={
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'evaluation_years':list(EVAL_YEARS),
        'model':'same ridge feature-generation and alpha as build_nfl_model.py; retrained on prior years for each year',
        'yearly_training_games':model_train_sizes,
        'evaluated_games':len(records),
        'thresholds_absolute_projected_margin':overall,
        'yearly_thresholds':by_year,
        'example_2026_phi_jax':future_example,
        'live_market_lines_used':False,
        'historical_market_lines_used':False,
        'qb_availability_adjusted':False,
        'ready_for_betting':False,
        'model_modified':False,
        'approved_picks':[],
        'warnings':[
            'Los años 2022-2025 son evaluaciones retrospectivas ya examinadas durante el proyecto, no holdouts independientes.',
            'Se evalúa la magnitud extrema absoluta de nuestro margen, NO una discrepancia frente a la línea del casino.',
            'El margen proyectado para Jacksonville en 2026 no se incluye como partido jugado.',
            'Una precisión de signo no equivale a cubrir spreads ni implica retorno positivo.',
            'No ajusta por QB, lesiones, viaje internacional o diferencias de fuerza específica de rivales más allá de las variables existentes.',
            'No optimizar umbrales con este mismo conjunto para luego afirmar validación independiente.'
        ]
    }
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Juegos evaluados:',len(records),'| tamaños de entrenamiento:',model_train_sizes)
    for t,result in overall.items():
        print(f'Margen proyectado >= {t} puntos: {result}')
    print('Archivo:',OUT,'| Picks aprobados: 0')

if __name__=='__main__':
    main()

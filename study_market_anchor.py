"""Auditoría retrospectiva: ¿aportan estadísticas NFL una corrección al mercado de cierre?

Las líneas de cierre NO representan precios disponibles anticipadamente.
No calcula EV, CLV ni rentabilidad. No modifica el modelo publicado.
"""
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from build_nfl_model import load_weekly, game_features, ridge_fit, predict, finite, SCHEDULE_URL

YEARS=(2022,2023,2024,2025)
OUTPUT=Path('data/market_anchor_study.json')

def evaluate(rows, kind):
    results={}; old=[]; new=[]
    for year in YEARS:
        train=[r for r in rows if r['season']<year]
        test=[r for r in rows if r['season']==year]
        if len(train)<500 or len(test)<100:
            raise RuntimeError(f'Cobertura insuficiente {year}: train={len(train)} test={len(test)}')
        fit=ridge_fit([r['x'] for r in train], [r[kind+'_actual']-r[kind+'_market'] for r in train])
        base=np.array([r[kind+'_market'] for r in test],float)
        actual=np.array([r[kind+'_actual'] for r in test],float)
        adjusted=base+np.asarray(predict(fit,[r['x'] for r in test]),float).reshape(-1)
        baseline=np.abs(actual-base); experimental=np.abs(actual-adjusted)
        results[str(year)]={'games':len(test),'train_games':len(train),
            'market_mae':round(float(baseline.mean()),4),
            'market_plus_stats_mae':round(float(experimental.mean()),4),
            'improvement_positive_is_better':round(float(baseline.mean()-experimental.mean()),4)}
        old.extend(baseline.tolist());new.extend(experimental.tolist())
    return {'by_year':results,'evaluated_games':len(old),'market_mae':round(float(np.mean(old)),4),
        'market_plus_stats_mae':round(float(np.mean(new)),4),
        'improvement_positive_is_better':round(float(np.mean(old)-np.mean(new)),4)}

def main():
    mapping,_=load_weekly()
    response=requests.get(SCHEDULE_URL,timeout=60);response.raise_for_status()
    schedule=pd.read_csv(StringIO(response.text),low_memory=False)
    required={'season','week','game_type','home_team','away_team','home_score','away_score','spread_line','total_line'}
    if not required.issubset(schedule.columns):
        raise RuntimeError('Faltan columnas: '+str(sorted(required-set(schedule.columns))))
    rows=[]
    for g in schedule.itertuples(index=False):
        try:y,w=int(g.season),int(g.week)
        except (ValueError,TypeError,AttributeError):continue
        if g.game_type!='REG' or y>max(YEARS):continue
        hs,aws,s,t=(finite(x) for x in (g.home_score,g.away_score,g.spread_line,g.total_line))
        if any(x is None for x in (hs,aws,s,t)):continue
        x=game_features(mapping,y,w,str(g.home_team),str(g.away_team))
        if x is None:continue
        rows.append({'season':y,'x':x,'margin_actual':hs-aws,'margin_market':s,
                     'total_actual':hs+aws,'total_market':t})
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
       'market_source':'nflverse nfldata games.csv closing lines',
       'market_data_are_closing':True,'can_assume_bettable_before_kickoff':False,
       'evaluation_years':list(YEARS),
       'margin':evaluate(rows,'margin'),'total':evaluate(rows,'total'),
       'model_modified':False,'approved_picks':[],
       'warnings':['Auditoria retrospectiva: los cierres NO prueban disponibilidad previa para apostar.',
        '2022-2025 ya fueron revisados y NO son un holdout intacto.',
        'Sin validacion de EV, retorno, momios disponibles ni ventaja real.',
        'La regresion solo puede modificar el residuo de la linea de cierre; no hay ajustes por QB o lesion.',
        'No modificar el modelo principal ni convertir estas diferencias en picks.']}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'margin':result['margin'],'total':result['total']},indent=2))

if __name__=='__main__': main()

"""NFL Edge Lab — prueba contextual exploratoria sobre cambios de pasador, 2018-2025.

NO mide efecto causal, lesiones ni titularidad. 'QB dominante' se determina por
intentos al terminar el partido; esta variable NO existe necesariamente antes del
kickoff y la prueba NO autoriza aplicar ajustes en pronósticos ni calcular EV.

Para cada partido usa SOLO partidos anteriores para medir fortaleza de equipos.
Compara regresiones Ridge de resultado real con/sin indicador de cambio QB.
La evaluación temporal 2024-2025 NO es holdout intocado del proyecto.
"""
from __future__ import annotations
import csv
import io
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import nflreadpy as nfl
import requests

YEARS = tuple(range(2018, 2026))
GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
OUT = Path('data/qb_context_study.json')
WINDOW = 5
MIN_PRIOR = 3
ALPHA = 40.0

def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (ValueError, TypeError):
        return None

def dominant_qbs():
    weekly = defaultdict(list)
    for year in YEARS:
        df = nfl.load_player_stats([year], summary_level='week')
        need = {'season','week','team','player_id','position','attempts'}
        if not need.issubset(set(df.columns)):
            raise RuntimeError(f'Faltan columnas de QB {year}: {sorted(need-set(df.columns))}')
        for row in df.select(sorted(need)).to_dicts():
            if row['position'] != 'QB' or not row['team'] or not row['player_id']:
                continue
            attempts = num(row['attempts'])
            if attempts is None or attempts < 0:
                continue
            weekly[(int(row['season']),int(row['week']),str(row['team']))].append((str(row['player_id']),attempts))
    ans = {}
    for key, ps in weekly.items():
        tot = sum(p[1] for p in ps)
        if not ps or tot < 10:
            continue
        qb, n = max(ps,key=lambda p:p[1])
        if n>=10 and n/tot>=0.65:
            ans[key] = qb
    return ans

def games_table():
    response = requests.get(GAMES_URL,timeout=60)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    fields = {'season','week','game_type','home_team','away_team','home_score','away_score'}
    if not fields.issubset(set(reader.fieldnames or [])):
        raise RuntimeError('Faltan columnas en calendario')
    games=[]
    for r in reader:
        try:
            y,w = int(r['season']),int(r['week'])
        except (ValueError,TypeError):
            continue
        hs,as_ = num(r.get('home_score')),num(r.get('away_score'))
        if y not in YEARS or r['game_type']!='REG' or hs is None or as_ is None:
            continue
        games.append((y,w,str(r['home_team']),str(r['away_team']),hs,as_))
    games.sort(key=lambda r:(r[0],r[1]))
    return games

def prior(hist,key):
    vals=hist[-WINDOW:]
    if len(vals)<MIN_PRIOR:return None
    return float(np.mean([r[key] for r in vals]))

def prepare(games,qbs):
    history=defaultdict(list)
    last_qb={}
    examples=[]
    rows=[]
    for y,w,home,away,hs,as_ in games:
        for team,other,points,against,venue in ((home,away,hs,as_,1),(away,home,as_,hs,0)):
            current=qbs.get((y,w,team))
            prev_qb=last_qb.get((y,team))
            own=history[(y,team)]
            opp=history[(y,other)]
            # Pre-game contextual features are strictly lagged, no future values.
            own_pts=prior(own,'points'); own_def=prior(own,'against'); own_margin=prior(own,'margin')
            opp_pts=prior(opp,'points'); opp_def=prior(opp,'against'); opp_margin=prior(opp,'margin')
            if current and prev_qb and None not in (own_pts,own_def,own_margin,opp_pts,opp_def,opp_margin):
                changed=int(current!=prev_qb)
                base=[venue,own_pts,own_def,own_margin,opp_pts,opp_def,opp_margin]
                rows.append({'year':y,'week':w,'team':team,'changed':changed,'x':base,'points':points,'margin':points-against})
                if changed and len(examples)<4:
                    examples.append({'season':y,'week':w,'team':team})
            # Important: advance QB identity even for rows with insufficient context.
            if current:
                last_qb[(y,team)]=current
            own.append({'points':points,'against':against,'margin':points-against})
    return rows,examples

def ridge(train,test,target,add_qb):
    def vec(r):return r['x']+([r['changed']] if add_qb else [])
    a=np.asarray([vec(r) for r in train],float)
    b=np.asarray([r[target] for r in train],float)
    z=np.asarray([vec(r) for r in test],float)
    mean=a.mean(axis=0);std=a.std(axis=0);std[std<1e-8]=1
    aa=(a-mean)/std;zz=(z-mean)/std
    design=np.column_stack([np.ones(len(aa)),aa]); test_design=np.column_stack([np.ones(len(zz)),zz])
    penalty=np.eye(design.shape[1])*ALPHA;penalty[0,0]=0
    beta=np.linalg.solve(design.T@design+penalty,design.T@b)
    pred=test_design@beta
    return pred

def diagnostics(rows,target):
    train=[r for r in rows if r['year']<=2023]
    test=[r for r in rows if r['year']>=2024]
    if len(train)<1300 or len(test)<450 or sum(r['changed'] for r in train)<90:
        raise RuntimeError(f'Cobertura insuficiente: train {len(train)}, test {len(test)}')
    truth=np.asarray([r[target] for r in test],float)
    base=ridge(train,test,target,False)
    qb=ridge(train,test,target,True)
    changed=np.asarray([r['changed'] for r in test],bool)
    def score(pred,mask):
        return round(float(np.mean(np.abs(pred[mask]-truth[mask]))),4) if int(np.sum(mask)) else None
    return {'training_period':'2018-2023','later_period':'2024-2025',
            'train_team_games':len(train),'later_team_games':len(test),
            'train_qb_changes':sum(r['changed'] for r in train),
            'later_qb_changes':int(np.sum(changed)),
            'base_mae':score(base,np.ones(len(test),bool)),
            'plus_qb_change_mae':score(qb,np.ones(len(test),bool)),
            'base_mae_changed_only':score(base,changed),
            'plus_qb_change_mae_changed_only':score(qb,changed),
            'mae_improvement_positive_is_better':round(score(base,np.ones(len(test),bool))-score(qb,np.ones(len(test),bool)),4),
            'interpretation':'retrospective oracle variable, NOT a pregame forecast' }

def main():
    qbs=dominant_qbs()
    rows,examples=prepare(games_table(),qbs)
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),
            'source':'nflreadpy weekly QB attempts + nflverse schedule results',
            'qb_identity_proxy':'dominant passer by game-end attempts, 65% threshold',
            'minimum_prior_games':MIN_PRIOR,'rolling_prior_games':WINDOW,
            'matched_team_games':len(rows),'identified_change_events':sum(r['changed'] for r in rows),
            'examples':examples,'margin':diagnostics(rows,'margin'),'points':diagnostics(rows,'points'),
            'uses_only_pregame_context_features':True,
            'qb_change_indicator_known_before_kickoff':False,
            'estimates_causal_qb_value':False,'safe_for_prediction_adjustment':False,
            'model_modified':False,'approved_picks':[],
            'warnings':[
                'El pasador dominante se determina despues de jugar; puede reflejar sustituciones EN el partido.',
                'La variable de cambio en esta prueba se conoce retrospectivamente y NO puede alimentar pronosticos en vivo.',
                'El contexto ajusta localia, promedio reciente de puntos, puntos permitidos y margenes de ambos equipos; faltan lesiones y otras variables.',
                'Los dos registros (local y visitante) de un mismo partido NO son observaciones estadisticamente independientes.',
                '2024-25 ya fue examinado durante el proyecto y NO es un holdout intacto.',
                'Mejorar MAE retrospectivo no demuestra causalidad ni rentabilidad frente al casino.',
                'Sin valor individual por QB, sin ajuste a spread/total, sin picks aprobados.'
            ]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'matched_team_games':result['matched_team_games'],'qb_changes':result['identified_change_events'],'margin':result['margin'],'points':result['points']},indent=2))
    print('NO model update, NO betting permission.')
if __name__=='__main__':main()

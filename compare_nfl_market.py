"""NFL Edge Lab: model versus historical published market reference (diagnostic).

No live prices, point-in-time quotes, vig, EV, ROI or approved picks.
Market spread convention in nflverse games.csv: spread_line is positive for
home favorite, so benchmark predicted home margin equals spread_line.
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

YEARS=(2022,2023,2024,2025)
OUT=Path('data/market_benchmark_diagnostic.json')

def mae(y,p): return round(float(np.mean(np.abs(np.asarray(y)-np.asarray(p)))),3)

def main():
    mapping,_=load_weekly()
    resp=requests.get(SCHEDULE_URL,timeout=60);resp.raise_for_status()
    games=pd.read_csv(StringIO(resp.text),low_memory=False)
    required={'season','week','home_team','away_team','home_score','away_score','spread_line','total_line','game_type'}
    if not required.issubset(games.columns):
        raise ValueError('Missing nflverse columns: '+str(sorted(required-set(games.columns))))
    data=[]
    for g in games.itertuples(index=False):
        if g.game_type!='REG':continue
        try: season,week=int(g.season),int(g.week)
        except (ValueError,TypeError):continue
        h,a,spread,total=map(finite,(g.home_score,g.away_score,g.spread_line,g.total_line))
        if any(v is None for v in (h,a)):continue
        features=game_features(mapping,season,week,str(g.home_team),str(g.away_team))
        if features is None:continue
        data.append({'season':season,'x':features,'margin':h-a,'total':h+a,'market_margin':spread,'market_total':total})
    report=[]
    for year in YEARS:
        train=[g for g in data if g['season']<year]
        test=[g for g in data if g['season']==year]
        if len(train)<500 or len(test)<100:
            raise ValueError(f'Insufficient games for {year}: train {len(train)} test {len(test)}')
        xtr=np.asarray([r['x'] for r in train],float)
        xte=np.asarray([r['x'] for r in test],float)
        item={'season':year,'training_games':len(train),'test_games':len(test)}
        for target,market in [('margin','market_margin'),('total','market_total')]:
            model=predict(ridge_fit(xtr,[r[target] for r in train]),xte)
            eligible=[(r,i) for i,r in enumerate(test) if r[market] is not None]
            if len(eligible)<100: raise ValueError(f'Insufficient market observations for {target} in {year}')
            y=np.array([r[target] for r,_ in eligible]);m=np.array([r[market] for r,_ in eligible]);p=np.array([model[i] for _,i in eligible])
            item[target]={'games_with_market_line':len(eligible),'model_mae':mae(y,p),'market_mae':mae(y,m),'model_minus_market_mae':round(mae(y,p)-mae(y,m),3),'model_bias':round(float(np.mean(p-y)),3),'market_bias':round(float(np.mean(m-y)),3),'mean_absolute_line_disagreement':round(float(np.mean(abs(p-m))),3)}
            # Dissent buckets are descriptive only: no odds/ROI/EV inference.
            bins=[]
            for low,high,label in [(0,1.5,'0-1.5'),(1.5,3,'1.5-3'),(3,5,'3-5'),(5,float('inf'),'5+')]:
                subset=(abs(p-m)>=low)&(abs(p-m)<high)
                if subset.any():
                    bins.append({'line_disagreement_points':label,'games':int(subset.sum()),'model_mae':mae(y[subset],p[subset]),'market_mae':mae(y[subset],m[subset])})
            item[target]['disagreement_buckets']=bins
        report.append(item)
    result={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'reference':'nflverse nfldata games.csv spread_line and total_line','evaluation':'expanding_window_by_season','results':report,'approved_picks':[],'warnings':['Spread/total historical lines are published reference values, NOT independently authenticated point-in-time tradable quotes.','Market lines may include future revisions; do not infer achievable betting ROI or closing-line value.','2022-2025 have been seen during development; diagnostic, not untouched holdout.','Comparison excludes games without at least four prior games of team stats; no injury/QB adjustments.','Model is not calibrated for cover probabilities, EV or betting approvals.']}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    for d in report:
        print(d['season'],'games',d['test_games'],'spread MAE model/market',d['margin']['model_mae'],d['margin']['market_mae'],'total MAE model/market',d['total']['model_mae'],d['total']['market_mae'])
    print('No approved picks. Historical reference lines are not live/executable prices.')
if __name__=='__main__':main()

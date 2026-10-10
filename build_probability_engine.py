"""NFL Edge Lab | Experimental independent probability engine V2.

Out-of-year residual volatility is derived from pre-2025 seasons, and checked on
2025 data already examined elsewhere (NOT an untouched holdout). Uses pregame
team efficiency only; QB/injuries are NOT added or claimed verified.
Probabilities/EV are mathematical estimates, NOT calibrated or approved picks.
Does not change model files, archived snapshots, or existing opportunity engine.
"""
from __future__ import annotations
import json, math
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from build_nfl_model import SCHEDULE_URL, load_weekly, game_features, ridge_fit, predict, finite
from build_opportunity_engine import NAMES, parse_dt, price_prob

ROOT=Path('data')
OUTPUT=ROOT/'probability_engine_v2.json'
CAL_YEARS=tuple(range(2018,2025))
VALIDATION_YEAR=2025
SQRT2=math.sqrt(2)

def normal_cdf(z):
    return 0.5*(1.0+math.erf(z/SQRT2))

def three_way(mu,sigma,line):
    # Discrete integer score margins/totals with continuity correction; this
    # is an APPROXIMATION, not a calibrated distribution of actual scores.
    center=mu+line
    push_possible=abs(line-round(line))<1e-8
    if push_possible:
        win=normal_cdf((center-0.5)/sigma)
        lose=normal_cdf((-0.5-center)/sigma)
        push=max(0.,1.-win-lose)
    else:
        win=normal_cdf(center/sigma)
        lose=1.-win
        push=0.
    return {'win':round(win,5),'lose':round(lose,5),'push':round(push,5)}

def moneyline_prob(mu,sigma,home):
    return round(normal_cdf(((mu if home else -mu)-0.5)/sigma),5)

def odds_profit(american):
    if american is None:return None
    v=float(american)
    if not math.isfinite(v) or v==0:return None
    return v/100. if v>0 else 100./(-v)

def format_quote(market,book,sel,price,prob,point=None):
    profit=odds_profit(price)
    result={'market':market,'book':book,'selection':sel,'line':point,
            'american_odds':price,'model_p_win_experimental':prob['win'],
            'model_p_push_experimental':prob['push'],
            'model_p_lose_experimental':prob['lose'],
            'break_even_probability_raw':price_prob(price),
            'experimental_ev_per_dollar':round(prob['win']*profit-prob['lose'],4) if profit is not None else None,
            'approved':False}
    return result

def historical_rows():
    mapping,_=load_weekly()
    response=requests.get(SCHEDULE_URL,timeout=60)
    response.raise_for_status()
    df=pd.read_csv(StringIO(response.text),low_memory=False)
    mandatory={'season','week','game_type','home_team','away_team','home_score','away_score'}
    if not mandatory.issubset(df.columns):raise ValueError('Missing nflverse schedule fields')
    rows=[]
    for g in df.itertuples(index=False):
        try:season,week=int(g.season),int(g.week)
        except (ValueError,TypeError):continue
        if g.game_type!='REG' or season>VALIDATION_YEAR:continue
        home,away=finite(g.home_score),finite(g.away_score)
        if home is None or away is None:continue
        x=game_features(mapping,season,week,str(g.home_team),str(g.away_team))
        if x is None:continue
        rows.append({'season':season,'x':x,'margin':home-away,'total':home+away})
    return rows

def errors_by_season(rows,season):
    train=[r for r in rows if r['season']<season]
    test=[r for r in rows if r['season']==season]
    if len(train)<500 or len(test)<100:
        raise RuntimeError(f'Insufficient historical data for {season}: {len(train)} train {len(test)} test')
    xtr=[r['x'] for r in train]; xte=[r['x'] for r in test]
    fm=ridge_fit(xtr,[r['margin'] for r in train])
    ft=ridge_fit(xtr,[r['total'] for r in train])
    m=np.asarray(predict(fm,xte),float).reshape(-1)
    t=np.asarray(predict(ft,xte),float).reshape(-1)
    return {'margin':np.asarray([r['margin'] for r in test])-m,
            'total':np.asarray([r['total'] for r in test])-t,
            'pred_margin':m,'truth_margin':np.asarray([r['margin'] for r in test]),
            'games':len(test)}

def main():
    now=datetime.now(timezone.utc)
    model=json.loads((ROOT/'model_diagnostics.json').read_text(encoding='utf-8'))
    odds=json.loads((ROOT/'nfl_odds_latest.json').read_text(encoding='utf-8'))
    quality=json.loads((ROOT/'availability_quality.json').read_text(encoding='utf-8'))
    model_at=parse_dt(model['generated_at_utc']);odds_at=parse_dt(odds['generated_at_utc'])
    if model_at>now or odds_at>now:raise ValueError('Future source timestamp')
    historical=historical_rows()
    cal=[errors_by_season(historical,year) for year in CAL_YEARS]
    residual_m=np.concatenate([x['margin'] for x in cal]);residual_t=np.concatenate([x['total'] for x in cal])
    sigma_m=float(np.std(residual_m,ddof=1));sigma_t=float(np.std(residual_t,ddof=1))
    if not 5<=sigma_m<=35 or not 5<=sigma_t<=35:
        raise RuntimeError('Residual variance implausible. Review inputs.')
    test=errors_by_season(historical,VALIDATION_YEAR)
    pred_home=np.array([normal_cdf((v-0.5)/sigma_m) for v in test['pred_margin']])
    actual_win=np.array(test['truth_margin']>0,dtype=float)
    brier=float(np.mean((pred_home-actual_win)**2))
    calibration_bins=[]
    for lo,hi in [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.00001)]:
        mask=(pred_home>=lo)&(pred_home<hi)
        calibration_bins.append({'range':[lo,min(hi,1)],'games':int(mask.sum()),
                                 'mean_estimated_probability':round(float(pred_home[mask].mean()),4) if mask.any() else None,
                                 'observed_win_rate':round(float(actual_win[mask].mean()),4) if mask.any() else None})
    lookup={(int(p['season']),p['home_team'],p['away_team']):p for p in model['prospective']}
    out=[];missing=[]
    for game in odds.get('games',[]):
        hc,ac=NAMES.get(game['home_team']),NAMES.get(game['away_team'])
        kickoff=parse_dt(game['commence_time'])
        pred=lookup.get((kickoff.year,hc,ac))
        if not pred:
            missing.append({'home':game['home_team'],'away':game['away_team']});continue
        mu_m=float(pred['home_margin_projection']);mu_t=float(pred['game_total_projection'])
        flags=[]
        if kickoff<=now: flags.append('not_bettable_game_started')
        if max(model_at,odds_at)>=kickoff:flags.append('source_not_before_kickoff')
        if (now-odds_at).total_seconds()>86400:flags.append('stale_odds_24h')
        if (now-model_at).total_seconds()>86400:flags.append('stale_model_24h')
        if not quality.get('starter_confirmation_available',False):flags.append('qb_not_verified')
        if not quality.get('ready_for_betting_model',False):flags.append('injury_data_not_verified')
        if int(pred['week'])>int(quality.get('latest_injury_week_in_dataset') or 0): flags.append('future_week_injury_data_missing')
        quotes=[]
        for book in game.get('bookmakers',[]):
            title=book.get('title','Unknown')
            for market in book.get('markets',[]):
                kind=market.get('key')
                for outcome in market.get('outcomes',[]):
                    selection=outcome.get('name');price=outcome.get('price');point=outcome.get('point')
                    if kind=='spreads' and point is not None and selection in (game['home_team'],game['away_team']):
                        home=selection==game['home_team']
                        probs=three_way(mu_m if home else -mu_m,sigma_m,float(point))
                        quotes.append(format_quote('spread',title,selection,price,probs,float(point)))
                    elif kind=='totals' and point is not None and selection in ('Over','Under'):
                        probs=three_way(mu_t-float(point) if selection=='Over' else float(point)-mu_t,sigma_t,0)
                        # For half-points there cannot be a push.
                        if abs(float(point)-round(float(point)))>1e-8:
                            center=(mu_t-float(point))*(1 if selection=='Over' else -1)
                            win=normal_cdf(center/sigma_t)
                            probs={'win':round(win,5),'lose':round(1-win,5),'push':0.0}
                        quotes.append(format_quote('total',title,selection,price,probs,float(point)))
                    elif kind=='h2h' and selection in (game['home_team'],game['away_team']):
                        # NFL ties are possible; ML settlement differs by book. Do not
                        # compute an EV using 2-way probabilities with unresolved tie terms.
                        p=moneyline_prob(mu_m,sigma_m,selection==game['home_team'])
                        quotes.append({'market':'moneyline','book':title,'selection':selection,
                                       'american_odds':price,'model_p_win_experimental':p,
                                       'break_even_probability_raw':price_prob(price),
                                       'experimental_ev_per_dollar':None,
                                       'ev_not_calculated_reason':'tie_settlement_unknown',
                                       'approved':False})
        quotes.sort(key=lambda q:(q['market'],q['book'],q['selection']))
        out.append({'game_id':pred['game_id'],'week':pred['week'],'kickoff_utc':kickoff.isoformat(),
                    'home':game['home_team'],'away':game['away_team'],
                    'independent_home_margin':mu_m,'independent_total':mu_t,
                    'model_home_win_probability_experimental':moneyline_prob(mu_m,sigma_m,True),
                    'flags':flags,'status':'historical_only' if kickoff<=now else 'experimental_review',
                    'quotes':quotes,'approved_picks':[]})
    out.sort(key=lambda g:(g['week'],g['kickoff_utc'],g['game_id']))
    report={'generated_at_utc':now.isoformat(),'odds_generated_at_utc':odds_at.isoformat(),
            'model_generated_at_utc':model_at.isoformat(),
            'method':'rolling_season_residual_normal_approximation',
            'residual_training_seasons':list(CAL_YEARS),
            'independent_margin_residual_sigma':round(sigma_m,4),
            'independent_total_residual_sigma':round(sigma_t,4),
            'residual_sample_size':len(residual_m),
            'retrospective_2025_test':{'games':test['games'],'home_win_brier':round(brier,5),'bins':calibration_bins,
                                       'not_an_untouched_holdout':True},
            'games_count':len(out),'games_without_model':missing,'games':out,
            'probabilities_calibrated_prospectively':False,'ev_validated':False,'approved_picks':[],
            'warnings':['Probabilities computed by NORMAL APPROXIMATION, not validated calibration.',
                        'Experimental EV on spread/total is NOT demonstrated betting edge.',
                        'No verified QB or player-injury adjustments are applied.',
                        '2025 has already been studied; not an independent untouched holdout.',
                        'Moneyline EV deliberately omitted because tie settlement is not modeled.',
                        'A large projection/market difference cannot itself approve a bet.',
                        'Quotes are timestamped snapshots and can be stale or no longer available.',
                        'Upcoming weeks may use incomplete current-season information.']}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Games:',len(out),'residual samples:',len(residual_m),'2025 Brier:',round(brier,5))
    print('Forecasts unchanged. Approved picks: 0. Output:',OUTPUT)

if __name__=='__main__':main()

"""NFL Edge Lab: motor independiente de candidatos, etapa 1.

Lee el modelo deportivo y cuotas existentes. No reentrena, no altera previsiones,
no inventa probabilidades, EV o picks aprobados. Analiza TODOS los mercados.
"""
from __future__ import annotations
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('data')
MODEL = ROOT / 'model_diagnostics.json'
ODDS = ROOT / 'nfl_odds_latest.json'
QUALITY = ROOT / 'availability_quality.json'
OUTPUT = ROOT / 'opportunity_engine.json'
NAMES = {
 'Arizona Cardinals':'ARI','Atlanta Falcons':'ATL','Baltimore Ravens':'BAL',
 'Buffalo Bills':'BUF','Carolina Panthers':'CAR','Chicago Bears':'CHI',
 'Cincinnati Bengals':'CIN','Cleveland Browns':'CLE','Dallas Cowboys':'DAL',
 'Denver Broncos':'DEN','Detroit Lions':'DET','Green Bay Packers':'GB',
 'Houston Texans':'HOU','Indianapolis Colts':'IND','Jacksonville Jaguars':'JAX',
 'Kansas City Chiefs':'KC','Las Vegas Raiders':'LV','Los Angeles Chargers':'LAC',
 'Los Angeles Rams':'LA','Miami Dolphins':'MIA','Minnesota Vikings':'MIN',
 'New England Patriots':'NE','New Orleans Saints':'NO','New York Giants':'NYG',
 'New York Jets':'NYJ','Philadelphia Eagles':'PHI','Pittsburgh Steelers':'PIT',
 'San Francisco 49ers':'SF','Seattle Seahawks':'SEA','Tampa Bay Buccaneers':'TB',
 'Tennessee Titans':'TEN','Washington Commanders':'WAS'
}

def parse_dt(x):
    dt = datetime.fromisoformat(str(x).replace('Z','+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timezone missing')
    return dt.astimezone(timezone.utc)

def price_prob(price):
    if price is None or float(price)==0: return None
    p=float(price)
    return round(100 / (p+100) if p>0 else -p/(100-p), 5)

def markets(book):
    return {m.get('key'):m for m in book.get('markets',[])}

def find_outcome(m, name):
    return next((o for o in m.get('outcomes',[]) if o.get('name')==name),None)

def r2(n):
    return round(float(n),2)

def main():
    model=json.loads(MODEL.read_text(encoding='utf-8'))
    odds=json.loads(ODDS.read_text(encoding='utf-8'))
    quality=json.loads(QUALITY.read_text(encoding='utf-8')) if QUALITY.exists() else {}
    now=datetime.now(timezone.utc)
    model_at=parse_dt(model['generated_at_utc'])
    odds_at=parse_dt(odds['generated_at_utc'])
    if model_at>now or odds_at>now: raise ValueError('Fuente fechada en el futuro')
    by_pair={(p['season'],p['home_team'],p['away_team']):p for p in model['prospective']}
    results=[]; missing=[]
    for event in odds.get('games',[]):
        home,away=event['home_team'],event['away_team']
        hc,ac=NAMES.get(home),NAMES.get(away)
        kickoff=parse_dt(event['commence_time'])
        candidates=[p for (s,h,a),p in by_pair.items() if h==hc and a==ac and s==kickoff.year]
        if len(candidates)!=1:
            missing.append({'match':f'{away} @ {home}','reason':'No unique season/team projection'})
            continue
        pred=candidates[0]
        week=int(pred['week'])
        flags=[]
        if kickoff<=now: flags.append('game_started_or_finished')
        if model_at>=kickoff or odds_at>=kickoff: flags.append('source_not_pregame')
        if (now-odds_at).total_seconds()>24*3600: flags.append('odds_older_than_24h')
        if (now-model_at).total_seconds()>24*3600: flags.append('model_older_than_24h')
        if not quality.get('starter_confirmation_available',False): flags.append('qb_starter_not_verified_for_all_games')
        if not quality.get('ready_for_betting_model',False): flags.append('injury_quality_not_approved')
        if week>int(quality.get('latest_injury_week_in_dataset') or 0): flags.append('future_week_injury_context_not_ready')
        # An immutable forecast may be compared as a historic snapshot, not as an actionable quote.
        margin=float(pred['home_margin_projection'])
        total=float(pred['game_total_projection'])
        lines=[]
        for book in event.get('bookmakers',[]):
            mm=markets(book)
            spread=mm.get('spreads',{})
            totals=mm.get('totals',{})
            ml=mm.get('h2h',{})
            for team,is_home in ((home,True),(away,False)):
                line=find_outcome(spread,team)
                if line and line.get('point') is not None:
                    point=float(line['point'])
                    projected_cover_margin=(margin if is_home else -margin)+point
                    lines.append({'market':'spread','book':book.get('title'),'selection':team,
                        'point':point,'american_odds':line.get('price'),
                        'implied_break_even_probability':price_prob(line.get('price')),
                        'projected_cover_margin':r2(projected_cover_margin),
                        'raw_discrepancy_pts':r2(abs(projected_cover_margin))})
                line=find_outcome(ml,team)
                if line:
                    lines.append({'market':'moneyline','book':book.get('title'),'selection':team,
                        'american_odds':line.get('price'),
                        'implied_break_even_probability':price_prob(line.get('price')),
                        'model_projected_winner':home if margin>0 else away if margin<0 else 'tie',
                        'model_winner_margin_pts':r2(abs(margin)),
                        'model_win_probability':None})
            for side in ('Over','Under'):
                line=find_outcome(totals,side)
                if line and line.get('point') is not None:
                    point=float(line['point']); edge=(total-point) * (1 if side=='Over' else -1)
                    lines.append({'market':'total','book':book.get('title'),'selection':side,
                        'point':point,'american_odds':line.get('price'),
                        'implied_break_even_probability':price_prob(line.get('price')),
                        'projected_side_margin':r2(edge),'raw_discrepancy_pts':r2(abs(edge))})
        # Candidate is a discrepancy for REVIEW, not a positive-EV bet.
        interesting=[x for x in lines if (x['market']=='spread' and x.get('projected_cover_margin',0)>=3) or
                     (x['market']=='total' and x.get('projected_side_margin',0)>=3)]
        interesting.sort(key=lambda x:(-x['raw_discrepancy_pts'],x['market'],x['book'] or ''))
        results.append({'game_id':pred['game_id'],'week':week,'kickoff_utc':kickoff.isoformat(),
            'away_team':away,'home_team':home,'model_home_margin':r2(margin),
            'model_total':r2(total),'flags':flags,
            'status':'historical_only' if kickoff<=now else 'requires_review',
            'review_candidates':interesting,
            'all_market_quotes':lines,
            'approved_picks':[]})
    results.sort(key=lambda x:(x['week'],x['kickoff_utc'],x['game_id']))
    report={'generated_at_utc':now.isoformat(),'model_generated_at_utc':model_at.isoformat(),
        'odds_generated_at_utc':odds_at.isoformat(),
        'games_with_comparisons':len(results),'games_without_matching_model':missing,
        'review_candidate_quotes':sum(len(x['review_candidates']) for x in results),
        'games_with_review_candidates':sum(bool(x['review_candidates']) for x in results),
        'games':results,'approved_picks':[],
        'probability_model_calibrated':False,'expected_value_calculated':False,
        'notes':['La predicción deportiva precede a la comparación de mercado.',
          'Las discrepancias no se interpretan como EV, confianza o ventaja comprobada.',
          'La probabilidad implícita es la tasa de equilibrio SIN ajustar el vigorish.',
          'Se listan cotizaciones de múltiples casas; no son apuestas independientes.',
          'Partidos iniciados o fuentes obsoletas quedan como historial, nunca disponibles para apostar.',
          'No modifica proyecciones, pronósticos archivados ni la web de producción.']}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f"Juegos: {len(results)}, juegos con señales preliminares: {report['games_with_review_candidates']}")
    print(f"Cotizaciones con discrepancia >=3: {report['review_candidate_quotes']}; picks aprobados: 0")
    print(f"Salida: {OUTPUT}")

if __name__=='__main__': main()

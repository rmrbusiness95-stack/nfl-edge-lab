"""NFL Edge Lab V3: decision review without pretending experimental EV is validated.

Consumes V2 results, availability quality, and previously archived odds snapshots.
Aggregates distinct opportunities and quotes per game instead of counting bookmakers
as separate opportunities. Monitors only genuine timestamped observations; does NOT
claim a trend where insufficient snapshots exist. Never approves bets.
Run from repository root: python build_decision_center_v3.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('data')
SOURCE = ROOT / 'probability_engine_v2.json'
QUALITY = ROOT / 'availability_quality.json'
HISTORY = ROOT / 'odds_history'
OUT = ROOT / 'decision_center_v3.json'


def utc(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp sin zona horaria')
    return dt.astimezone(timezone.utc)


def finite(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def price_candidates(q):
    """Return bookmaker quotes in V2 as independently quoted markets."""
    for entry in q:
        if entry.get('market') not in ('spread', 'total', 'moneyline'):
            continue
        p = finite(entry.get('model_p_win_experimental'))
        ev = finite(entry.get('experimental_ev_per_dollar'))
        if p is None or not (0 <= p <= 1):
            continue
        yield {'market': entry['market'], 'selection':entry['selection'],
               'book':entry.get('book'), 'line':entry.get('line'),
               'american_odds':entry.get('american_odds'),
               'probability_experimental':round(p,5),
               'ev_experimental_per_dollar':round(ev,4) if ev is not None else None,
               'push_probability_experimental':entry.get('model_p_push_experimental')}


def best_quotes(entries):
    """Best recorded EV per selection, never mix book/line/momio combinations."""
    grouped=defaultdict(list)
    for q in price_candidates(entries):
        grouped[(q['market'],q['selection'])].append(q)
    chosen=[]
    for (market,selection),choices in grouped.items():
        priced=[q for q in choices if q['ev_experimental_per_dollar'] is not None]
        if priced:
            # Experimental metric is only for *comparison* of simultaneously quoted prices.
            best=max(priced,key=lambda q:q['ev_experimental_per_dollar'])
        else:
            best=max(choices,key=lambda q:q['probability_experimental'])
        chosen.append({**best,'books_compared':len(choices)})
    chosen.sort(key=lambda q:(q['market'],q['selection']))
    return chosen


def archived_quotes():
    """Use archive's odds timestamp (not file creation date) and preserve book identity."""
    observations=defaultdict(list)
    files=sorted(HISTORY.glob('**/*.json')) if HISTORY.exists() else []
    for filename in files:
        try:
            archive=json.loads(filename.read_text(encoding='utf-8'))
            observed=utc(archive['odds_collected_utc'])
            for event in archive.get('games',[]):
                start=utc(event['commence_time'])
                if observed>=start: continue
                for book in event.get('bookmakers',[]):
                    key=book.get('key') or book.get('title')
                    for market in book.get('markets',[]):
                        if market.get('key') not in ('spreads','totals','h2h'):continue
                        for item in market.get('outcomes',[]):
                            obskey=(event['home_team'],event['away_team'],market['key'],item.get('name'),key)
                            observations[obskey].append({'at_utc':observed.isoformat(),
                                                           'point':item.get('point'),
                                                           'american_odds':item.get('price')})
        except (KeyError,TypeError,ValueError,json.JSONDecodeError):
            continue
    for key,values in observations.items():
        # Repeated archive of identical source must not imply new market data.
        unique={v['at_utc']:v for v in values}
        observations[key]=sorted(unique.values(),key=lambda v:v['at_utc'])
    return observations,len(files)


def main():
    now=datetime.now(timezone.utc)
    v2=json.loads(SOURCE.read_text(encoding='utf-8'))
    quality=json.loads(QUALITY.read_text(encoding='utf-8')) if QUALITY.exists() else {}
    if v2.get('probabilities_calibrated_prospectively') is not False:
        raise RuntimeError('No se reconoce la declaracion de calibracion V2')
    quotes_at=utc(v2['odds_generated_at_utc'])
    model_at=utc(v2['model_generated_at_utc'])
    history,histfiles=archived_quotes()
    outputs=[]
    for game in v2.get('games',[]):
        start=utc(game['kickoff_utc'])
        issues=list(game.get('flags',[]))
        if now>=start and 'game_already_started' not in issues:issues.append('game_already_started')
        if max(model_at,quotes_at)>=start and 'sources_not_before_kickoff' not in issues:
            issues.append('sources_not_before_kickoff')
        if (now-quotes_at).total_seconds()>86400 and 'odds_older_than_24h' not in issues:
            issues.append('odds_older_than_24h')
        if (now-model_at).total_seconds()>86400 and 'model_older_than_24h' not in issues:
            issues.append('model_older_than_24h')
        # Explicit no-approval gate: data integrity + predictive calibration missing.
        if not quality.get('ready_for_betting_model',False):
            if 'availability_quality_not_verified' not in issues:issues.append('availability_quality_not_verified')
        options=best_quotes(game.get('quotes',[]))
        for option in options:
            market_key={'spread':'spreads','total':'totals','moneyline':'h2h'}[option['market']]
            source_key=(game['home'],game['away'],market_key,option['selection'])
            # Histories per bookmaker use keys while V2 stores title. Match by
            # full title if provider key differs; prefer V2 book title's token.
            known=[]
            for k,observations in history.items():
                if k[:4]!=source_key:continue
                if k[4] and (str(k[4]).casefold()==str(option['book']).casefold() or
                            str(k[4]).replace('.','').casefold()==str(option['book']).replace('.','').casefold()):
                    known.extend(observations)
            # Fallback: report earliest/latest seen across books, never as a
            # matched movement; lines across different books aren't comparable.
            known=sorted({o['at_utc']:o for o in known}.values(), key=lambda o:o['at_utc'])
            if len(known)>=2:
                option['movement']={'status':'matched_book_observations',
                                    'first':known[0], 'last':known[-1],
                                    'observations':len(known)}
            elif len(known)==1:
                option['movement']={'status':'single_archived_observation','first':known[0]}
            else:
                option['movement']={'status':'insufficient_matched_history'}
            # A price choice across different books is still a quote, not verified
            # accessible to this user or at this precise moment.
            option['recommendation']='NO_APROBADA'
        outputs.append({'game_id':game['game_id'],'week':game['week'],
                        'kickoff_utc':game['kickoff_utc'],
                        'away_team':game['away'],'home_team':game['home'],
                        'independent_home_margin':game['independent_home_margin'],
                        'independent_total':game['independent_total'],
                        'model_home_win_probability_experimental':game['model_home_win_probability_experimental'],
                        'status':'HISTORICO' if start<=now else 'SEGUIMIENTO_SIN_APROBACION',
                        'blocking_issues':sorted(set(issues+['probabilities_not_prospectively_validated'])),
                        'best_quotes_by_selection':options,
                        'approved_picks':[]})
    outputs.sort(key=lambda o:(o['week'],o['kickoff_utc'],o['game_id']))
    report={'generated_at_utc':now.isoformat(),
            'source_probabilities_utc':v2['generated_at_utc'],
            'source_odds_utc':v2['odds_generated_at_utc'],
            'source_model_utc':v2['model_generated_at_utc'],
            'archived_odds_files_inspected':histfiles,
            'games_count':len(outputs),
            'distinct_game_market_selections':sum(len(o['best_quotes_by_selection']) for o in outputs),
            'games':outputs,'approved_picks':[],
            'alerts_operational':False,'probabilities_prospectively_calibrated':False,
            'notes':['La opinion deportiva es independiente; cuotas solo comparan el precio.',
                     'Se consolida cada seleccion sin tratar 9 casas como 9 apuestas independientes.',
                     'La mejor cotizacion proviene de una sola casa y contiene juntos linea y momio.',
                     'Una primera y ultima captura no son necesariamente apertura y cierre.',
                     'Los momios archivados pueden haber caducado o ser inaccesibles.',
                     'EV V2 es experimental, no ventaja demostrada ni consejo de entrar.',
                     'No se autorizan apuestas sin validacion prospectiva y controles de disponibilidad.',
                     'Este reporte no modifica el modelo ni los snapshots anteriores.']}
    OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Juegos:',len(outputs),'selecciones distintas:',report['distinct_game_market_selections'])
    print('Archivos historicos de cuotas:',histfiles,'picks aprobados: 0')

if __name__=='__main__':main()

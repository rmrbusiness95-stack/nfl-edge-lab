"""NFL Edge Lab: descriptive QB-change team results, 2018-2025.

Changes use dominant passer, NOT verified starter or injury. Scores from nflverse.
This does not estimate player-specific value, causal effects, betting EV or picks.
"""
from __future__ import annotations
import csv
import io
import json
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path

import nflreadpy as nfl
import requests

YEARS = range(2018, 2026)
GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
OUT = Path('data/qb_change_study.json')

def number(x):
    try: return float(x)
    except (TypeError, ValueError): return None

def round_or_null(x):
    return round(x, 3) if x is not None else None

def group_summary(rows):
    if not rows: return {'games': 0, 'mean_points': None, 'mean_margin': None, 'mean_point_change_from_previous': None, 'mean_margin_change_from_previous': None}
    return {'games': len(rows), 'mean_points': round_or_null(sum(r['points'] for r in rows)/len(rows)),
            'mean_margin': round_or_null(sum(r['margin'] for r in rows)/len(rows)),
            'mean_point_change_from_previous': round_or_null(sum(r['points_delta'] for r in rows)/len(rows)),
            'mean_margin_change_from_previous': round_or_null(sum(r['margin_delta'] for r in rows)/len(rows))}

def main():
    # Load weekly passing boxscores for QB identity and attempts.
    by_week = defaultdict(list)
    for year in YEARS:
        df = nfl.load_player_stats([year], summary_level='week')
        required = {'season','week','team','player_id','position','attempts'}
        missing = required - set(df.columns)
        if missing: raise RuntimeError(f'Missing weekly passing fields in {year}: {sorted(missing)}')
        for p in df.select(sorted(required)).to_dicts():
            if p.get('position') != 'QB' or not p.get('team') or not p.get('player_id'): continue
            attempts = number(p.get('attempts'))
            if attempts is None or attempts < 0: continue
            week = int(p['week'])
            if week < 1 or week > 18: continue
            by_week[(year,week,p['team'])].append((str(p['player_id']),attempts))
    top = {}
    for key, players in by_week.items():
        attempts = sum(x[1] for x in players)
        best = max(players,key=lambda x:x[1])
        if attempts>=10 and best[1]>=10 and best[1]/attempts>=0.65:
            top[key]=best[0]
    r=requests.get(GAMES_URL,timeout=60)
    r.raise_for_status()
    games=csv.DictReader(io.StringIO(r.text))
    required={'season','week','game_type','home_team','away_team','home_score','away_score'}
    if not required.issubset(games.fieldnames or []): raise RuntimeError('Missing games.csv columns')
    scores={}
    for g in games:
        try: season,week=int(g['season']),int(g['week'])
        except (TypeError,ValueError): continue
        if season not in YEARS or g['game_type']!='REG':continue
        home_score,away_score=number(g['home_score']),number(g['away_score'])
        if home_score is None or away_score is None:continue
        for team,points,allowed,venue in [(g['home_team'],home_score,away_score,'home'),(g['away_team'],away_score,home_score,'away')]:
            key=(season,week,team)
            if key in scores:raise RuntimeError(f'Duplicate team-game key {key}')
            scores[key]={'points':points,'margin':points-allowed,'venue':venue}
    rows=[]
    for (year,week,team),curr in sorted(top.items()):
        prev=top.get((year,week-1,team))
        now_score=scores.get((year,week,team))
        prev_score=scores.get((year,week-1,team))
        if not prev or not now_score or not prev_score:continue
        # Observational grouping; previous week score can be distorted by injury midgame.
        rows.append({'season':year,'week':week,'team':team,'changed':curr!=prev,
                     'points':now_score['points'],'margin':now_score['margin'],
                     'points_delta':now_score['points']-prev_score['points'],
                     'margin_delta':now_score['margin']-prev_score['margin'],
                     'venue':now_score['venue']})
    if len(rows)<2000:raise RuntimeError(f'Not enough matched team-weeks: {len(rows)}')
    results={}
    for label,years in [('development',set(range(2018,2024))),('later_check',{2024,2025})]:
        subset=[r for r in rows if r['season'] in years]
        change=[r for r in subset if r['changed']]
        stable=[r for r in subset if not r['changed']]
        results[label]={'seasons':sorted(years),'qb_change':group_summary(change),
                        'no_qb_change':group_summary(stable),
                        'difference_of_mean_points_delta':round_or_null(group_summary(change)['mean_point_change_from_previous']-group_summary(stable)['mean_point_change_from_previous']) if change and stable else None,
                        'difference_of_mean_margin_delta':round_or_null(group_summary(change)['mean_margin_change_from_previous']-group_summary(stable)['mean_margin_change_from_previous']) if change and stable else None}
    out={'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'source':'nflreadpy weekly player stats + nflverse games.csv',
         'matched_team_weeks':len(rows),'observational_results':results,
         'qb_starter_verified':False,'injury_cause_verified':False,'causal_effect_estimated':False,'model_modified':False,'approved_picks':[],
         'warnings':['QB dominante por intentos NO es titular confirmado ni prueba lesión.',
                     'Comparar puntos entre semanas NO controla calidad de rivales, localía, cambios de juego ni selección de suplentes.',
                     '2018-2023 es exploración y 2024-2025 es un contraste descriptivo, NO un holdout causal.',
                     'La variación encontrada NO equivale a puntos que deban restarse de un spread.',
                     'Nunca aplicar automáticamente ajustes a proyecciones ni aprobar apuestas con esta prueba.']}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'team_weeks':len(rows),'results':results},ensure_ascii=False,indent=2))
    print('Descriptive study only: zero QB adjustments or approved picks.')

if __name__=='__main__':main()

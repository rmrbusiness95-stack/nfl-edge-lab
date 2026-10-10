"""NFL Edge Lab: evaluar señales prospectivas congeladas sin reescribir su historia.

Entrada: data/signal_snapshots/**/*.json y calendario nflverse games.csv.
Salida: data/signal_results.json. No crea/aprueba apuestas.
Cada captura se conserva; la vista 'first_signal_per_selection' cuenta una sola
observación inicial por (game_id, market, selection), sin duplicar semanas/casas.
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = Path('data/signal_snapshots')
OUTPUT = Path('data/signal_results.json')
GAMES_URL = 'https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
WAIT_HOURS = 24


def utc(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Datetime sin zona horaria')
    return dt.astimezone(timezone.utc)


def score(value):
    try:
        f = float(value)
        return int(f) if pd.notna(f) and f.is_integer() and f >= 0 else None
    except (TypeError, ValueError):
        return None


def grade(signal, home_name, away_name, hs, aws):
    market = signal['market']
    choice = signal['selection']
    if market in ('moneyline', 'spread'):
        if choice not in (home_name, away_name):
            return 'INVALID_SELECTION'
        delta = hs - aws if choice == home_name else aws - hs
        if market == 'moneyline':
            # Una igualdad en NFL no puede liquidarse universalmente sin las
            # reglas de la casa. No adjudicar apuesta ganada/perdida.
            if delta == 0:
                return 'TIE_REQUIRES_BOOK_RULES'
        else:
            point = signal.get('line')
            if point is None:
                return 'INVALID_LINE'
            delta += float(point)
    elif market == 'total':
        point = signal.get('line')
        if point is None:
            return 'INVALID_LINE'
        if choice == 'Over':
            delta = hs + aws - float(point)
        elif choice == 'Under':
            delta = float(point) - (hs + aws)
        else:
            return 'INVALID_SELECTION'
    else:
        return 'INVALID_MARKET'
    return 'WIN' if delta > 0 else 'LOSS' if delta < 0 else 'PUSH'


def summary(records):
    counted = Counter(r['result'] for r in records)
    wins, losses, pushes = (counted[k] for k in ('WIN', 'LOSS', 'PUSH'))
    denomin = wins + losses
    return {'signals': len(records), 'wins': wins, 'losses': losses,
            'pushes': pushes, 'pending': counted['PENDING'],
            'unresolved_or_invalid': len(records) - wins - losses - pushes - counted['PENDING'],
            'win_rate_excluding_pushes': round(wins / denomin, 4) if denomin else None}


def main():
    files = sorted(ARCHIVE.glob('**/*.json'))
    if not files:
        raise RuntimeError('Todavia no existen capturas de senales. No generar evaluacion vacia.')
    response = requests.get(GAMES_URL, timeout=60)
    response.raise_for_status()
    fixture = pd.read_csv(StringIO(response.text), low_memory=False)
    expected = {'game_id', 'game_type', 'home_score', 'away_score', 'home_team', 'away_team'}
    if not expected.issubset(fixture.columns):
        raise RuntimeError('Calendario nflverse sin campos: ' + str(sorted(expected-set(fixture.columns))))
    finals = {}
    for g in fixture.itertuples(index=False):
        if g.game_type != 'REG':
            continue
        hs, aws = score(g.home_score), score(g.away_score)
        if hs is not None and aws is not None:
            finals[str(g.game_id)] = (hs, aws, str(g.home_team), str(g.away_team))
    # El calendario usa abreviaturas. Las capturas usan nombres completos.
    # La identidad del juego viene de game_id; sus equipos se toman de la captura.
    now = datetime.now(timezone.utc)
    rows = []
    excluded = Counter()
    for filepath in files:
        capture = json.loads(filepath.read_text(encoding='utf-8'))
        cap_at = utc(capture['captured_at_utc'])
        stamps = [cap_at, utc(capture['v3_generated_at_utc']),
                  utc(capture['model_generated_at_utc']),
                  utc(capture['odds_generated_at_utc']),
                  utc(capture['v2_generated_at_utc'])]
        for g in capture.get('games', []):
            kickoff = utc(g['kickoff_utc'])
            if any(stamp >= kickoff for stamp in stamps):
                excluded['invalid_timing_games'] += 1
                continue
            final = finals.get(g['game_id']) if now >= kickoff + timedelta(hours=WAIT_HOURS) else None
            for s in g.get('signals', []):
                result = grade(s, g['home_team'], g['away_team'], final[0], final[1]) if final else 'PENDING'
                rows.append({
                    'snapshot_file':str(filepath), 'captured_at_utc':cap_at.isoformat(),
                    'game_id':g['game_id'], 'week':g['week'], 'kickoff_utc':kickoff.isoformat(),
                    'home_team':g['home_team'], 'away_team':g['away_team'],
                    'market':s['market'], 'selection':s['selection'], 'line':s.get('line'),
                    'american_odds':s.get('american_odds'), 'book':s.get('book'),
                    'classification':s.get('classification'),
                    'model_probability_experimental':s.get('estimated_win_probability_experimental'),
                    'model_ev_experimental':s.get('estimated_ev_per_dollar_experimental'),
                    'blocking_issues_at_capture':g.get('blocking_issues_at_capture', []),
                    'result':result,
                    'final_home_score':final[0] if final else None,
                    'final_away_score':final[1] if final else None,
                    'recorded_as_bet':False, 'approved_pick':False,
                })
    if not rows:
        raise RuntimeError('No se encontraron señales con captura pregame valida')
    rows.sort(key=lambda r:(r['captured_at_utc'],r['game_id'],r['market'],r['selection']))
    first = {}
    for r in rows:
        key = (r['game_id'],r['market'],r['selection'])
        if key not in first:
            first[key] = r
    earliest = list(first.values())
    per_market = {m:summary([r for r in earliest if r['market']==m]) for m in ('moneyline','spread','total')}
    report = {
        'generated_at_utc':now.isoformat(),'score_source':GAMES_URL,
        'snapshots_processed':len(files),'snapshot_signal_records':len(rows),
        'distinct_first_selections':len(earliest),
        'all_snapshots_summary':summary(rows),
        'first_signal_per_selection_summary':summary(earliest),
        'first_signal_by_market':per_market,
        'first_signal_per_selection':earliest,
        'all_snapshots':rows,
        'excluded':dict(excluded),
        'approved_picks':[],
        'notes':[
            'Las señales son experimentales y no son apuestas realizadas ni picks aprobados.',
            'Las capturas repetidas no son apuestas independientes: primera selección por juego/mercado/lado para métricas principales.',
            'La probabilidad usada era experimental; WIN/LOSS no demuestra ventaja predictiva.',
            'El registro inicial de moneyline puede incluir ambos lados del mismo partido: son observaciones, no picks recomendados.',
            'Los marcadores se evalúan tras 24 horas del kickoff cuando el proveedor ya los registra.',
            'Empates NFL en moneyline requieren reglas especificas de la casa; no adjudicados.',
            'Para CLV se requiere cotejar momios equivalentes y observaciones previas al kickoff; este archivo no lo infiere.',
        ]}
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    OUTPUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Capturas:',len(files),'registros:',len(rows),'primeras selecciones:',len(earliest))
    print('Primeras selecciones:',report['first_signal_per_selection_summary'])
    print('Salida:',OUTPUT)


if __name__ == '__main__':
    main()

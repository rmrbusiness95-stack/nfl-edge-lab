"""NFL Edge Lab — registro prospectivo e inmutable de señales experimentales.

Entrada: data/decision_center_v3.json (requiere salida V3).
Salida: data/signal_snapshots/YYYY-MM-DD/<sha256>.json.
Solo archiva datos efectivamente disponibles ANTES del kickoff, sin modificar
proyecciones y sin declarar apuestas aprobadas ni rentabilidad demostrada.

Se conserva como máximo una cotización por (partido, mercado, selección) de V3.
Una señal se registra solo si el EV experimental es positivo; NO es una
recomendación y no es prueba de valor real. Las moneylines se preservan como
observaciones de probabilidad, con EV desconocido.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('data/decision_center_v3.json')
ROOT = Path('data/signal_snapshots')


def utc(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('La fecha no contiene zona horaria')
    return dt.astimezone(timezone.utc)


def numeric(value):
    try:
        num = float(value)
        return num if math.isfinite(num) else None
    except (ValueError, TypeError):
        return None


def main():
    raw = SOURCE.read_bytes()
    doc = json.loads(raw)
    now = datetime.now(timezone.utc)
    v3_at = utc(doc['generated_at_utc'])
    model_at = utc(doc['source_model_utc'])
    odds_at = utc(doc['source_odds_utc'])
    v2_at = utc(doc['source_probabilities_utc'])
    timestamps = (v3_at, model_at, odds_at, v2_at)
    if any(t > now for t in timestamps):
        raise ValueError('Una fuente indica una fecha futura')
    if doc.get('alerts_operational') is not False or doc.get('probabilities_prospectively_calibrated') is not False:
        raise ValueError('Esquema V3 inesperado: detener para revision')
    if not isinstance(doc.get('games'), list) or not doc['games']:
        raise ValueError('No hay encuentros en V3')

    games = []
    excluded = {'already_started': 0, 'source_after_kickoff': 0, 'no_valid_signals': 0}
    for game in doc['games']:
        kickoff = utc(game['kickoff_utc'])
        if now >= kickoff:
            excluded['already_started'] += 1
            continue
        if any(t >= kickoff for t in timestamps):
            excluded['source_after_kickoff'] += 1
            continue
        selected = []
        seen = set()
        for quote in game.get('best_quotes_by_selection', []):
            market = quote.get('market')
            selection = quote.get('selection')
            if market not in ('spread', 'total', 'moneyline') or not selection:
                continue
            key = (market, selection)
            if key in seen:
                raise ValueError(f'Duplicado en V3: {game["game_id"]} / {key}')
            seen.add(key)
            prob = numeric(quote.get('probability_experimental'))
            american = numeric(quote.get('american_odds'))
            ev = numeric(quote.get('ev_experimental_per_dollar'))
            if prob is None or not 0 <= prob <= 1 or american is None or american == 0:
                continue
            # Register V3 market-interest signals; do not manufacture ML EV.
            if market == 'moneyline':
                if ev is not None:
                    raise ValueError('EV moneyline no respaldado por V2')
                classification = 'moneyline_observation'
            elif ev is not None and ev > 0:
                classification = 'experimental_positive_ev_candidate'
            else:
                continue
            selected.append({
                'market': market,
                'selection': selection,
                'line': quote.get('line'),
                'book': quote.get('book'),
                'american_odds': quote.get('american_odds'),
                'estimated_win_probability_experimental': prob,
                'estimated_push_probability_experimental': quote.get('push_probability_experimental'),
                'estimated_ev_per_dollar_experimental': ev,
                'classification': classification,
                'movement': quote.get('movement'),
                'status_at_capture': 'EXPERIMENTAL_NOT_APPROVED',
            })
        if not selected:
            excluded['no_valid_signals'] += 1
            continue
        games.append({
            'game_id': game['game_id'],
            'week': game['week'],
            'kickoff_utc': game['kickoff_utc'],
            'home_team': game['home_team'],
            'away_team': game['away_team'],
            'independent_home_margin': game['independent_home_margin'],
            'independent_total': game['independent_total'],
            'home_win_probability_experimental': game['model_home_win_probability_experimental'],
            'blocking_issues_at_capture': game.get('blocking_issues', []),
            'signals': selected,
        })
    if not games:
        raise RuntimeError('Sin señales anteriores al kickoff: no generar registro vacío')
    sha = hashlib.sha256(raw).hexdigest()
    output = ROOT / v3_at.date().isoformat() / f'{sha[:20]}.json'
    if output.exists():
        old = json.loads(output.read_text(encoding='utf-8'))
        if old.get('v3_source_sha256') != sha:
            raise RuntimeError('Conflicto de huella digital de fuente')
        print('Fuente archivada previamente; no se sobrescribe:', output)
        return
    report = {
        'captured_at_utc': now.isoformat(),
        'v3_generated_at_utc': v3_at.isoformat(),
        'model_generated_at_utc': model_at.isoformat(),
        'odds_generated_at_utc': odds_at.isoformat(),
        'v2_generated_at_utc': v2_at.isoformat(),
        'v3_source_sha256': sha,
        'games_recorded': len(games),
        'signals_recorded': sum(len(g['signals']) for g in games),
        'excluded': excluded,
        'games': games,
        'approved_picks': [],
        'real_bets_placed': [],
        'notes': [
            'Registro de señales observadas antes del kickoff; no son apuestas realizadas.',
            'Las selecciones con EV positivo usan probabilidades experimentales sin calibracion prospectiva.',
            'Las observaciones de moneyline NO tienen EV validado ni necesariamente sugieren entrar.',
            'La misma selección puede aparecer en capturas distintas: no equivale a apuestas independientes.',
            'Las observaciones se congelan y jamás se sobrescriben; el resultado final se evalúa aparte.',
            'No se aprueban apuestas automáticamente.',
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Registro creado: {output}')
    print('Partidos:', len(games), 'señales:', report['signals_recorded'], 'excluidos:', excluded)
    print('Apuestas aprobadas: 0; apuestas realizadas: 0')


if __name__ == '__main__':
    main()

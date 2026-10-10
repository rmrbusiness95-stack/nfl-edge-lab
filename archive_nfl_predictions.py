"""Archive an immutable pre-kickoff model/odds snapshot from existing GitHub files.

No API calls, no recalibration, no picks. Run from the repository root.
Do not represent these model forecasts as validated betting edges.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

MODEL = Path('data/model_diagnostics.json')
ODDS = Path('data/nfl_odds_latest.json')
ROOT = Path('data/prediction_snapshots')

NAME_TO_CODE = {
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
    'Tennessee Titans':'TEN','Washington Commanders':'WAS',
}

def parse_utc(text: str) -> datetime:
    parsed = datetime.fromisoformat(str(text).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamp sin zona horaria')
    return parsed.astimezone(timezone.utc)


def main() -> None:
    model_bytes, odds_bytes = MODEL.read_bytes(), ODDS.read_bytes()
    model, odds = json.loads(model_bytes), json.loads(odds_bytes)
    created = datetime.now(timezone.utc)
    model_at, odds_at = parse_utc(model['generated_at_utc']), parse_utc(odds['generated_at_utc'])
    if model_at > created or odds_at > created:
        raise RuntimeError('Archivo de entrada fechado en el futuro; abortando.')
    projections = {row['game_id']: row for row in model.get('prospective', [])}
    if not projections or not odds.get('games'):
        raise RuntimeError('Sin proyecciones o cuotas: no se genera un respaldo vacío.')

    captured = []
    skipped = {'already_started': 0, 'no_model_match': 0, 'source_after_kickoff': 0, 'missing_bookmakers': 0}
    for event in odds['games']:
        kickoff = parse_utc(event['commence_time'])
        if kickoff <= created:
            skipped['already_started'] += 1
            continue
        if max(model_at, odds_at) >= kickoff:
            skipped['source_after_kickoff'] += 1
            continue
        home_code = NAME_TO_CODE.get(event.get('home_team'))
        away_code = NAME_TO_CODE.get(event.get('away_team'))
        if not home_code or not away_code:
            skipped['no_model_match'] += 1
            continue
        matches = [p for p in projections.values() if p['home_team'] == home_code and p['away_team'] == away_code
                   and p['season'] == kickoff.year]
        if len(matches) != 1:
            skipped['no_model_match'] += 1
            continue
        if not event.get('bookmakers'):
            skipped['missing_bookmakers'] += 1
            continue
        pred = matches[0]
        captured.append({
            'game_id': pred['game_id'], 'kickoff_utc': kickoff.isoformat(),
            'home_team': event['home_team'], 'away_team': event['away_team'],
            'model_home_margin_projection': pred['home_margin_projection'],
            'model_total_projection': pred['game_total_projection'],
            'projection_status': pred.get('status', 'experimental_unvalidated'),
            'books': event['bookmakers'],
        })

    if not captured:
        raise RuntimeError('Ningún partido pendiente con coincidencia exacta y cuotas; no se archiva.')
    # Identical input versions must never make duplicate snapshot files.
    identity = hashlib.sha256(model_bytes + b'\x00' + odds_bytes).hexdigest()[:20]
    destination = ROOT / odds_at.date().isoformat() / f'{identity}.json'
    if destination.exists():
        print('Ya archivado, sin sobrescribir:', destination)
        return
    record = {
        'archived_at_utc': created.isoformat(),
        'model_generated_at_utc': model_at.isoformat(),
        'odds_generated_at_utc': odds_at.isoformat(),
        'age_of_model_hours_at_archive': round((created-model_at).total_seconds()/3600, 2),
        'age_of_odds_hours_at_archive': round((created-odds_at).total_seconds()/3600, 2),
        'source_file_sha256': {'model': hashlib.sha256(model_bytes).hexdigest(), 'odds': hashlib.sha256(odds_bytes).hexdigest()},
        'archived_games': len(captured), 'skipped': skipped,
        'games': captured, 'approved_picks': [],
        'warnings': [
            'Registro del archivo disponible al archivar; NO prueba que el modelo haya sido creado por primera vez en ese instante.',
            'No son proyecciones ni cuotas actuales si las fuentes son antiguas; consultar las marcas de tiempo.',
            'No se confirma disponibilidad futura de cuotas ni apuestas efectivamente realizadas.',
            'Sin QB/lesiones verificados, probabilidades calibradas ni EV validado.',
            'No utilizar estos datos como picks aprobados.',
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Archivados {len(captured)} partidos en {destination}')
    print('Fuentes: modelo', model_at.isoformat(), 'cuotas', odds_at.isoformat())
    print('Omitidos:', skipped)

if __name__ == '__main__':
    main()

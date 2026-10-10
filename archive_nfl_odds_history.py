"""NFL Edge Lab — histórico inmutable de cuotas, para seguimiento de CLV.

Lee la última consulta guardada de The Odds API; NO llama a la API.
Conserva la respuesta completa, incluida la línea, momio, casa y hora por mercado.
No infiere que la cuota sea apertura/cierre, ni genera picks o EV.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SOURCE = Path('data/nfl_odds_latest.json')
HISTORY = Path('data/odds_history')


def timestamp(value):
    dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Se requiere una fecha UTC con zona horaria')
    return dt.astimezone(timezone.utc)


def main():
    source_bytes = SOURCE.read_bytes()
    data = json.loads(source_bytes)
    if data.get('source') != 'The Odds API v4' or not isinstance(data.get('games'), list):
        raise ValueError('El archivo no contiene una respuesta NFL reconocida')
    if not data['games']:
        raise ValueError('Sin partidos: no se genera archivo vacío')

    collected = timestamp(data['generated_at_utc'])
    now = datetime.now(timezone.utc)
    if collected > now:
        raise ValueError('La marca de tiempo de las cuotas está en el futuro')
    if (now - collected).total_seconds() > 7 * 86400:
        raise ValueError('Cuotas con más de 7 días: no archivar como consulta reciente')

    events = []
    skipped = {'started': 0, 'without_bookmakers': 0}
    for game in data['games']:
        kickoff = timestamp(game['commence_time'])
        # Este snapshot corresponde a la hora REAL de consulta, no a la hora del archivo.
        if collected >= kickoff:
            skipped['started'] += 1
            continue
        if not game.get('bookmakers'):
            skipped['without_bookmakers'] += 1
            continue
        events.append(game)
    if not events:
        raise ValueError('No hay partidos con cuotas anteriores al kickoff')

    digest = hashlib.sha256(source_bytes).hexdigest()
    folder = HISTORY / collected.strftime('%Y') / collected.strftime('%m')
    filename = collected.strftime('%Y%m%dT%H%M%SZ') + '-' + digest[:12] + '.json'
    target = folder / filename
    if target.exists():
        previous = json.loads(target.read_text(encoding='utf-8'))
        if previous.get('source_sha256') != digest:
            raise RuntimeError('Conflicto: mismo archivo de destino con distinto contenido')
        print('Consulta ya archivada:', target)
        return

    record = {
        'archive_created_utc': now.isoformat(),
        'odds_collected_utc': collected.isoformat(),
        'source_sha256': digest,
        'source': data.get('source'),
        'games_count': len(events),
        'skipped': skipped,
        'games': events,
        'approved_picks': [],
        'notes': [
            'Snapshot de cuotas existente, sin llamada adicional a la API.',
            'NO se identifica esta consulta automáticamente como apertura ni cierre.',
            'Para CLV se requieren al menos dos consultas anteriores al kickoff del mismo partido, mercado y casa.',
            'Comparar también el momio: una diferencia de puntos no basta para medir valor monetario.',
            'Sin pronósticos aprobados ni rentabilidad inferida.'
        ],
    }
    folder.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Guardada consulta del {collected.isoformat()} en {target}; partidos: {len(events)}')
    print('Sin consumo de créditos de The Odds API')


if __name__ == '__main__':
    main()

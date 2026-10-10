"""NFL Edge Lab — auditoría de disponibilidad por partido, sin afirmaciones de titularidad.

Lee fuentes existentes y publica una matriz fechada. No modifica proyecciones ni picks.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BASE=Path('data')
SOURCE=BASE/'nfl_availability.json'
ODDS=BASE/'nfl_odds_latest.json'
OUT=BASE/'game_availability_audit.json'
CODES={
'Arizona Cardinals':'ARI','Atlanta Falcons':'ATL','Baltimore Ravens':'BAL','Buffalo Bills':'BUF',
'Carolina Panthers':'CAR','Chicago Bears':'CHI','Cincinnati Bengals':'CIN','Cleveland Browns':'CLE',
'Dallas Cowboys':'DAL','Denver Broncos':'DEN','Detroit Lions':'DET','Green Bay Packers':'GB',
'Houston Texans':'HOU','Indianapolis Colts':'IND','Jacksonville Jaguars':'JAX',
'Kansas City Chiefs':'KC','Las Vegas Raiders':'LV','Los Angeles Chargers':'LAC',
'Los Angeles Rams':'LA','Miami Dolphins':'MIA','Minnesota Vikings':'MIN',
'New England Patriots':'NE','New Orleans Saints':'NO','New York Giants':'NYG',
'New York Jets':'NYJ','Philadelphia Eagles':'PHI','Pittsburgh Steelers':'PIT',
'San Francisco 49ers':'SF','Seattle Seahawks':'SEA','Tampa Bay Buccaneers':'TB',
'Tennessee Titans':'TEN','Washington Commanders':'WAS'
}

def dt(v):
    try:
        d=datetime.fromisoformat(str(v).replace('Z','+00:00'))
        return d.astimezone(timezone.utc) if d.tzinfo else None
    except (ValueError,TypeError): return None

def build(availability,odds,now=None):
    now=now or datetime.now(timezone.utc)
    byqb=defaultdict(list);byinj=defaultdict(list)
    for q in availability.get('qb_depth_chart_candidates',[]):
        if q.get('team'):byqb[q['team']].append(q)
    for i in availability.get('injury_reports',[]):
        if i.get('team'):byinj[i['team']].append(i)
    latest_by_team={t:max((int(i['week']) for i in rows if str(i.get('week','')).isdigit()),default=None) for t,rows in byinj.items()}
    output=[]
    for game in odds.get('games',[]):
        kickoff=dt(game.get('commence_time'))
        if kickoff is None or kickoff<=now:continue
        teams=[]
        for side in ('away','home'):
            name=game.get(side+'_team');code=CODES.get(name)
            if not code:raise ValueError('Nombre de equipo sin mapa: '+str(name))
            candidates=sorted(byqb.get(code,[]),key=lambda q:int(q.get('depth_rank')) if str(q.get('depth_rank','')).isdigit() else 999)
            latest=latest_by_team.get(code)
            injuryrows=[i for i in byinj.get(code,[]) if latest is not None and str(i.get('week'))==str(latest)]
            dated=[i for i in injuryrows if dt(i.get('report_date')) and dt(i.get('report_date'))<=now]
            # Reportes extraídos no identifican de modo fiable una designación vigente para este partido.
            snap=max((dt(q.get('snapshot_at_utc')) for q in candidates if dt(q.get('snapshot_at_utc'))),default=None)
            teams.append({'team':name,'abbr':code,'side':side,'qb_candidates':[
                {'name':q.get('player'),'depth_rank':q.get('depth_rank'),'source':q.get('source')}
                for q in candidates],
                'depth_chart_snapshot_at_utc':snap.isoformat() if snap else None,
                'qb_starter_confirmed':False,'qb_starter_name':None,
                'injury_latest_source_week':latest,'injury_rows_in_latest_week':len(injuryrows),
                'injury_rows_with_parseable_date':len(dated),
                'current_game_designations_verified':False,
                'safe_for_prediction_adjustments':False})
        output.append({'event_id':game.get('id'),'kickoff_utc':kickoff.isoformat(),
                       'away_team':game.get('away_team'),'home_team':game.get('home_team'),
                       'teams':teams,'prediction_eligibility':'blocked_unverified_qb_injuries'})
    return {'generated_at_utc':now.isoformat(),'availability_source_at_utc':availability.get('generated_at_utc'),
            'odds_source_at_utc':odds.get('generated_at_utc'),'upcoming_games':len(output),
            'ready_for_injury_adjusted_picks':False,'games':output,
            'warnings':['Las posiciones QB en depth chart NO prueban titularidad.',
                        'Lesiones sin fecha individual y designación del encuentro NO prueban situación actual.',
                        'La lista de juegos deriva de la última captura de cuotas, puede estar desactualizada.',
                        'No se modifica el modelo ni se generan picks; se necesitan fuentes oficiales fechadas por partido.']}

def main():
    availability=json.loads(SOURCE.read_text(encoding='utf-8'))
    odds=json.loads(ODDS.read_text(encoding='utf-8'))
    result=build(availability,odds)
    OUT.parent.mkdir(exist_ok=True,parents=True)
    OUT.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Partidos pendientes auditados:',result['upcoming_games'])
    print('Estado de ajustes por lesiones: BLOQUEADO (faltan verificaciones)')
    print('Salida:',OUT)
if __name__=='__main__':main()

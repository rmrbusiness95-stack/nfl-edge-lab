"""NFL Edge Lab: compact, dated QB depth snapshots and NFL injury reports.

No betting projections or confirmed starters. Do not rely on this as a live game-status service.
"""
from __future__ import annotations
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import nflreadpy as nfl

OUT = Path('data/nfl_availability.json')
SEASON = datetime.now(timezone.utc).year


def val(row, *cols):
    for col in cols:
        item = row.get(col)
        if item is not None and str(item).strip() and str(item).lower() not in ('nan','none'):
            return str(item).strip()
    return None


def read(loader, source):
    try:
        frame = loader([SEASON])
        if frame is None or frame.height == 0:
            raise ValueError('empty source')
        return frame.to_dicts(), {'status':'ok','raw_rows':frame.height,'columns':frame.columns}
    except Exception as exc:
        print(f'{source}: {type(exc).__name__}: {exc}')
        return [], {'status':'unavailable','error_type':type(exc).__name__}


def parse_time(txt):
    if not txt:
        return None
    try:
        d = datetime.fromisoformat(str(txt).replace('Z','+00:00'))
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def extract(charts, injuries, now=None):
    now = now or datetime.now(timezone.utc)
    snapshots = defaultdict(list)
    for r in charts:
        if val(r, 'pos_abb','position','depth_position','pos') != 'QB':
            continue
        team = val(r,'team','club_code','team_abbr')
        d = parse_time(val(r,'dt','date'))
        if not team or not d or d > now:
            continue
        snapshots[(team, d)].append(r)

    freshest = {}
    for (team,d), rows in snapshots.items():
        if team not in freshest or d > freshest[team][0]:
            freshest[team] = (d,rows)

    qbs=[]
    for team,(dt,rows) in sorted(freshest.items()):
        used=set()
        for row in sorted(rows,key=lambda r:(int(val(r,'pos_rank','depth_chart_order','depth_team') or '999') if (val(r,'pos_rank','depth_chart_order','depth_team') or '').isdigit() else 999, val(r,'player_name','full_name','name') or '')):
            pid = val(row,'gsis_id','espn_id')
            name=val(row,'player_name','full_name','name')
            if not name or (pid or name) in used:
                continue
            used.add(pid or name)
            qbs.append({'team':team,'player':name,'player_id':pid,
                        'depth_rank':val(row,'pos_rank','depth_chart_order','depth_team'),
                        'snapshot_at_utc':dt.isoformat(),
                        'starter_confirmed':False,
                        'source':'nflverse depth charts'})
    report=[]
    seen=set()
    for r in injuries:
        item={
            'team':val(r,'team','team_abbr'),
            'week':val(r,'week'),
            'player':val(r,'full_name','player_name','name'),
            'player_id':val(r,'gsis_id','player_id'),
            'position':val(r,'position','pos'),
            'practice_status':val(r,'practice_status','practice'),
            'game_designation':val(r,'report_status','game_status','injury_status'),
            'injury':val(r,'report_primary_injury','practice_primary_injury'),
            'report_date':val(r,'report_date','date_modified','date'),
            'source':'nflverse injury report',
        }
        if not item['player'] or not item['week']:
            continue
        key=tuple(item.values())
        if key not in seen:
            seen.add(key)
            report.append(item)
    return qbs,report,len(freshest)


def main():
    charts,cstatus=read(nfl.load_depth_charts,'depth_charts')
    injuries,istatus=read(nfl.load_injuries,'injuries')
    qb,report,teams=extract(charts,injuries)
    if cstatus['status']=='ok' and not qb:
        cstatus['status']='no_current_qb_matches'
    payload={
        'season':SEASON,
        'generated_at_utc':datetime.now(timezone.utc).isoformat(),
        'sources':{'depth_charts':cstatus,'injuries':istatus},
        'qb_snapshot_teams':teams,
        'qb_depth_chart_candidates':qb,
        'injury_reports':report,
        'prediction_ready':False,
        'note':('Las listas QB representan la ultima instantanea POR EQUIPO; no son titulares confirmados. '
                'Lesiones incluyen semanas previas, no estatus en vivo. Validar cobertura y fechas antes de pronosticar.'),
    }
    OUT.parent.mkdir(exist_ok=True,parents=True)
    temp=OUT.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    temp.replace(OUT)
    print(f'Equipos QB: {teams}; candidatos QB: {len(qb)}; reportes de lesión: {len(report)}')

if __name__=='__main__':
    main()

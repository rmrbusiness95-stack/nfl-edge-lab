import pandas as pd,json,math,datetime,os
from pathlib import Path
from io import StringIO
import requests
from collections import defaultdict
INPUT=os.getenv('NFL_GAMES_CSV')
if INPUT:
    x=pd.read_csv(INPUT,low_memory=False)
else:
    url='https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv'
    response=requests.get(url,timeout=60)
    response.raise_for_status()
    x=pd.read_csv(StringIO(response.text),low_memory=False)
required={'gameday','game_id','game_type','home_score','away_score','spread_line','home_team','away_team','season','week','home_rest','away_rest','weekday','div_game'}
if not required.issubset(x.columns):
    raise ValueError('Faltan columnas: '+str(sorted(required-set(x.columns))))
x['game_date']=pd.to_datetime(x['gameday'],errors='coerce')
x=x.sort_values(['game_date','game_id']).reset_index(drop=True)
reference_day=pd.Timestamp(os.getenv('NFL_REFERENCE_DATE', datetime.date.today().isoformat()))
played=x[(x.home_score.notna()) & (x.away_score.notna()) & (x.spread_line.notna()) & (x.game_type=='REG') & (x.season>=2002) & (x.game_date<reference_day)].copy()
all_by_team=defaultdict(list)
for r in played.itertuples(index=False):
    margin=float(r.home_score)-float(r.away_score)
    for side,team,opp,score_diff in [('home',r.home_team,r.away_team,margin),('away',r.away_team,r.home_team,-margin)]:
        ats_margin=margin-float(r.spread_line) if side=='home' else -(margin-float(r.spread_line))
        all_by_team[team].append(dict(date=r.game_date,season=int(r.season),week=int(r.week),side=side,opponent=opp,div=bool(r.div_game),rest=float(r.home_rest if side=='home' else r.away_rest) if pd.notna(r.home_rest if side=='home' else r.away_rest) else None,spread=(-float(r.spread_line) if side=='home' else float(r.spread_line)),ats=1 if ats_margin>0.01 else -1 if ats_margin<-.01 else 0, margin=score_diff,weekday=r.weekday,game_id=r.game_id))
for team,rs in all_by_team.items():
    rs.sort(key=lambda r:(r['date'],r['game_id']))
    for i,r in enumerate(rs):
        prev=rs[i-1] if i else None
        pp=rs[i-2] if i>1 else None
        r['prior_loss']=prev is not None and prev['margin']<0
        r['prior_ats_loss']=prev is not None and prev['ats']==-1
        r['prior_two_ats_losses']=prev is not None and pp is not None and prev['ats']==-1 and pp['ats']==-1
        r['prior_big_loss']=prev is not None and prev['margin']<=-14
        r['home_after_away_loss']=r['side']=='home' and prev is not None and prev['side']=='away' and prev['margin']<0
        r['previous_mnf']=prev is not None and prev['weekday']=='Monday'

criteria=[
 ('home_away','En esta condición de local/visitante',lambda r,c:r['side']==c['side']),
 ('divisional','En partidos divisionales',lambda r,c:r['div'] if c['div'] else False),
 ('prior_loss','Después de perder',lambda r,c:r['prior_loss']),
 ('prior_ats_loss','Después de no cubrir el spread',lambda r,c:r['prior_ats_loss']),
 ('prior_two_ats_losses','Después de 2 partidos sin cubrir',lambda r,c:r['prior_two_ats_losses']),
 ('prior_big_loss','Después de perder por 14+ puntos',lambda r,c:r['prior_big_loss']),
 ('previous_mnf','Después de jugar Monday Night',lambda r,c:r['previous_mnf']),
 ('short_rest','Con descanso corto (≤6 días)',lambda r,c:r['rest'] is not None and r['rest']<=6 and c['rest'] is not None and c['rest']<=6),
 ('long_rest','Con descanso largo (≥9 días)',lambda r,c:r['rest'] is not None and r['rest']>=9 and c['rest'] is not None and c['rest']>=9),
 ('home_after_away_loss','Local después de perder de visitante',lambda r,c:r['home_after_away_loss'] if c['side']=='home' else False),
 ('underdog','Cuando era underdog',lambda r,c:r['spread']>0 if c['spread']>0 else False),
 ('favorite','Cuando era favorito',lambda r,c:r['spread']<0 if c['spread']<0 else False),
]
today=pd.Timestamp(os.getenv('NFL_REFERENCE_DATE', datetime.date.today().isoformat()))
# Próximo domingo; incluir únicamente la jornada domingo-lunes.
next_sunday=today+pd.Timedelta(days=(6-today.weekday())%7)
end_monday=next_sunday+pd.Timedelta(days=1)
upcoming=x[(x.game_date>=next_sunday)&(x.game_date<=end_monday) & x.home_score.isna() & (x.game_type=='REG')].copy()
if upcoming.empty:
    raise RuntimeError(f'No hay partidos domingo-lunes pendientes entre {next_sunday.date()} y {end_monday.date()}; no sobrescribir datos antiguos.')
output=[]
for g in upcoming.itertuples(index=False):
  gdate=g.game_date
  game={'game_id':g.game_id,'date':g.gameday,'week':int(g.week),'home':g.home_team,'away':g.away_team,'market_reference_spread_home':float(g.spread_line) if pd.notna(g.spread_line) else None,'market_reference_verified_live':False,'teams':[]}
  for side in ['home','away']:
    team=g.home_team if side=='home' else g.away_team
    history=[r for r in all_by_team[team] if r['date']<gdate and r['season']>=2002]
    prev=history[-1] if history else None
    spread=-float(g.spread_line) if side=='home' else float(g.spread_line)
    rest=g.home_rest if side=='home' else g.away_rest
    ctx={'side':side,'div':bool(g.div_game),'rest':float(rest) if pd.notna(rest) else None,'spread':spread}
    matched=[]
    # recent team ATS
    relevant=history[-12:]
    if len(relevant)>=8:
      w=sum(r['ats']==1 for r in relevant);l=sum(r['ats']==-1 for r in relevant);p=sum(r['ats']==0 for r in relevant)
      matched.append({'id':'recent12','label':'Últimos 12 partidos (ATS)','w':w,'l':l,'p':p,'rate':round(w/(w+l)*100,1) if w+l else None,'n':len(relevant),'scope':'equipo, 2002–presente'})
    for id,label,fn in criteria:
      if id=='prior_loss' and not(prev and prev['margin']<0):continue
      if id=='prior_ats_loss' and not(prev and prev['ats']==-1):continue
      if id=='prior_two_ats_losses' and not(len(history)>=2 and all(r['ats']==-1 for r in history[-2:])):continue
      if id=='prior_big_loss' and not(prev and prev['margin']<=-14):continue
      if id=='previous_mnf' and not(prev and prev['weekday']=='Monday'):continue
      if id=='home_after_away_loss' and not(side=='home' and prev and prev['side']=='away' and prev['margin']<0):continue
      if id=='short_rest' and not(ctx['rest'] is not None and ctx['rest']<=6):continue
      if id=='long_rest' and not(ctx['rest'] is not None and ctx['rest']>=9):continue
      if id=='divisional' and not ctx['div']:continue
      if id=='underdog' and not spread>0:continue
      if id=='favorite' and not spread<0:continue
      rel=[r for r in history if fn(r,ctx)]
      if len(rel)<12:continue
      w=sum(r['ats']==1 for r in rel);l=sum(r['ats']==-1 for r in rel);p=sum(r['ats']==0 for r in rel)
      matched.append({'id':id,'label':label,'w':w,'l':l,'p':p,'rate':round(w/(w+l)*100,1) if w+l else None,'n':len(rel),'scope':'equipo, desde 2002'})
    game['teams'].append({'team':team,'side':side,'trends':matched})
  output.append(game)
result={'generated_for_date':str(today.date()),'source':'nflverse nfldata games.csv','trend_sample_start':2002,'methodology':'ATS: margen equipo + puntos del equipo > 0; pushes excluidos del porcentaje; sólo partidos previos; condiciones situacionales históricas','warnings':['Tendencias descriptivas; no predicen probabilidades ni son picks.','Las líneas del CSV no son cuotas actuales verificadas; tendencias favorito/underdog podrían cambiar con mercado.','Sin corrección por búsqueda de múltiples condiciones; no interpretar récords extremos como ventaja.'],'games':output}
# Validaciones: totales ATS consistentes, porcentajes y fechas previas.
for game in output:
    for team in game['teams']:
        for t in team['trends']:
            if t['w']+t['l']+t['p']!=t['n']:
                raise ValueError(f'Recuento inconsistente: {game["game_id"]}, {team["team"]}, {t["id"]}')
            expected=round(100*t['w']/(t['w']+t['l']),1) if t['w']+t['l'] else None
            if t['rate']!=expected:
                raise ValueError('Porcentaje ATS inconsistente')
if len({g['game_id'] for g in output})!=len(output):
    raise ValueError('Partidos duplicados')
result['audit']={'games':len(output), 'trend_records':sum(len(t['trends']) for g in output for t in g['teams']), 'historical_seasons':int(played.season.nunique()),'historical_played_games_with_spreads':int(len(played)), 'passed':True}
result['warnings'].append('Auditoría interna superada; aún falta comprobar spreads contra proveedor independiente.')
Path('data').mkdir(exist_ok=True)
with open('data/nfl_weekend_trends.json','w',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('games',len(output),'total trends',sum(len(t['trends']) for g in output for t in g['teams']))
for g in output:print(g['away'], '@',g['home'], ' / '.join(t['team']+':'+str(len(t['trends'])) for t in g['teams']))

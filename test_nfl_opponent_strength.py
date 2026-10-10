"""NFL Edge Lab: diagnostic opponent-strength adjustment vs original model.

No future-game data enters features. Historical odds are reference lines only.
Does NOT publish picks or change the production model.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from build_nfl_model import SCHEDULE_URL, finite, game_features, load_weekly, ridge_fit, predict

YEARS = (2022, 2023, 2024, 2025)
OUT = Path('data/opponent_strength_test.json')
WINDOW = 8


def mae(actual, forecast):
    return round(float(np.mean(np.abs(np.array(actual) - np.array(forecast)))), 3)


def previous_team_avg(mapping, season, team, side, week, metric):
    past = [(w, r) for w, r in mapping.get((season, team, side), {}).items() if w < week]
    past.sort(key=lambda v: v[0], reverse=True)
    values = [finite(row.get(metric)) for _, row in past[:WINDOW]]
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if len(values) >= 3 else None


def opponent_features(mapping, schedule_opponents, season, week, home, away):
    """Mean strength of opponents faced in past 8 team games, known before game."""
    stats = []
    for team in (home, away):
        previous = sorted([w for w in mapping.get((season, team, 'offense'), {}) if w < week], reverse=True)[:WINDOW]
        def_opp, off_opp = [], []
        for prior_week in previous:
            opponent = schedule_opponents.get((season, prior_week, team))
            if not opponent:
                continue
            # All opposing statistics are restricted to < current week.
            def_value = previous_team_avg(mapping, season, opponent, 'defense', week, 'epa_per_play_allowed')
            off_value = previous_team_avg(mapping, season, opponent, 'offense', week, 'epa_per_play')
            if def_value is not None:
                def_opp.append(def_value)
            if off_value is not None:
                off_opp.append(off_value)
        if len(def_opp) < 3 or len(off_opp) < 3:
            return None
        stats.extend([float(np.mean(def_opp)), float(np.mean(off_opp))])
    # Home opponent defense, home opponent offense, away equivalents;
    # two relative strength differences are convenient additional features.
    return stats + [stats[0] - stats[2], stats[1] - stats[3]]


def main():
    mapping, _ = load_weekly()
    response = requests.get(SCHEDULE_URL, timeout=60)
    response.raise_for_status()
    games = pd.read_csv(StringIO(response.text), low_memory=False)
    required = {'season', 'week', 'game_type', 'home_team', 'away_team',
                'home_score', 'away_score', 'spread_line', 'total_line'}
    if not required.issubset(games.columns):
        raise ValueError(f'Missing columns: {sorted(required-set(games.columns))}')
    opponents = {}
    for g in games.itertuples(index=False):
        if str(g.game_type) != 'REG':
            continue
        try:
            season, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        home, away = str(g.home_team), str(g.away_team)
        opponents[season, week, home] = away
        opponents[season, week, away] = home

    rows = []
    for g in games.itertuples(index=False):
        if str(g.game_type) != 'REG':
            continue
        try:
            season, week = int(g.season), int(g.week)
        except (TypeError, ValueError):
            continue
        hs, aws = finite(g.home_score), finite(g.away_score)
        if hs is None or aws is None:
            continue
        home, away = str(g.home_team), str(g.away_team)
        base = game_features(mapping, season, week, home, away)
        if base is None:
            continue
        extra = opponent_features(mapping, opponents, season, week, home, away)
        if extra is None:
            continue
        rows.append({'season': season, 'week': week, 'base': base, 'candidate': base+extra,
                     'margin': hs-aws, 'total': hs+aws,
                     'market_margin': finite(g.spread_line), 'market_total': finite(g.total_line)})

    yearly = []
    for year in YEARS:
        train = [r for r in rows if r['season'] < year]
        test = [r for r in rows if r['season'] == year]
        if len(train) < 500 or len(test) < 100:
            raise RuntimeError(f'Not enough matched data in {year}: train={len(train)}, test={len(test)}')
        report = {'year': year, 'train_games': len(train), 'test_games': len(test), 'metrics': {}}
        for target in ('margin', 'total'):
            ytr = np.array([r[target] for r in train])
            yt = np.array([r[target] for r in test])
            original = predict(ridge_fit([r['base'] for r in train], ytr), [r['base'] for r in test])
            candidate = predict(ridge_fit([r['candidate'] for r in train], ytr), [r['candidate'] for r in test])
            market_key = 'market_'+target
            eligible = [i for i,r in enumerate(test) if r[market_key] is not None]
            if len(eligible) < 100:
                raise RuntimeError(f'Insufficient market records for {year} {target}')
            actual = yt[eligible]
            market = [test[i][market_key] for i in eligible]
            report['metrics'][target] = {
                'n': len(eligible),
                'original_mae': mae(actual, original[eligible]),
                'candidate_mae': mae(actual, candidate[eligible]),
                'market_mae': mae(actual, market),
                'late_17_18': {
                    'n': sum(test[i]['week']>=17 for i in eligible),
                    'original_mae': mae([yt[i] for i in eligible if test[i]['week']>=17], [original[i] for i in eligible if test[i]['week']>=17]),
                    'candidate_mae': mae([yt[i] for i in eligible if test[i]['week']>=17], [candidate[i] for i in eligible if test[i]['week']>=17]),
                }
            }
        yearly.append(report)
    output = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'experiment':'historical_opponent_strength_as_of_predicted_week',
              'years':yearly, 'approved_picks':[], 'automatic_promotion':False,
              'limitations':[
                  'Only matched samples with at least three prior opponent strength observations; not directly comparable to previous 831-game tests.',
                  'Reference market spreads and totals are not independently authenticated point-in-time executable prices.',
                  '2022-2025 already inspected; exploratory retrospective results, not an untouched holdout.',
                  'No quarterback injury adjustment or calibrated probability, ROI or EV.',
                  'Opponent strength values computed using opponents prior to current game, not necessarily their rating at the time the earlier matchup was played.',
                  'Does not replace the existing production model or authorize picks.'
              ]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    for r in yearly:
        print(r['year'], r['test_games'], {t: r['metrics'][t] for t in ('margin','total')})
    print('Output:',OUT,' approved picks: 0')

if __name__ == '__main__':
    main()

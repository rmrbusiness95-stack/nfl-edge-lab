"""NFL Edge Lab: auditoría RETROSPECTIVA vs líneas de CIERRE NFLverse.

Las líneas no tienen marca temporal anterior a kickoff; NO prueban posibilidad
de apostar a ese precio. No usa cuotas de 2026 ni valida EV / ROI / picks.
El entrenamiento usa años anteriores a la temporada evaluada; las variables
de equipo usan únicamente semanas anteriores.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from build_nfl_model import load_weekly, game_features, ridge_fit, predict, finite, SCHEDULE_URL

OUT = Path("data/market_discrepancy_audit.json")
YEARS = (2022, 2023, 2024, 2025)
THRESHOLDS = (0, 3, 5, 7)


def summarize(rows):
    n = len(rows)
    if not n:
        return {"games": 0, "decisions": 0, "covers": 0, "losses": 0, "pushes": 0, "cover_rate_excluding_pushes": None}
    covers = sum(r["ats_result"] == "cover" for r in rows)
    losses = sum(r["ats_result"] == "loss" for r in rows)
    pushes = n - covers - losses
    return {
        "games": n, "decisions": covers + losses,
        "covers": covers, "losses": losses, "pushes": pushes,
        "cover_rate_excluding_pushes": round(covers / (covers + losses), 4) if covers + losses else None,
        "mean_abs_model_market_difference": round(float(np.mean([abs(r["edge_points"]) for r in rows])), 3),
        "model_margin_mae": round(float(np.mean([abs(r["actual_home_margin"] - r["model_home_margin"]) for r in rows])), 3),
        "market_margin_mae": round(float(np.mean([abs(r["actual_home_margin"] - r["market_home_margin"]) for r in rows])), 3),
    }


def main():
    mapping, _ = load_weekly()
    resp = requests.get(SCHEDULE_URL, timeout=60)
    resp.raise_for_status()
    games = pd.read_csv(StringIO(resp.text), low_memory=False)
    needed = {"season", "week", "game_type", "home_team", "away_team",
              "home_score", "away_score", "spread_line"}
    missing = needed - set(games.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas en NFLverse: {sorted(missing)}")
    historical, eval_games = [], []
    for g in games.itertuples(index=False):
        try:
            year, week = int(g.season), int(g.week)
        except (ValueError, TypeError, AttributeError):
            continue
        if g.game_type != "REG" or year > max(YEARS):
            continue
        hs, aws = finite(g.home_score), finite(g.away_score)
        if hs is None or aws is None:
            continue
        features = game_features(mapping, year, week, str(g.home_team), str(g.away_team))
        if features is None:
            continue
        rec = {"year": year, "week": week, "home": str(g.home_team),
               "away": str(g.away_team), "x": features, "actual_margin": hs - aws,
               "market_margin": finite(g.spread_line)}
        historical.append(rec)
        if year in YEARS and rec["market_margin"] is not None:
            eval_games.append(rec)

    predictions = []
    train_sizes = {}
    for year in YEARS:
        train = [r for r in historical if r["year"] < year]
        test = [r for r in eval_games if r["year"] == year]
        if len(train) < 500 or len(test) < 100:
            raise RuntimeError(f"Cobertura insuficiente en {year}: train={len(train)} test={len(test)}")
        fit = ridge_fit([r["x"] for r in train], [r["actual_margin"] for r in train])
        pred = predict(fit, [r["x"] for r in test]).reshape(-1)
        train_sizes[str(year)] = len(train)
        for r, p in zip(test, pred):
            edge = float(p) - r["market_margin"]
            if abs(edge) < 1e-10:
                continue  # Sin diferencia, no existe selección de lado.
            signed_cover_margin = (r["actual_margin"] - r["market_margin"]) * (1 if edge > 0 else -1)
            result = "cover" if signed_cover_margin > 1e-9 else ("loss" if signed_cover_margin < -1e-9 else "push")
            predictions.append({"year": year, "model_home_margin": round(float(p), 5),
                                "market_home_margin": r["market_margin"],
                                "actual_home_margin": r["actual_margin"],
                                "edge_points": round(edge, 5),
                                "ats_result": result})

    overall = {str(t): summarize([r for r in predictions if abs(r["edge_points"]) >= t]) for t in THRESHOLDS}
    yearly = {str(y): {str(t): summarize([r for r in predictions if r["year"] == y and abs(r["edge_points"]) >= t]) for t in THRESHOLDS} for y in YEARS}
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "market_source": "nflverse nfldata games.csv / PFR closing spread",
        "market_spread_sign": "positive = home favored; convert to expected home margin",
        "market_lines_are_closing": True,
        "lines_available_before_kickoff_verified": False,
        "evaluation_years": list(YEARS), "train_games_by_year": train_sizes,
        "evaluated_games_with_nonzero_discrepancy": len(predictions),
        "thresholds_absolute_edge_points": overall,
        "yearly_thresholds": yearly,
        "no_vig_calibration_done": False, "roi_measured": False,
        "prospective_bettable_edge_proven": False,
        "model_modified": False, "approved_picks": [],
        "warnings": [
            "Usar cierres históricos como precio seleccionable antes del partido sería sesgo de anticipación de mercado.",
            "Un 55% ATS frente al cierre retrospectivo NO prueba apuestas disponibles ni rentabilidad.",
            "2022-2025 ya se revisaron durante el proyecto; no son validación verdaderamente intocada.",
            "Los umbrales 0,3,5,7 son descriptivos; no optimizarlos sobre la misma muestra.",
            "Falta comparar cuotas capturadas ANTES del kickoff y sus momios para calcular EV real.",
            "Cero picks aprobados. No ajustar proyecciones con estos resultados sin nueva validación."
        ]
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Partidos con discrepancia:", len(predictions))
    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("SIN VALIDACIÓN DE EV NI APUESTAS: solo referencia retrospectiva de cierre.")


if __name__ == "__main__":
    main()

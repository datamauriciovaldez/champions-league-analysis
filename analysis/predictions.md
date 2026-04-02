# Análisis de Modelos de Machine Learning

Generado: 2026-04-02

## Mejor Modelo: Random Forest

**Accuracy:** 80.65%
**F1 Macro:** 72.74%
**Log Loss:** 0.5185

## Reporte de Clasificación (Mejor Modelo)

```
                    precision    recall  f1-score   support

    Victoria Local       0.78      1.00      0.87        31
            Empate       1.00      0.25      0.40        16
Victoria Visitante       0.83      1.00      0.91        15

          accuracy                           0.81        62
         macro avg       0.87      0.75      0.73        62
      weighted avg       0.85      0.81      0.76        62

```

## Features Utilizadas

- `form_home`
- `form_away`
- `rating_diff`
- `xG_diff`
- `possession_diff`
- `shots_diff`
- `elo_diff`
- `phase_num`

## Metodología de Validación

Validación temporal (backtesting por temporadas):
- **Train:** 2015-16 → 2022-23
- **Test:** 2023-24 → 2024-25

*Generado por ClawdBot · 2026-04-02*

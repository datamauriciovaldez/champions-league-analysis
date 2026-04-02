# Memoria Académica — UEFA Champions League Data Analysis

**Proyecto de Ciencia de Datos**
**Autor:** Mauricio Valdez Fusté
**Fecha:** 2026-04-02

---

## 1. Introducción y Objetivos

La UEFA Champions League constituye el torneo de fútbol de clubes más prestigioso del mundo,
ofreciendo un entorno competitivo de alta calidad con datos estructurados a lo largo de
múltiples temporadas. Este proyecto plantea la aplicación de técnicas de ciencia de datos y
machine learning para:

1. Caracterizar estadísticamente el torneo (2015-2025)
2. Identificar factores determinantes del resultado
3. Comparar modelos de predicción de resultados (H/D/A)
4. Establecer una metodología reproducible y extensible

## 2. Revisión de Trabajos Relacionados

La predicción de resultados en fútbol ha sido abordada desde múltiples perspectivas:

- **Modelos de Poisson bivariado** (Dixon & Coles, 1997): modelan goles independientes por equipo
- **Random Forests** (Hvattum & Arntzen, 2010): uso de ratings Elo como features principales
- **Gradient Boosting** (Constantinou et al., 2012): superiores en datos desequilibrados
- **Redes neuronales recurrentes** (reciente): capturan dependencias temporales en rachas

Este proyecto adopta un enfoque híbrido con features de Elo + estadísticas de partido.

## 3. Datos y Metodología

### 3.1 Fuente de Datos

- **Dataset:** UCL 2015-2025 (310 partidos, 10 temporadas)
- **Variables disponibles:** 35 columnas incluyendo xG, posesión, tiro, tarjetas
- **Período de cobertura:** Fase de grupos → Final en cada temporada

### 3.2 Feature Engineering

Se construyeron las siguientes variables derivadas:

| Feature | Descripción | Justificación |
|---------|-------------|---------------|
| `form_home` / `form_away` | Media W-rate últimos 5 partidos | Captura momentum reciente |
| `elo_diff` | Diferencia de rating Elo | Proxy de calidad relativa |
| `xG_diff` | Diferencia Expected Goals | Calidad de ocasiones generadas |
| `rating_diff` | Diferencia de rating UEFA | Fuerza histórica relativa |
| `phase_num` | Codificación ordinal de fase | Importancia del partido |

### 3.3 Validación Temporal (Backtesting)

Se empleó validación temporal en lugar de cross-validation aleatorio:
- **Train:** Temporadas 2015-16 a 2022-23
- **Test:** Temporadas 2023-24 y 2024-25

Este enfoque evita data leakage temporal, siendo más realista para evaluación predictiva.

## 4. Resultados EDA

### 4.1 Distribución de Resultados

- Victoria local: 54.8% — ventaja de campo significativa en UCL.
- Fase más goleadora: Final (4.60 goles/partido).
- Equipo dominante: Arsenal con 26 victorias.
- Los ratings Elo reflejan ciclos de dominio europeo por equipo.
- Correlación xG_home vs goles_home: r=0.959 — alta validez del indicador.

### 4.2 Análisis por Fase

Las fases eliminatorias presentan tendencia a mayor igualdad (más empates y victorias
visitante), consistente con la teoría de que en eliminatorias los equipos más equiparados
se enfrentan entre sí.

## 5. Modelos, Resultados y Discusión

### 5.1 Métricas de Evaluación

Se emplean tres métricas complementarias:

- **Accuracy:** proporción de predicciones correctas (interpretabilidad)
- **F1 Macro:** media de F1 por clase, penaliza desequilibrios (empates sub-representados)
- **Log Loss:** evalúa calibración de probabilidades (rigor probabilístico)

### 5.2 Resultados Comparativos

```
Modelo                  Accuracy    F1 Macro    Log Loss
─────────────────────────────────────────────────────
Baseline (mayoría)          50.0%      22.22%    18.0218
Regresión Logística        70.97%      58.64%     0.4492
Random Forest              80.65%      72.74%     0.5185
Gradient Boosting          79.03%      73.61%     1.0799
```

**Modelo recomendado:** Random Forest (Accuracy: 80.65%, F1: 72.74%)

### 5.3 Discusión

El Gradient Boosting y Random Forest superan consistentemente a la regresión logística,
lo que sugiere relaciones no lineales entre features y resultado. El feature más importante
es la diferencia de Elo, confirmando que la calidad relativa de los equipos es el predictor
más robusto del resultado en UCL.

## 6. Conclusiones y Trabajo Futuro

### Conclusiones principales

1. La ventaja local existe pero es menos pronunciada que en ligas nacionales (~54.8%)
2. El Elo rating es el predictor más potente de resultados
3. Las fases eliminatorias muestran mayor incertidumbre que la fase de grupos
4. Random Forest y Gradient Boosting son los modelos más adecuados para este problema

### Líneas Futuras

- Integración de datos reales de lesiones y sanciones
- Modelos de Poisson bivariado para predicción de marcadores exactos
- LSTM/Transformers para capturar dependencias secuenciales
- Incorporar datos de Transfermarkt (valor de mercado de plantillas)

---

# SoccerSolver — Market Value Simulator

Simulador **what-if** de valor de mercado de futbolistas. Predice el valor de
un jugador con 4 modelos XGBoost entrenados por separado según su posición
(GK / DEF / MID / FWD) sobre el logaritmo del valor de mercado, y explica la
predicción con SHAP. Permite mover sliders de edad, años de contrato y
métricas de rendimiento para ver cómo cambiaría el valor estimado.

## Contenido

- `app.py` — interfaz Streamlit del simulador.
- `entrenar_modelos.py` — entrena los 4 modelos y genera `modelos_xgb.pkl` (se
  ejecuta una sola vez en local; la app en la nube solo carga el resultado).
- `modelos_xgb.pkl` — modelos ya entrenados + features por grupo + explainers
  SHAP, listos para cargar.
- `dataset_model_2425_FINAL.csv` — dataset oficial del proyecto, completo
  (temporada 24/25, fuentes: SofaScore + Transfermarkt).
- `requirements.txt` — dependencias con versión fijada.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

Si quieres reentrenar los modelos con datos nuevos (mismo esquema de
columnas que `dataset_model_2425_FINAL.csv`):

```bash
python entrenar_modelos.py
```

Esto regenera `modelos_xgb.pkl`, que `app.py` carga por ruta relativa.

## Desplegar en Streamlit Cloud

1. Sube esta carpeta a un repositorio de GitHub (ver comandos más abajo).
2. En [share.streamlit.io](https://share.streamlit.io), crea una nueva app
   apuntando al repo, rama y archivo `app.py`.
3. Streamlit Cloud instala `requirements.txt` y arranca la app; no necesita
   acceso a ningún archivo fuera del repositorio.

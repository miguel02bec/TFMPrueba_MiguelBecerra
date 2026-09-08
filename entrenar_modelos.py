"""
entrenar_modelos.py
Entrena los 4 modelos XGBoost estratificados por posicion (GK/DEF/MID/FWD)
sobre log(valor de mercado) y calcula SHAP. Guarda todo en modelos_xgb.pkl
para que app.py solo tenga que cargar el archivo (sin reentrenar en la nube).

Ejecutar una sola vez, desde esta carpeta:
    python entrenar_modelos.py
"""
import logging
import platform
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ── Configuracion ──────────────────────────────────────────────────
# Ruta relativa: el csv oficial vive en esta misma carpeta del proyecto. No
# se copia ni se recompone automaticamente desde ningun sitio del PC: si
# falta, el script debe fallar en vez de sobrescribirlo desde otra fuente.
DATASET_PATH = "dataset_model_2425_FINAL.csv"
MODEL_OUTPUT_PATH = "modelos_xgb.pkl"

# Grupo de posicion al que pertenece cada codigo de posicion del dataset.
POSITION_GROUPS = {
    "GK": ["G"],
    "DEF": ["D"],
    "MID": ["M"],
    "FWD": ["F"],
}

# Features por grupo. age y contract_years_left estan en los 4 grupos porque
# el simulador (app.py) tiene sliders de edad y anios de contrato que solo
# mueven la prediccion si el modelo los usa como feature.
FEATURES_BY_GROUP = {
    "GK": [
        "total_minutes", "avg_rating",
        "passes_p90", "pass_accuracy",
        "total_matches_missed", "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "DEF": [
        "total_minutes", "avg_rating",
        "passes_p90", "pass_accuracy", "key_passes_p90",
        "duels_won_p90", "interceptions_p90", "tackles_p90",
        "goals_p90", "assists_p90",
        "total_matches_missed", "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "MID": [
        "total_minutes", "avg_rating",
        "goals_p90", "assists_p90", "shots_p90",
        "xg_p90", "key_passes_p90", "pass_accuracy",
        "duels_won_p90", "interceptions_p90",
        "total_matches_missed", "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "FWD": [
        "total_minutes", "avg_rating",
        "goals_p90", "assists_p90", "shots_p90",
        "xg_p90", "key_passes_p90", "pass_accuracy",
        "duels_won_p90",
        "total_matches_missed", "avg_days_per_injury",
        "age", "contract_years_left",
    ],
}

TARGET_COLUMN = "log_target_value"

# Semilla unica compartida por el split de CV y por los modelos, para que
# todo el script sea reproducible con un solo numero.
SEED = 42
N_CV_FOLDS = 5

# Hiperparametros comunes a todos los grupos.
BASE_MODEL_PARAMS = dict(
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    n_jobs=-1,
)

# GK tiene muchas menos muestras (162 jugadores, ~130 por fold de CV) que el
# resto de grupos (394-807). Con la misma profundidad/arboles que los grupos
# grandes, el modelo de porteros sobreajusta: en las pruebas anteriores fue
# el grupo con peor R2 (0.485) y mas varianza entre folds (0.360-0.590).
#   - max_depth=3 (vs 6): arboles mas superficiales (max. 8 hojas en vez de
#     64) limitan la capacidad de memorizar ruido de muestra.
#   - n_estimators=200 (vs 500): arboles mas simples necesitan menos rondas
#     de boosting para capturar la senal; menos rondas tambien reduce el
#     riesgo de sobreajustar poco a poco el ruido residual.
#   - reg_lambda=5.0 (vs 1.0 por defecto): mas regularizacion L2 sobre el
#     valor de cada hoja, que con pocas muestras por hoja es una estimacion
#     mas ruidosa y conviene encoger hacia la media.
#   - min_child_weight=5 (vs 1 por defecto): obliga a cada hoja a apoyarse
#     en varias muestras en vez de aislar a un solo jugador atipico.
SMALL_GROUP_PARAMS = dict(
    **BASE_MODEL_PARAMS,
    max_depth=3,
    n_estimators=200,
    reg_lambda=5.0,
    min_child_weight=5,
)

# DEF/MID/FWD tienen entre 394 y 807 jugadores: suficiente muestra para
# mantener la configuracion original, que no mostro señales de sobreajuste.
LARGE_GROUP_PARAMS = dict(
    **BASE_MODEL_PARAMS,
    max_depth=6,
    n_estimators=500,
    reg_lambda=1.0,
    min_child_weight=1,
)

MODEL_PARAMS_BY_GROUP = {
    "GK": SMALL_GROUP_PARAMS,
    "DEF": LARGE_GROUP_PARAMS,
    "MID": LARGE_GROUP_PARAMS,
    "FWD": LARGE_GROUP_PARAMS,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Funciones ─────────────────────────────────────────────────────

def get_position_group(position: str) -> str:
    """Traduce un codigo de posicion (G/D/M/F) a su grupo de modelo."""
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return "MID"


def validar_columnas(df: pd.DataFrame) -> None:
    """Comprueba que estan todas las columnas necesarias antes de entrenar,
    para fallar con un mensaje claro en vez de un KeyError generico a mitad
    de entrenamiento."""
    columnas_necesarias = {"position", TARGET_COLUMN}
    for features in FEATURES_BY_GROUP.values():
        columnas_necesarias.update(features)

    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en '{DATASET_PATH}' necesarias para entrenar: "
            f"{sorted(faltantes)}"
        )


def evaluar_modelo(params: dict, X: pd.DataFrame, y: pd.Series, group_name: str) -> float:
    """Valida con K-Fold CV y devuelve el R2 medio, para comprobar que
    entrena razonablemente antes de ajustar el modelo final. Crea un
    XGBRegressor nuevo en cada fold: nunca reutiliza ni muta el modelo que
    luego se entrena con el 100% de los datos y se guarda en el .pkl."""
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    mae_scores, rmse_scores, r2_scores = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        fold_model = xgb.XGBRegressor(**params)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)
        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        r2_scores.append(r2_score(y_val, y_pred))
        logger.info("  [%s] Fold %d - MAE: %.3f | RMSE: %.3f | R2: %.3f",
                    group_name, fold + 1, mae_scores[-1], rmse_scores[-1], r2_scores[-1])

    r2_medio = float(np.mean(r2_scores))
    logger.info("[%s] Media CV - MAE: %.3f | RMSE: %.3f | R2: %.3f +/- %.3f",
                group_name, np.mean(mae_scores), np.mean(rmse_scores),
                r2_medio, np.std(r2_scores))
    return r2_medio


def entrenar_grupo(df: pd.DataFrame, group: str, features: list) -> dict:
    """Entrena y explica (SHAP) el modelo de un grupo de posicion."""
    df_group = df[df["position_group"] == group].copy()
    logger.info("Grupo %s - %d jugadores", group, len(df_group))

    if len(df_group) < N_CV_FOLDS:
        raise ValueError(
            f"Grupo '{group}' tiene solo {len(df_group)} jugadores, menos que "
            f"N_CV_FOLDS={N_CV_FOLDS}. No se puede validar con K-Fold; revisa "
            f"el dataset o baja N_CV_FOLDS."
        )

    X = df_group[features]
    y = df_group[TARGET_COLUMN]

    nulos = X.isna().sum()
    for feature, n_nulos in nulos[nulos > 0].items():
        logger.info("  [%s] %s: %d/%d nulos (%.1f%%)",
                    group, feature, n_nulos, len(X), 100 * n_nulos / len(X))

    params = MODEL_PARAMS_BY_GROUP[group]
    r2_cv = evaluar_modelo(params, X, y, group)

    model = xgb.XGBRegressor(**params)  # instancia nueva para el modelo final
    model.fit(X, y)  # reentrena con todos los datos del grupo tras validar

    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "features": features,
        "explainer": explainer,
        "r2_cv": r2_cv,
    }


def construir_metadata(df: pd.DataFrame, modelos: dict) -> dict:
    """Recoge version de librerias y metricas de entrenamiento junto al
    .pkl, para poder diagnosticar en el futuro con que datos/entorno se
    entreno sin tener que volver a ejecutar nada."""
    return {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": DATASET_PATH,
        "n_rows_total": len(df),
        "n_rows_by_group": {
            group: int((df["position_group"] == group).sum())
            for group in FEATURES_BY_GROUP
        },
        "r2_cv_by_group": {group: data["r2_cv"] for group, data in modelos.items()},
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "shap": shap.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }


def verificar_pkl_guardado(df: pd.DataFrame) -> None:
    """Recarga el .pkl recien escrito y hace una prediccion de prueba por
    grupo, para detectar problemas de serializacion (ej. que el explainer
    SHAP no se haya guardado bien) en el momento, no cuando la app ya este
    desplegada."""
    reloaded = joblib.load(MODEL_OUTPUT_PATH)
    for group, data in reloaded["models"].items():
        df_group = df[df["position_group"] == group]
        if df_group.empty:
            logger.warning("  [%s] sin filas para el sanity check, se omite", group)
            continue

        features = data["features"]
        sample = df_group.iloc[[0]][features].fillna(0)

        pred_eur = float(np.expm1(data["model"].predict(sample)[0]))
        shap_vals = data["explainer"].shap_values(sample)

        if not np.isfinite(pred_eur):
            raise RuntimeError(f"Sanity check fallido en grupo {group}: prediccion no finita")

        logger.info("  [%s] sanity check OK - prediccion ejemplo: EUR %.1fM, shap shape %s",
                    group, pred_eur / 1e6, shap_vals.shape)


def run() -> None:
    df = pd.read_csv(DATASET_PATH)
    validar_columnas(df)
    df["position_group"] = df["position"].apply(get_position_group)

    modelos = {}
    for group, features in FEATURES_BY_GROUP.items():
        modelos[group] = entrenar_grupo(df, group, features)

    metadata = construir_metadata(df, modelos)

    joblib.dump({
        "models": modelos,
        "position_groups": POSITION_GROUPS,
        "metadata": metadata,
    }, MODEL_OUTPUT_PATH)
    logger.info("Modelos guardados en %s", MODEL_OUTPUT_PATH)

    verificar_pkl_guardado(df)

    # ── Resumen final: R2 por posicion ───────────────────────────────
    print("\n=== R2 (validacion cruzada, media 5 folds) por posicion ===")
    for group, data in modelos.items():
        print(f"  {group}: R2 = {data['r2_cv']:.3f}")


if __name__ == "__main__":
    run()

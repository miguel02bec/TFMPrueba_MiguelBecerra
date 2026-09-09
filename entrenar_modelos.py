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
# avg_rating se quito por fuga (el rating medio de SofaScore ya recoge una
# valoracion casi directa del jugador, correlacionada con su valor de
# mercado). matches_played y total_matches_missed se quitaron por
# multicolinealidad (redundantes entre si y con total_minutes /
# avg_days_per_injury).
FEATURES_BY_GROUP = {
    "GK": [
        "total_minutes",
        "passes_p90", "pass_accuracy", "saves_p90",
        "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "DEF": [
        "total_minutes",
        "passes_p90", "pass_accuracy", "key_passes_p90",
        "duels_won_p90", "interceptions_p90", "tackles_p90",
        "goals_p90", "assists_p90",
        "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "MID": [
        "total_minutes",
        "goals_p90", "assists_p90", "shots_p90",
        "xg_p90", "key_passes_p90", "pass_accuracy",
        "duels_won_p90", "interceptions_p90",
        "avg_days_per_injury",
        "age", "contract_years_left",
    ],
    "FWD": [
        "total_minutes",
        "goals_p90", "assists_p90", "shots_p90",
        "xg_p90", "key_passes_p90", "pass_accuracy",
        "duels_won_p90",
        "avg_days_per_injury",
        "age", "contract_years_left",
    ],
}

TARGET_COLUMN = "log_target_value"
LEAGUE_COLUMN = "league"

# Columna de fuga: es una estimacion de valor de mercado de la propia fuente
# SofaScore, correlacionada casi directamente con el target (market_value_eur /
# log_target_value). Se descarta tanto del dataset como de las features de
# entrenamiento.
LEAKAGE_COLUMNS = ["market_value_sofascore"]

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
    columnas_necesarias = {"position", TARGET_COLUMN, LEAGUE_COLUMN}
    for features in FEATURES_BY_GROUP.values():
        columnas_necesarias.update(features)

    faltantes = columnas_necesarias - set(df.columns)
    if faltantes:
        raise ValueError(
            f"Faltan columnas en '{DATASET_PATH}' necesarias para entrenar: "
            f"{sorted(faltantes)}"
        )


def quitar_columnas_fuga(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina del dataset las columnas de fuga antes de que puedan llegar a
    usarse como feature en ningun grupo."""
    presentes = [c for c in LEAKAGE_COLUMNS if c in df.columns]
    if presentes:
        logger.info("Quitando columnas de fuga del dataset: %s", presentes)
        df = df.drop(columns=presentes)
    return df


def construir_matriz(df_slice: pd.DataFrame, base_features: list, league_categories: list) -> pd.DataFrame:
    """Construye la matriz de diseño (features numericas + liga one-hot) para
    un grupo de posicion. La liga se fija como Categorical con las categorias
    de TODO el dataset (no solo las del grupo/fold) para que las columnas
    dummy resultantes sean siempre las mismas, entrene con el grupo completo,
    con un fold de CV, o con un solo jugador en el simulador."""
    sub = df_slice[base_features + [LEAGUE_COLUMN]].copy()
    sub[LEAGUE_COLUMN] = pd.Categorical(sub[LEAGUE_COLUMN], categories=league_categories)
    return pd.get_dummies(sub, columns=[LEAGUE_COLUMN])


def evaluar_modelo(params: dict, X: pd.DataFrame, y: pd.Series, group_name: str) -> tuple:
    """Valida con K-Fold CV y devuelve (R2 medio, R2 desviacion tipica), para
    comprobar que entrena razonablemente antes de ajustar el modelo final.
    Crea un XGBRegressor nuevo en cada fold: nunca reutiliza ni muta el
    modelo que luego se entrena con el 100% de los datos y se guarda en el
    .pkl."""
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
    r2_std = float(np.std(r2_scores))
    logger.info("[%s] Media CV - MAE: %.3f | RMSE: %.3f | R2: %.3f +/- %.3f",
                group_name, np.mean(mae_scores), np.mean(rmse_scores),
                r2_medio, r2_std)
    return r2_medio, r2_std


def entrenar_grupo(df: pd.DataFrame, group: str, base_features: list, league_categories: list) -> dict:
    """Entrena y explica (SHAP) el modelo de un grupo de posicion. La liga se
    añade como one-hot junto a las features numericas del grupo."""
    df_group = df[df["position_group"] == group].copy()
    logger.info("Grupo %s - %d jugadores", group, len(df_group))

    if len(df_group) < N_CV_FOLDS:
        raise ValueError(
            f"Grupo '{group}' tiene solo {len(df_group)} jugadores, menos que "
            f"N_CV_FOLDS={N_CV_FOLDS}. No se puede validar con K-Fold; revisa "
            f"el dataset o baja N_CV_FOLDS."
        )

    X = construir_matriz(df_group, base_features, league_categories)
    y = df_group[TARGET_COLUMN]
    features = list(X.columns)

    nulos = X.isna().sum()
    for feature, n_nulos in nulos[nulos > 0].items():
        logger.info("  [%s] %s: %d/%d nulos (%.1f%%)",
                    group, feature, n_nulos, len(X), 100 * n_nulos / len(X))

    params = MODEL_PARAMS_BY_GROUP[group]
    r2_cv, r2_cv_std = evaluar_modelo(params, X, y, group)

    model = xgb.XGBRegressor(**params)  # instancia nueva para el modelo final
    model.fit(X, y)  # reentrena con todos los datos del grupo tras validar

    explainer = shap.TreeExplainer(model)

    return {
        "model": model,
        "base_features": base_features,
        "features": features,
        "explainer": explainer,
        "r2_cv": r2_cv,
        "r2_cv_std": r2_cv_std,
    }


def construir_metadata(df: pd.DataFrame, modelos: dict, league_categories: list) -> dict:
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
        "league_categories": league_categories,
        "r2_cv_by_group": {group: data["r2_cv"] for group, data in modelos.items()},
        "r2_cv_std_by_group": {group: data["r2_cv_std"] for group, data in modelos.items()},
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
    league_categories = reloaded["metadata"]["league_categories"]
    for group, data in reloaded["models"].items():
        df_group = df[df["position_group"] == group]
        if df_group.empty:
            logger.warning("  [%s] sin filas para el sanity check, se omite", group)
            continue

        sample = construir_matriz(df_group.iloc[[0]], data["base_features"], league_categories)
        sample = sample.reindex(columns=data["features"], fill_value=0).fillna(0)

        pred_eur = float(np.expm1(data["model"].predict(sample)[0]))
        shap_vals = data["explainer"].shap_values(sample)

        if not np.isfinite(pred_eur):
            raise RuntimeError(f"Sanity check fallido en grupo {group}: prediccion no finita")

        logger.info("  [%s] sanity check OK - prediccion ejemplo: EUR %.1fM, shap shape %s",
                    group, pred_eur / 1e6, shap_vals.shape)


def run() -> None:
    df = pd.read_csv(DATASET_PATH)
    df = quitar_columnas_fuga(df)
    validar_columnas(df)
    df["position_group"] = df["position"].apply(get_position_group)

    # Categorias de liga fijas para TODO el dataset: asi las columnas dummy
    # de cada grupo/fold/prediccion individual son siempre las mismas,
    # aunque ese subconjunto no contenga todas las ligas.
    league_categories = sorted(df[LEAGUE_COLUMN].dropna().unique().tolist())
    logger.info("Ligas (categorias fijas para el one-hot): %s", league_categories)

    modelos = {}
    for group, base_features in FEATURES_BY_GROUP.items():
        modelos[group] = entrenar_grupo(df, group, base_features, league_categories)

    metadata = construir_metadata(df, modelos, league_categories)

    joblib.dump({
        "models": modelos,
        "position_groups": POSITION_GROUPS,
        "metadata": metadata,
    }, MODEL_OUTPUT_PATH)
    logger.info("Modelos guardados en %s", MODEL_OUTPUT_PATH)

    verificar_pkl_guardado(df)

    # ── Resumen final: R2 por posicion ───────────────────────────────
    print("\n=== R2 (validacion cruzada, media +/- desviacion, 5 folds) por posicion ===")
    for group, data in modelos.items():
        print(f"  {group}: R2 = {data['r2_cv']:.3f} +/- {data['r2_cv_std']:.3f}")


if __name__ == "__main__":
    run()

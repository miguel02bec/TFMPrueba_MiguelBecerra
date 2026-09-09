"""
app.py
SoccerSolver - Simulador what-if de valor de mercado de futbolistas.
Modelos XGBoost estratificados por posición (GK/DEF/MID/FWD), explicados con SHAP.
Carga modelos_xgb.pkl y dataset_model_2425_FINAL.csv con rutas relativas: no depende
de ningún archivo fuera de esta carpeta, listo para Streamlit Cloud.
"""
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Configuración ─────────────────────────────────────────────
st.set_page_config(
    page_title="Market Value Simulator",
    page_icon="⚽",
    layout="wide",
)

DATASET_PATH = "dataset_model_2425_FINAL.csv"
MODEL_PATH = "modelos_xgb.pkl"
LEAGUE_COLUMN = "league"


# ── Estilo personalizado (solo presentación) ────────────────────
def aplicar_estilo():
    """Oculta el chrome por defecto de Streamlit (menú, footer, badge).
    No fija colores ni fondos propios: el tema (claro, ver .streamlit/config.toml)
    ya da contraste consistente sin depender del modo claro/oscuro del navegador
    de quien visita la app. No toca datos ni modelo."""
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stDecoration"] {display: none;}
        .viewerBadge_container__1QSob {display: none;}
    </style>
    """, unsafe_allow_html=True)


aplicar_estilo()


# ── Cargar modelo y datos (rutas relativas, sin nada del disco local) ─────
@st.cache_resource(ttl=0)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATASET_PATH)


artifact = load_model()
models = artifact["models"]
POSITION_GROUPS = artifact["position_groups"]
LEAGUE_CATEGORIES = artifact["metadata"]["league_categories"]

df = load_dataset()

POSITION_MAP = {
    "Todas": None,
    "Defensa": "D",
    "Centrocampista": "M",
    "Delantero": "F",
    "Portero": "G",
}

POSITION_NAMES = {
    "D": "Defensa",
    "M": "Centrocampista",
    "F": "Delantero",
    "G": "Portero",
}


def get_position_group(position: str) -> str:
    for group, positions in POSITION_GROUPS.items():
        if position in positions:
            return group
    return "MID"


# ── Funciones ─────────────────────────────────────────────────

def build_model_input(player_features: dict, league: str, position_group: str) -> pd.DataFrame:
    """Construye la matriz de entrada del modelo igual que entrenar_modelos.py:
    features numericas del grupo + liga one-hot con las mismas categorias
    fijas usadas en el entrenamiento (guardadas en el .pkl). El reindex
    garantiza que, aunque la liga del jugador no aparezca en get_dummies (p.ej.
    si faltara del dataset), las columnas finales coincidan con las que el
    modelo espera."""
    model_data = models[position_group]
    base_features = model_data["base_features"]
    row = {f: player_features[f] for f in base_features}
    row[LEAGUE_COLUMN] = league
    X = pd.DataFrame([row])
    X[LEAGUE_COLUMN] = pd.Categorical(X[LEAGUE_COLUMN], categories=LEAGUE_CATEGORIES)
    X = pd.get_dummies(X, columns=[LEAGUE_COLUMN])
    return X.reindex(columns=model_data["features"], fill_value=0)


def predict_value(player_features: dict, league: str, position_group: str) -> float:
    model_data = models[position_group]
    X = build_model_input(player_features, league, position_group)
    log_val = model_data["model"].predict(X)[0]
    return float(np.expm1(log_val))


def get_shap_values(player_features: dict, league: str, position_group: str) -> pd.DataFrame:
    model_data = models[position_group]
    explainer = model_data.get("explainer")
    if explainer is None:
        explainer = shap.TreeExplainer(model_data["model"])
    X = build_model_input(player_features, league, position_group)
    shap_vals = explainer.shap_values(X)[0]
    feature_names = [f.replace("_", " ").title() for f in X.columns]
    return pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals),
    }).sort_values("abs_shap", ascending=False)


def format_value(value_eur: float) -> str:
    if value_eur >= 1_000_000:
        return f"€{value_eur/1_000_000:.1f}M"
    return f"€{value_eur/1_000:.0f}K"


# ── Cabecera ─────────────────────────────────────────────────
st.title("Market Value Simulator")
st.markdown(""Busca un jugador, ajusta sus atributos y observa cómo cambia su valor de mercado.")
st.divider()

# ── Sidebar: guardar escenarios ─────────────────────────────────
with st.sidebar:
    st.header("💾 Escenarios")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {}

    scenario_name = st.text_input("Nombre del escenario")
    save_btn = st.button("Guardar escenario", use_container_width=True)


# ── Layout principal: dos columnas (buscador + sliders | valor + SHAP) ───
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("##### 🔍 Buscar jugador")
    row_pos, row_player = st.columns([1, 2])
    with row_pos:
        position_filter = st.selectbox("Posición", list(POSITION_MAP.keys()))
    pos_code = POSITION_MAP[position_filter]

    if pos_code:
        df_filtered = df[df["position"] == pos_code]
    else:
        df_filtered = df

    player_names = sorted(df_filtered["player_name"].unique().tolist())
    with row_player:
        selected_player = st.selectbox("Jugador", player_names)

    # ── Cargar datos del jugador seleccionado ─────────────────
    player_data = df[df["player_name"] == selected_player].iloc[0]
    position_group = get_position_group(player_data["position"])
    features = models[position_group]["base_features"]
    player_league = player_data[LEAGUE_COLUMN]

    # Si al jugador le falta algún dato (p.ej. contract_years_left en agentes
    # libres sin contrato registrado), usamos la mediana del grupo como valor
    # de partida para que el slider tenga con qué arrancar.
    group_defaults = df[df["position"].isin(POSITION_GROUPS[position_group])][features].median(numeric_only=True)
    base_features = {}
    for f in features:
        val = player_data[f]
        if pd.isna(val):
            val = group_defaults[f]
        base_features[f] = float(val)

    # Base value fijo por jugador — no cambia al mover sliders
    if "last_player" not in st.session_state or st.session_state.last_player != selected_player:
        st.session_state.last_player = selected_player
        st.session_state.base_value = predict_value(base_features, player_league, position_group)

    base_value = st.session_state.base_value
    pos_name = POSITION_NAMES.get(player_data["position"], player_data["position"])

    st.subheader(f"🏃 {selected_player}")
    st.caption(
        f"Posición: {pos_name} | Grupo: {position_group} | Liga: {player_league} | "
        f"Partidos: {player_data['matches_played']:.0f} | Minutos: {player_data['total_minutes']:.0f}"
    )

    st.markdown("### Ajustar atributos")
    current_features = base_features.copy()

    with st.expander("🧑 Perfil y contrato", expanded=True):
        current_features["age"] = float(st.slider(
            "Edad", 16, 42, int(round(base_features["age"])), 1
        ))
        current_features["contract_years_left"] = float(st.slider(
            "Años de contrato restantes", 0, 10, int(round(base_features["contract_years_left"])), 1
        ))

    with st.expander("⚡ Rendimiento ofensivo", expanded=True):
        if "goals_p90" in features:
            current_features["goals_p90"] = st.slider(
                "Goles por 90", 0.0, 2.0, float(base_features["goals_p90"]), 0.01
            )
        if "assists_p90" in features:
            current_features["assists_p90"] = st.slider(
                "Asistencias por 90", 0.0, 1.5, float(base_features["assists_p90"]), 0.01
            )
        if "shots_p90" in features:
            current_features["shots_p90"] = st.slider(
                "Tiros por 90", 0.0, 8.0, float(base_features["shots_p90"]), 0.1
            )
        if "xg_p90" in features:
            current_features["xg_p90"] = st.slider(
                "xG por 90", 0.0, 1.5, float(base_features["xg_p90"]), 0.01
            )

    with st.expander("🎯 Creación de juego"):
        current_features["passes_p90"] = st.slider(
            "Pases por 90", 0.0, 120.0, float(base_features["passes_p90"]), 0.5
        )
        current_features["pass_accuracy"] = st.slider(
            "Precisión de pase", 0.0, 1.0, float(base_features["pass_accuracy"]), 0.01
        )
        if "key_passes_p90" in features:
            current_features["key_passes_p90"] = st.slider(
                "Pases clave por 90", 0.0, 5.0, float(base_features["key_passes_p90"]), 0.1
            )
        if "xa_p90" in features:
            current_features["xa_p90"] = st.slider(
                "xA por 90", 0.0, 1.0, float(base_features["xa_p90"]), 0.01
            )

    with st.expander("🛡️ Defensa y duelos"):
        if "duels_won_p90" in features:
            current_features["duels_won_p90"] = st.slider(
                "Duelos ganados por 90", 0.0, 20.0, float(base_features["duels_won_p90"]), 0.1
            )
        if "interceptions_p90" in features:
            current_features["interceptions_p90"] = st.slider(
                "Intercepciones por 90", 0.0, 10.0, float(base_features["interceptions_p90"]), 0.1
            )
        if "tackles_p90" in features:
            current_features["tackles_p90"] = st.slider(
                "Tackles por 90", 0.0, 10.0, float(base_features["tackles_p90"]), 0.1
            )

    with st.expander("🏥 Historial de lesiones"):
        current_features["avg_days_per_injury"] = st.slider(
            "Días medios por lesión", 0.0, 100.0, float(base_features["avg_days_per_injury"]), 0.5
        )

with col2:
    current_value = predict_value(current_features, player_league, position_group)
    delta = current_value - base_value
    delta_pct = (delta / base_value) * 100

    with st.container(border=True):
        st.markdown("### 💰 Valor de Mercado")

        m_orig, m_sim = st.columns(2)
        m_orig.metric("Valor Original", format_value(base_value))
        m_sim.metric(
            "Valor Simulado",
            format_value(current_value),
            delta=f"{delta_pct:+.1f}% ({'+' if delta >= 0 else ''}{format_value(abs(delta))})",
        )

        if save_btn and scenario_name:
            st.session_state.scenarios[scenario_name] = {
                "value": current_value,
                "features": current_features.copy(),
                "player": selected_player,
            }
            st.success(f"Escenario '{scenario_name}' guardado")

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Valor Inicial", "Valor Actual"],
            y=[base_value / 1e6, current_value / 1e6],
            marker_color=["#3b82f6", "#16A34A" if delta >= 0 else "#ef4444"],
            text=[format_value(base_value), format_value(current_value)],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Inicial vs Actual",
            yaxis_title="Millones €",
            height=260,
            margin=dict(t=40, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1F2937",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.container(border=True):
        st.markdown("### 🔑 Atributos con mayor impacto")
        st.caption("Explicación SHAP de la predicción actual (con los sliders ajustados a la izquierda).")
        shap_df = get_shap_values(current_features, player_league, position_group)
        top_shap = shap_df.head(5)

        fig_shap = go.Figure()
        fig_shap.add_trace(go.Bar(
            x=top_shap["shap_value"],
            y=top_shap["feature"],
            orientation="h",
            marker_color=["#16A34A" if v >= 0 else "#ef4444" for v in top_shap["shap_value"]],
        ))
        fig_shap.update_layout(
            title="Top 5 factores",
            height=320,
            margin=dict(t=40, b=20),
            xaxis_title="Impacto SHAP",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1F2937",
        )
        st.plotly_chart(fig_shap, use_container_width=True)


# ── Comparar escenarios ────────────────────────────────────────
st.divider()

if st.session_state.scenarios:
    st.subheader("📊 Comparar Escenarios")

    scenario_data = []
    for name, data in st.session_state.scenarios.items():
        scenario_data.append({
            "Escenario": name,
            "Jugador": data["player"],
            "Valor": format_value(data["value"]),
            "Valor (€M)": round(data["value"] / 1e6, 1),
        })

    df_scenarios = pd.DataFrame(scenario_data)

    col_table, col_chart = st.columns([1, 1], gap="large")
    with col_table:
        with st.container(border=True):
            st.dataframe(df_scenarios[["Escenario", "Jugador", "Valor"]], use_container_width=True)

    with col_chart:
        with st.container(border=True):
            fig_comp = px.bar(
                df_scenarios, x="Escenario", y="Valor (€M)",
                color="Jugador", title="Comparación de escenarios",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_comp.update_layout(
                height=320, margin=dict(t=40, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1F2937",
            )
            st.plotly_chart(fig_comp, use_container_width=True)
else:
    st.info("Guarda al menos un escenario desde la barra lateral (⬅️ **💾 Escenarios**) para poder compararlos aquí.")


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("SoccerSolver Market Value Engine | Datos: SofaScore + Transfermarkt | Modelo: XGBoost estratificado por posición")

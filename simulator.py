"""
simulator.py
Simulador interactivo de valor de mercado de jugadores.
Modelos estratificados por posición: GK, DEF, MID, FWD.
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

# ── Cargar modelo y datos ─────────────────────────────────────
@st.cache_resource(ttl=0)
def load_model():
    return joblib.load("model_market_value.pkl")

@st.cache_data
def load_dataset():
    df = pd.read_csv("dataset_model.csv")
    return df

artifact = load_model()
models = artifact["models"]
POSITION_GROUPS = artifact["position_groups"]

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

def predict_value(player_features: dict, position_group: str) -> float:
    model_data = models[position_group]
    features = model_data["features"]
    model = model_data["model"]
    X = pd.DataFrame([player_features])[features]
    log_val = model.predict(X)[0]
    return np.expm1(log_val)


def get_shap_values(player_features: dict, position_group: str) -> pd.DataFrame:
    model_data = models[position_group]
    features = model_data["features"]
    model = model_data["model"]
    X = pd.DataFrame([player_features])[features]
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]
    feature_names = [f.replace("_", " ").title() for f in features]
    return pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals,
        "abs_shap": np.abs(shap_vals),
    }).sort_values("abs_shap", ascending=False)


def format_value(value_eur: float) -> str:
    if value_eur >= 1_000_000:
        return f"€{value_eur/1_000_000:.1f}M"
    return f"€{value_eur/1_000:.0f}K"


# ── UI ────────────────────────────────────────────────────────

st.title("⚽ Market Value Simulator")
st.markdown("Busca un jugador, ajusta sus atributos y observa cómo cambia su valor de mercado.")

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("🔍 Seleccionar Jugador")

    position_filter = st.selectbox("Posición", list(POSITION_MAP.keys()))
    pos_code = POSITION_MAP[position_filter]

    if pos_code:
        df_filtered = df[df["position"] == pos_code]
    else:
        df_filtered = df

    player_names = sorted(df_filtered["player_name"].unique().tolist())
    selected_player = st.selectbox("Jugador", player_names)

    st.divider()
    st.header("💾 Escenarios")

    if "scenarios" not in st.session_state:
        st.session_state.scenarios = {}

    scenario_name = st.text_input("Nombre del escenario")
    save_btn = st.button("Guardar escenario")


# ── Cargar datos del jugador seleccionado ─────────────────────
player_data = df[df["player_name"] == selected_player].iloc[0]
position_group = get_position_group(player_data["position"])
features = models[position_group]["features"]

base_features = {f: float(player_data[f]) for f in features}

# Base value fijo por jugador — no cambia al mover sliders
if "last_player" not in st.session_state or st.session_state.last_player != selected_player:
    st.session_state.last_player = selected_player
    st.session_state.base_value = predict_value(base_features, position_group)

base_value = st.session_state.base_value


# ── Layout principal ──────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    pos_name = POSITION_NAMES.get(player_data["position"], player_data["position"])
    st.subheader(f"🏃 {selected_player}")
    st.caption(f"Posición: {pos_name} | Grupo: {position_group} | Partidos: {player_data['matches_played']:.0f} | Minutos: {player_data['total_minutes']:.0f}")

    st.markdown("### Ajustar atributos")
    current_features = base_features.copy()

    with st.expander("⚡ Rendimiento ofensivo", expanded=True):
        current_features["avg_rating"] = st.slider(
            "Rating medio", 4.0, 10.0, float(base_features["avg_rating"]), 0.1
        )
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
        current_features["total_matches_missed"] = st.slider(
            "Partidos perdidos total", 0, 100, int(base_features["total_matches_missed"]), 1
        )
        current_features["avg_days_per_injury"] = st.slider(
            "Días medios por lesión", 0.0, 100.0, float(base_features["avg_days_per_injury"]), 0.5
        )

with col2:
    current_value = predict_value(current_features, position_group)
    delta = current_value - base_value
    delta_pct = (delta / base_value) * 100

    st.markdown("### 💰 Valor de Mercado")
    st.metric("Valor actual", format_value(current_value))
    arrow = "↑" if delta >= 0 else "↓"
    st.markdown(f"**{arrow} {'+' if delta >= 0 else ''}{format_value(abs(delta))} ({delta_pct:+.1f}%)**")

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
        marker_color=["#3b82f6", "#10b981" if delta >= 0 else "#ef4444"],
        text=[format_value(base_value), format_value(current_value)],
        textposition="outside",
    ))
    fig_bar.update_layout(
        title="Inicial vs Actual",
        yaxis_title="Millones €",
        height=300,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("### 🔑 Atributos con mayor impacto")
    shap_df = get_shap_values(current_features, position_group)
    top_shap = shap_df.head(5)

    fig_shap = go.Figure()
    fig_shap.add_trace(go.Bar(
        x=top_shap["shap_value"],
        y=top_shap["feature"],
        orientation="h",
        marker_color=["#10b981" if v >= 0 else "#ef4444" for v in top_shap["shap_value"]],
    ))
    fig_shap.update_layout(
        title="Top 5 factores",
        height=300,
        margin=dict(t=40, b=20),
        xaxis_title="Impacto SHAP",
    )
    st.plotly_chart(fig_shap, use_container_width=True)


# ── Comparar escenarios ────────────────────────────────────────
if st.session_state.scenarios:
    st.divider()
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

    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.dataframe(df_scenarios[["Escenario", "Jugador", "Valor"]], use_container_width=True)

    with col_chart:
        fig_comp = px.bar(
            df_scenarios, x="Escenario", y="Valor (€M)",
            color="Jugador", title="Comparación de escenarios",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_comp.update_layout(height=300, margin=dict(t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("SoccerSolver Market Value Engine | Datos: SofaScore + Transfermarkt | Modelo: XGBoost estratificado por posición")
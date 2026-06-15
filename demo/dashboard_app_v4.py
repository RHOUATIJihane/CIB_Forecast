"""
CIB Forecasting — Dashboard politique par catégorie (v4)
Justifie les règles : champion modèle + activation externes par catégorie A–F.
Visualise le lien EXP → catégorie → décision.

Lancer :
  streamlit run rapport_pfe/dashboard/dashboard_app_v4.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Constantes ───────────────────────────────────────────────────────────────

C_POS = "#1D9E75"
C_NEG = "#D85A30"
C_NEU = "#EF9F27"
C_BASE = "#94A3B8"
C_EXT = "#2563EB"
C_CHAMP = "#F59E0B"
LIFT_THRESHOLD = 3.0
ML_MODELS = ["ridge", "rf", "lgbm"]

EXP_SPECS: dict[str, dict] = {
    "EXP1": {
        "label": "EXP1 — Qualité du signal",
        "conditions": ["A_clean", "B_noisy", "C_short"],
        "role": "calibration",
        "question": "Signal propre, bruité ou historique court : les externes compensent-ils ?",
        "policy_impact": (
            "Encapsule A_regular_stable (A_clean), D_noisy (B_noisy) et E_short_history (C_short). "
            "Justifie l'activation externes sur A (+5 %) et le refus sur D_noisy."
        ),
    },
    "EXP2": {
        "label": "EXP2 — Ruptures calendaires",
        "conditions": ["A_no_break", "B_amplified_w70", "C_level_shift_w70"],
        "role": "calibration",
        "question": "Les flags calendaires aident-ils après amplification des effets de paie / fin de mois ?",
        "policy_impact": (
            "Encapsule C_trending (C_level_shift) — RF sans externes (+2 %). "
            "B_amplified → flags calendaires dans le mapping sectoriel, pas une catégorie compte."
        ),
    },
    "EXP3": {
        "label": "EXP3 — Segmentation profil & secteur",
        "conditions": ["regular", "irregular"],
        "role": "calibration",
        "question": "Le lift varie-t-il selon la régularité du compte ?",
        "policy_impact": (
            "Encapsule D_irregular (irregular, externes ON +5 %) et B_seasonal (seasonal). "
            "Sépare le pool D_volatile v6 en profils distincts."
        ),
    },
    "EXP4": {
        "label": "EXP4 — Qualité des externes",
        "conditions": ["A_perfect", "B_lag1w", "C_lag3w", "D_noisy30", "E_missing20"],
        "role": "calibration",
        "question": "Retard, bruit ou valeurs manquantes sur les macro : à partir de quand le gain disparaît ?",
        "policy_impact": "Fixe les exigences SLA sur les flux macro (choix de conception, pas règle par compte).",
    },
    "EXP5": {
        "label": "EXP5 — Causalité (garde-fou)",
        "conditions": ["correct_external", "wrong_external", "random_external"],
        "role": "validation",
        "question": "Le gain vient-il d'un signal causal ou du surapprentissage ?",
        "policy_impact": "Valide la politique : pas de gain avec externes aléatoires → règles fiables.",
    },
    "EXP6": {
        "label": "EXP6 — Profils extrêmes",
        "conditions": ["sparse", "volatile", "flat", "irregular_normal"],
        "role": "calibration",
        "question": "Comptes sparse, flat, très volatils : les externes aident-ils ?",
        "policy_impact": (
            "Encapsule D_sparse (sparse) et F_flat (flat) — externes OFF. "
            "Complète E_short_history pour les comptes à faible historique."
        ),
    },
    "EXP7": {
        "label": "EXP7 — Robustesse des seuils",
        "conditions": ["strict", "moderate", "loose"],
        "role": "validation",
        "question": "Les conclusions changent-elles si on assouplit les seuils de qualification ?",
        "policy_impact": "Confirme la stabilité des seuils de catégorisation — pas de règle prod directe.",
    },
}

CAT_LABELS = {
    "A_regular_stable": "A — Régulier stable · RF ON",
    "B_seasonal": "B — Saisonnier · LGBM OFF",
    "C_trending": "C — Tendance · RF OFF",
    "D_noisy": "D₁ — Bruité · RIDGE OFF",
    "D_irregular": "D₂ — Irrégulier · RIDGE ON",
    "D_sparse": "D₃ — Sparse · RIDGE OFF",
    "D_volatile": "D — Volatile (v6 legacy)",
    "E_short_history": "E — Historique court · LGBM OFF",
    "F_flat": "F — Quasi plat · RF OFF",
    "F_mixed": "F — Mixte (v6 legacy)",
}

CAT_POLICY_SOURCE: dict[str, str] = {
    "E_short_history": "EXP1 · C_short",
    "D_noisy": "EXP1 · B_noisy",
    "B_seasonal": "EXP3 · account_type=seasonal",
    "D_sparse": "EXP6 · sparse",
    "F_flat": "EXP6 · flat",
    "C_trending": "EXP2 · C_level_shift_w70",
    "A_regular_stable": "EXP1 · A_clean",
    "D_irregular": "EXP3 · irregular",
}

COND_LABELS = {
    "A_clean": "Signal propre",
    "B_noisy": "Signal bruité",
    "C_short": "Historique court (28s)",
    "A_no_break": "Sans rupture",
    "B_amplified_w70": "Amplif. calendaire w70",
    "C_level_shift_w70": "Choc de niveau w70",
    "regular": "Compte régulier",
    "irregular": "Compte irrégulier",
    "A_perfect": "Externes parfaits",
    "B_lag1w": "Retard 1 sem.",
    "C_lag3w": "Retard 3 sem.",
    "D_noisy30": "Bruit +30 %",
    "E_missing20": "Manquants 20 %",
    "correct_external": "Bon secteur",
    "wrong_external": "Mauvais secteur",
    "random_external": "Aléatoire",
    "sparse": "Sparse",
    "volatile": "Volatile extrême",
    "flat": "Quasi constant",
    "irregular_normal": "Irrégulier normal",
    "strict": "Seuils stricts",
    "moderate": "Seuils modérés",
    "loose": "Seuils larges",
}


# ── Chargement ───────────────────────────────────────────────────────────────

def _first_existing(*paths: str) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def resolve_data_dir() -> Path:
    candidates = [
        os.environ.get("CIB_DATA_DIR", ""),
        "/kaggle/input/datasets/jijirh/data-v6",
        "/kaggle/working",
        str(Path(__file__).resolve().parents[2] / "cib_project_outputs" / "cib_experiment_outputs"),
        str(Path(__file__).resolve().parents[2] / "cib_experiment_outputs" / "cib_experiment_outputs"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and os.path.exists(os.path.join(c, "lift_summary_v6.csv")):
            return Path(c)
    return Path(candidates[-1])


@st.cache_data
def load_all(data_dir: str) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    out: dict[str, pd.DataFrame] = {}

    def _read(name: str) -> pd.DataFrame:
        p = d / name
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if col in {
                "rmse", "mae", "rmse_baseline", "rmse_with_externals",
                "rmse_lift_pct", "rmse_lift_pct_mean", "rmse_lift_pct_median",
                "activate_externals", "is_champion_category", "n_accounts",
            }:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "activate_externals" in df.columns:
            df["activate_externals"] = df["activate_externals"].map(
                lambda x: str(x).lower() in {"true", "1", "yes"}
            )
        return df

    out["lift"] = _read("lift_summary_v6.csv")
    policy_v7 = _read("policy_v7_encapsulated.csv")
    out["policy"] = policy_v7 if not policy_v7.empty else _read("policy_v6_final_3pct.csv")
    out["policy_version"] = "v7" if not policy_v7.empty else "v6"
    out["cat_bench"] = _read("category_benchmark_v6.csv")
    out["cat_by_cond"] = _read("category_benchmark_by_condition_v6.csv")
    out["raw"] = _read("experiment_results_v6.csv")
    return out


def exp_from_account(name: str) -> str:
    parts = str(name).split("_")
    if len(parts) >= 1 and parts[0].startswith("EXP") and parts[0][3:].isdigit():
        return parts[0]
    return "autre"


def cat_label(cat: str) -> str:
    return CAT_LABELS.get(cat, cat)


def cond_label(cond: str) -> str:
    return COND_LABELS.get(cond, cond)


# ── Agrégations ────────────────────────────────────────────────────────────────

def calibration_lift(lift: pd.DataFrame) -> pd.DataFrame:
    df = lift.copy()
    if "experiment_role" in df.columns:
        df = df[df["experiment_role"] == "calibration"]
    return df


def lift_by_category_model(lift: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    df = calibration_lift(lift)
    if df.empty or "account_category" not in df.columns:
        return pd.DataFrame()
    if models and "model_name" in df.columns:
        df = df[df["model_name"].isin(models)]
    agg = (
        df.groupby(["account_category", "model_name"], dropna=False)["rmse_lift_pct"]
        .agg(lift_mean="mean", lift_median="median", n="count")
        .reset_index()
    )
    return agg


def lift_by_exp_category(lift: pd.DataFrame, exp_id: str) -> pd.DataFrame:
    spec = EXP_SPECS.get(exp_id, {})
    conds = spec.get("conditions", [])
    df = lift.copy()
    if spec.get("role") == "validation" and "experiment_role" in df.columns:
        sub = df[df["experiment_role"] == "validation"]
    else:
        sub = calibration_lift(df)
    if not conds:
        return pd.DataFrame()
    sub = sub[sub["condition"].isin(conds)]
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["condition", "account_category"], dropna=False)["rmse_lift_pct"]
        .agg(lift_median="median", lift_mean="mean", n_accounts=("name", "nunique") if "name" in sub.columns else ("rmse_lift_pct", "count"))
        .reset_index()
    )


def rmse_by_category_model(lift: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    df = calibration_lift(lift)
    if models and "model_name" in df.columns:
        df = df[df["model_name"].isin(models)]
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby(["account_category", "model_name"], dropna=False)
        .agg(
            rmse_ext_mean=("rmse_with_externals", "mean"),
            rmse_ext_median=("rmse_with_externals", "median"),
            rmse_base_mean=("rmse_baseline", "mean"),
        )
        .reset_index()
    )


def exp_category_matrix(lift: pd.DataFrame) -> pd.DataFrame:
    """Matrice EXP × catégorie (lift médian)."""
    df = lift.copy()
    if df.empty or "account_category" not in df.columns:
        return pd.DataFrame()
    rows = []
    for exp_id, spec in EXP_SPECS.items():
        conds = spec["conditions"]
        sub = df[df["condition"].isin(conds)]
        if sub.empty:
            continue
        g = (
            sub.groupby("account_category")["rmse_lift_pct"]
            .median()
            .reset_index()
        )
        g["exp"] = exp_id
        rows.append(g)
    if not rows:
        return pd.DataFrame()
    mat = pd.concat(rows, ignore_index=True)
    return mat.pivot(index="exp", columns="account_category", values="rmse_lift_pct")


# ── Graphiques ───────────────────────────────────────────────────────────────

def champion_bar(rmse_df: pd.DataFrame, category: str, champion: str) -> go.Figure:
    sub = rmse_df[rmse_df["account_category"] == category].copy()
    sub = sub.sort_values("rmse_ext_median")
    colors = [C_CHAMP if m == champion else C_EXT for m in sub["model_name"]]
    fig = go.Figure(go.Bar(
        x=sub["model_name"],
        y=sub["rmse_ext_median"],
        marker_color=colors,
        text=[f"{v:,.0f}" for v in sub["rmse_ext_median"]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"RMSE médiane (avec externes) — {cat_label(category)}",
        yaxis_title="RMSE",
        height=320,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def lift_activation_bar(lift_cat: pd.DataFrame, category: str, activate: bool) -> go.Figure:
    sub = lift_cat[lift_cat["account_category"] == category].copy()
    sub = sub.sort_values("lift_median", ascending=False)
    colors = [C_POS if v >= LIFT_THRESHOLD else (C_NEG if v < 0 else C_NEU) for v in sub["lift_median"]]
    fig = go.Figure(go.Bar(
        x=sub["model_name"],
        y=sub["lift_median"],
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in sub["lift_median"]],
        textposition="outside",
    ))
    fig.add_hline(y=LIFT_THRESHOLD, line_dash="dash", line_color=C_POS,
                  annotation_text=f"Seuil {LIFT_THRESHOLD:.0f} %")
    fig.add_hline(y=0, line_dash="dot", line_color=C_BASE)
    decision = "EXTERNES ON" if activate else "EXTERNES OFF"
    fig.update_layout(
        title=f"Lift RMSE médian — {cat_label(category)} → {decision}",
        yaxis_title="Lift RMSE (%)",
        height=320,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def exp_category_chart(exp_cat: pd.DataFrame, exp_id: str) -> go.Figure | None:
    if exp_cat.empty:
        return None
    exp_cat = exp_cat.copy()
    exp_cat["cat_label"] = exp_cat["account_category"].map(cat_label)
    exp_cat["cond_label"] = exp_cat["condition"].map(cond_label)
    fig = px.bar(
        exp_cat,
        x="lift_median",
        y="cat_label",
        color="cond_label",
        orientation="h",
        barmode="group",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"lift_median": "Lift RMSE médian (%)", "cat_label": "Catégorie", "cond_label": "Condition"},
        title=f"{EXP_SPECS[exp_id]['label']} — lift par catégorie",
    )
    fig.add_vline(x=0, line_dash="dot", line_color=C_BASE)
    fig.add_vline(x=LIFT_THRESHOLD, line_dash="dash", line_color=C_POS)
    fig.update_layout(height=max(280, 55 * exp_cat["cat_label"].nunique()), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def policy_summary_table(policy: pd.DataFrame) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame()
    t = policy.copy()
    champ_col = "champion_model" if "champion_model" in t.columns else "regression_model"
    lift_col = (
        "rmse_lift_pct_median"
        if "rmse_lift_pct_median" in t.columns
        else "lift_med_champion_pct"
    )
    n_col = "n_accounts" if "n_accounts" in t.columns else "n_accounts_v6"
    t["Catégorie"] = t["account_category"].map(cat_label)
    t["Externes"] = t["activate_externals"].map(lambda x: "✅ Oui" if x else "❌ Non")
    t["Champion"] = t[champ_col].astype(str).str.upper()
    t["Lift méd. (%)"] = t[lift_col].map(lambda x: f"{x:+.2f}" if pd.notna(x) else "—")
    t["n comptes"] = t[n_col].astype("Int64") if n_col in t.columns else pd.NA
    cols = ["Catégorie", "Externes", "Champion", "Lift méd. (%)", "n comptes"]
    if "source_exp" in t.columns:
        t["EXP source"] = t["source_exp"]
        cols.append("EXP source")
    return t[cols]


def insight_box(text: str) -> None:
    safe = text.replace("_", "&#95;").replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f'<div style="border-left:3px solid #2563eb;background:#f0f6ff;padding:12px 16px;'
        f'border-radius:0 8px 8px 0;font-size:13px;line-height:1.7;color:#1e293b;margin:10px 0;">'
        f"{safe}</div>",
        unsafe_allow_html=True,
    )


# ── UI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="CIB — Politique par catégorie", page_icon="🏦", layout="wide")

    st.markdown(
        """
        <div style="background:linear-gradient(135deg,#0f172a,#1e3a5f);padding:24px 28px;border-radius:12px;margin-bottom:16px;">
          <div style="font-size:11px;letter-spacing:2px;color:#64748b;text-transform:uppercase;">CIB · PFE · v4 · politique v7</div>
          <div style="font-size:22px;font-weight:600;color:#f8fafc;">Politique encapsulée — EXP → catégorie → règle</div>
          <div style="font-size:13px;color:#94a3b8;margin-top:4px;">
            Catégories = décision prod · Expériences = preuves &amp; analyse
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_dir = resolve_data_dir()
    data = load_all(str(data_dir))

    if data["lift"].empty:
        st.error(f"Fichier `lift_summary_v6.csv` introuvable dans `{data_dir}`. Lancez d'abord le notebook v6.")
        st.stop()

    lift = data["lift"]
    policy = data["policy"]
    policy_version = data.get("policy_version", "v6")
    cat_bench = data["cat_bench"]

    lift_cat_ml = lift_by_category_model(lift, ML_MODELS)
    rmse_cat_ml = rmse_by_category_model(lift, ML_MODELS)
    exp_matrix = exp_category_matrix(lift)

    cal = calibration_lift(lift)
    n_cal = cal["name"].nunique() if "name" in cal.columns else 0
    n_cat = cal["account_category"].nunique() if "account_category" in cal.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Comptes calibration", f"{n_cal:,}")
    c2.metric("Catégories observées", n_cat)
    c3.metric("Seuil activation", f"{LIFT_THRESHOLD:.0f} %")
    c4.metric("Politique", policy_version.upper())

    st.sidebar.title("Filtres")
    all_cats = sorted(cal["account_category"].dropna().unique().tolist())
    sel_cat = st.sidebar.selectbox(
        "Catégorie (drill-down)",
        options=all_cats,
        format_func=cat_label,
        index=0,
    )
    sel_model = st.sidebar.selectbox("Modèle ML (onglet EXP)", ML_MODELS, index=ML_MODELS.index("ridge"))

    tabs = st.tabs([
        "📋 Politique finale",
        "🏆 Champion par catégorie",
        "🔬 Par expérience",
        "🔗 EXP → Catégorie → Règle",
        "✅ Validation EXP5/7",
    ])

    # ── Onglet 1 : Politique ─────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Table de politique — une règle par catégorie")
        st.caption(
            "Décision agrégée sur **toutes** les EXP de calibration : "
            "chaque compte est catégorisé (bloc B), puis on regroupe par catégorie."
        )
        if policy.empty:
            st.warning("`policy_v7_encapsulated.csv` (ou v6) absent — politique non exportée.")
        else:
            st.dataframe(policy_summary_table(policy), use_container_width=True, hide_index=True)

            for _, row in policy.iterrows():
                cat = row["account_category"]
                champ = str(row.get("champion_model", row.get("regression_model", ""))).lower()
                act = bool(row.get("activate_externals", False))
                lift_med = row.get(
                    "rmse_lift_pct_median", row.get("lift_med_champion_pct", np.nan)
                )
                n = int(row.get("n_accounts", row.get("n_accounts_v6", 0)))

                col_a, col_b = st.columns(2)
                with col_a:
                    if not rmse_cat_ml.empty:
                        st.plotly_chart(champion_bar(rmse_cat_ml, cat, champ), use_container_width=True)
                with col_b:
                    if not lift_cat_ml.empty:
                        st.plotly_chart(lift_activation_bar(lift_cat_ml, cat, act), use_container_width=True)

                src = row.get("source_exp", CAT_POLICY_SOURCE.get(cat, "—"))
                reason = (
                    f"**{cat_label(cat)}** — {n} comptes · source **{src}**\n\n"
                    f"• **Décision propre** : {champ.upper()} + externes {'ON' if act else 'OFF'}\n"
                    f"• **Champion** : RMSE médiane la plus basse sur l'EXP source\n"
                    f"• **Externes : {'OUI' if act else 'NON'}** — lift médian {lift_med:+.2f} % "
                    f"({'≥' if act else '<'} seuil {LIFT_THRESHOLD:.0f} %)"
                )
                insight_box(reason)

    # ── Onglet 2 : Champion ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader(f"Justification du champion — {cat_label(sel_cat)}")
        if policy.empty:
            pol_row = {}
        else:
            pr = policy[policy["account_category"] == sel_cat]
            pol_row = pr.iloc[0].to_dict() if not pr.empty else {}

        champ = str(pol_row.get("champion_model", pol_row.get("regression_model", "ridge"))).lower()
        act = bool(pol_row.get("activate_externals", False))
        lift_pol = pol_row.get("rmse_lift_pct_median", pol_row.get("lift_med_champion_pct", np.nan))

        col1, col2, col3 = st.columns(3)
        col1.metric("Champion retenu", champ.upper())
        col2.metric("Activer externes", "Oui" if act else "Non")
        col3.metric("Lift médian politique", f"{lift_pol:+.2f} %" if pd.notna(lift_pol) else "—")

        if not rmse_cat_ml.empty:
            st.plotly_chart(champion_bar(rmse_cat_ml, sel_cat, champ), use_container_width=True)
        if not lift_cat_ml.empty:
            st.plotly_chart(lift_activation_bar(lift_cat_ml, sel_cat, act), use_container_width=True)

        # Contribution par EXP
        if "name" in cal.columns:
            sub = cal[cal["account_category"] == sel_cat].copy()
            sub["exp"] = sub["name"].map(exp_from_account)
            contrib = sub.groupby("exp")["name"].nunique().reset_index(name="n_comptes")
            contrib = contrib.sort_values("n_comptes", ascending=True)
            fig_c = go.Figure(go.Bar(
                x=contrib["n_comptes"],
                y=contrib["exp"],
                orientation="h",
                marker_color=C_EXT,
                text=contrib["n_comptes"],
                textposition="outside",
            ))
            fig_c.update_layout(
                title=f"Quelles EXP alimentent la catégorie {cat_label(sel_cat)} ?",
                xaxis_title="Nombre de comptes",
                height=280,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_c, use_container_width=True)
            insight_box(
                f"Les comptes de {cat_label(sel_cat)} viennent de plusieurs EXP "
                f"(EXP1, EXP3, EXP6…). La politique **ignore** l'EXP d'origine "
                f"et ne retient que la catégorie + les performances agrégées."
            )

        if not cat_bench.empty:
            section = cat_bench[
                (cat_bench["account_category"] == sel_cat) & (cat_bench["model_name"].isin(ML_MODELS))
            ].sort_values("rmse_with_externals_median")
            st.markdown("**Benchmark complet (tous modèles, calibration)**")
            st.dataframe(
                section[[
                    "model_name", "n_accounts",
                    "rmse_with_externals_median", "rmse_lift_pct_median", "is_champion_category",
                ]].rename(columns={
                    "model_name": "Modèle",
                    "n_accounts": "n",
                    "rmse_with_externals_median": "RMSE ext. méd.",
                    "rmse_lift_pct_median": "Lift méd. %",
                    "is_champion_category": "Champion (RMSE moy.)",
                }),
                use_container_width=True,
                hide_index=True,
            )

    # ── Onglet 3 : Par expérience ────────────────────────────────────────
    with tabs[2]:
        exp_ids = list(EXP_SPECS.keys())
        sel_exp = st.selectbox("Expérience", exp_ids, format_func=lambda x: EXP_SPECS[x]["label"])

        spec = EXP_SPECS[sel_exp]
        st.markdown(f"**Question :** {spec['question']}")
        st.markdown(f"**Rôle :** `{spec['role']}` — {spec['policy_impact']}")

        sub_lift = lift.copy()
        if spec["role"] == "validation" and "experiment_role" in sub_lift.columns:
            sub_lift = sub_lift[sub_lift["experiment_role"] == "validation"]
        else:
            sub_lift = calibration_lift(sub_lift)
        sub_lift = sub_lift[sub_lift["condition"].isin(spec["conditions"])]
        if sel_model and "model_name" in sub_lift.columns:
            sub_lift = sub_lift[sub_lift["model_name"] == sel_model]

        exp_cat = lift_by_exp_category(lift, sel_exp)
        if sel_model and "model_name" in lift.columns:
            sub_m = lift.copy()
            if spec["role"] == "validation" and "experiment_role" in sub_m.columns:
                sub_m = sub_m[sub_m["experiment_role"] == "validation"]
            else:
                sub_m = calibration_lift(sub_m)
            sub_m = sub_m[
                sub_m["condition"].isin(spec["conditions"]) & (sub_m["model_name"] == sel_model)
            ]
            if not sub_m.empty:
                exp_cat = (
                    sub_m.groupby(["condition", "account_category"])["rmse_lift_pct"]
                    .agg(lift_median="median", lift_mean="mean", n_accounts=("name", "nunique"))
                    .reset_index()
                )

        col_a, col_b = st.columns(2)
        with col_a:
            fig = exp_category_chart(exp_cat, sel_exp)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Pas de données pour cette EXP.")
        with col_b:
            if not exp_cat.empty:
                heat = exp_cat.pivot_table(
                    index="account_category", columns="condition", values="lift_median", aggfunc="mean"
                )
                heat.index = [cat_label(c) for c in heat.index]
                heat.columns = [cond_label(c) for c in heat.columns]
                fig_h = px.imshow(
                    heat,
                    text_auto=".1f",
                    color_continuous_scale=[[0, C_NEG], [0.5, "#F8FAFC"], [1, C_POS]],
                    color_continuous_midpoint=0,
                    labels=dict(color="Lift méd. %"),
                    title=f"Heatmap catégorie × condition ({sel_model})",
                )
                fig_h.update_layout(height=320)
                st.plotly_chart(fig_h, use_container_width=True)

        if not sub_lift.empty and "account_category" in sub_lift.columns:
            agg_cond = (
                sub_lift.groupby("account_category")["rmse_lift_pct"]
                .median()
                .sort_values(ascending=False)
            )
            impact_lines = []
            for cat, val in agg_cond.items():
                if pd.isna(val):
                    continue
                if spec["role"] == "validation":
                    impact_lines.append(f"• {cat_label(cat)} : lift médian {val:+.1f} % (validation)")
                elif val >= LIFT_THRESHOLD:
                    impact_lines.append(
                        f"• {cat_label(cat)} : {val:+.1f} % ≥ seuil → renforce activation externes"
                    )
                else:
                    impact_lines.append(
                        f"• {cat_label(cat)} : {val:+.1f} % < seuil → confirme externes OFF"
                    )
            insight_box(
                f"**Impact de {sel_exp} sur la politique**\n\n"
                + "\n".join(impact_lines)
                + f"\n\n{spec['policy_impact']}"
            )

    # ── Onglet 4 : Pont EXP → Catégorie ──────────────────────────────────
    with tabs[3]:
        st.subheader("Matrice EXP × catégorie — lift RMSE médian (%)")
        st.caption(
            "Chaque cellule = lift médian des comptes de cette catégorie, "
            "créés dans cette EXP. La politique finale agrège **toutes les lignes d'une colonne**."
        )
        if exp_matrix.empty:
            st.warning("Matrice vide.")
        else:
            mat = exp_matrix.copy()
            mat.index = [EXP_SPECS.get(e, {}).get("label", e) for e in mat.index]
            mat.columns = [cat_label(c) for c in mat.columns]
            fig_m = px.imshow(
                mat,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale=[[0, C_NEG], [0.5, "#F8FAFC"], [1, C_POS]],
                color_continuous_midpoint=0,
                labels=dict(color="Lift méd. %", x="Catégorie", y="Expérience"),
            )
            fig_m.update_layout(height=480)
            st.plotly_chart(fig_m, use_container_width=True)

        if not policy.empty:
            st.markdown("**Synthèse : comment chaque EXP nourrit la politique**")
            for exp_id, spec in EXP_SPECS.items():
                role_badge = "🧪 calibration" if spec["role"] == "calibration" else "✅ validation"
                st.markdown(
                    f"**{spec['label']}** `{role_badge}`  \n"
                    f"{spec['policy_impact']}"
                )

    # ── Onglet 5 : Validation ────────────────────────────────────────────
    with tabs[4]:
        st.subheader("EXP5 & EXP7 — garde-fous (hors agrégation politique)")
        for exp_id in ("EXP5", "EXP7"):
            spec = EXP_SPECS[exp_id]
            st.markdown(f"### {spec['label']}")
            st.caption(spec["question"])

            sub = lift[lift["condition"].isin(spec["conditions"])].copy()
            if "experiment_role" in sub.columns:
                sub = sub[sub["experiment_role"] == "validation"]
            if "model_name" in sub.columns:
                sub = sub[sub["model_name"] == sel_model]

            if sub.empty:
                st.warning("Données absentes.")
                continue

            agg = (
                sub.groupby("condition")["rmse_lift_pct"]
                .agg(median="median", mean="mean")
                .reindex(spec["conditions"])
                .dropna(how="all")
            )
            labels = [cond_label(c) for c in agg.index]
            colors = [C_POS if v > 1 else (C_NEG if v < -1 else C_NEU) for v in agg["median"]]
            fig = go.Figure(go.Bar(
                x=labels, y=agg["median"],
                marker_color=colors,
                text=[f"{v:+.1f}%" for v in agg["median"]],
                textposition="outside",
            ))
            fig.add_hline(y=0, line_dash="dash", line_color=C_BASE)
            fig.update_layout(
                title=f"Lift médian par condition ({sel_model})",
                yaxis_title="Lift RMSE (%)",
                height=300,
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)
            insight_box(spec["policy_impact"])


if __name__ == "__main__":
    main()

import warnings
warnings.filterwarnings("ignore")

import io
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score

import umap
import hdbscan
import shap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ──────────────────────────────────────────────────────────────
#  CONFIGURATION PAGE
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alerte Fournisseurs Maroc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
#  CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
.header-box {
    background: linear-gradient(135deg, #1F3864, #2E5496);
    padding: 20px 28px; border-radius: 10px; margin-bottom: 20px;
}
.header-box h1 { color: white; font-size: 1.7rem; margin: 0; }
.header-box p  { color: #BDD7EE; font-size: 0.9rem; margin: 4px 0 0; }
.kpi-card {
    border: 1px solid #e0e0e0; border-radius: 8px;
    padding: 16px 10px; text-align: center;
    background: white; box-shadow: 0 2px 6px rgba(0,0,0,.07);
}
.kpi-val { font-size: 2.2rem; font-weight: 700; color: #1F3864; }
.kpi-lab { font-size: 0.8rem; color: #595959; margin-top: 4px; }
.alerte-rouge  { background:#ffcccc; border-left:5px solid #C00000;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.alerte-orange { background:#ffe0b2; border-left:5px solid #C55A11;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.alerte-vert   { background:#ccffcc; border-left:5px solid #375623;
                  padding:12px 18px; border-radius:6px; font-weight:700; margin:8px 0; }
.shap-bar {
    display:inline-block; height:11px; border-radius:3px;
    vertical-align:middle; margin-left:8px;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  CONSTANTES
# ──────────────────────────────────────────────────────────────
EXCLURE = [
    "ID_Fournisseur", "Nom_Fournisseur", "Secteur", "Region_Maroc",
    "Cluster_Reel", "Note_Risque_Pays", "Certification",
    "Niveau_Alerte", "Priorite_Action", "Score_Risque",
    "Alerte_ML", "Priorite_ML", "Cluster_HDBSCAN",
]
COULEURS = {"🟢 Vert": "#4CAF50", "🟠 Orange": "#FF9800", "🔴 Rouge": "#F44336"}

# ──────────────────────────────────────────────────────────────
#  BARRE LATERALE
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    st.markdown("---")
    st.markdown("### UMAP")
    p_neighbors   = st.slider("n_neighbors",         5,  50, 15)
    p_min_dist    = st.slider("min_dist",           0.0, 0.5, 0.1, 0.05)
    st.markdown("### HDBSCAN")
    p_min_cluster = st.slider("min_cluster_size",   3,  30, 10)
    p_min_samples = st.slider("min_samples",        1,  15,  5)
    st.markdown("### Isolation Forest")
    p_contam      = st.slider("Contamination (%)",  1,  20,  5) / 100
    st.markdown("### Poids score composite")
    p_w_cl = st.slider("Poids Cluster (%)",         10, 60, 35) / 100
    p_w_an = st.slider("Poids Anomalie (%)",        10, 60, 30) / 100
    p_w_te = st.slider("Poids Temporel (%)",         5, 40, 20) / 100
    p_w_dt = max(0.0, 1.0 - p_w_cl - p_w_an - p_w_te)
    st.info(f"Poids DTW (auto) : {p_w_dt*100:.0f}%")
    st.markdown("### SHAP")
    p_shap     = st.checkbox("Calculer SHAP", value=True)
    p_shap_top = st.slider("Top N variables", 5, 30, 15)
    st.markdown("### Seuils d'alerte")
    p_seuil_vert   = st.slider("Seuil Vert / Orange",  10, 45, 29)
    p_seuil_orange = st.slider("Seuil Orange / Rouge", 40, 80, 59)
    st.markdown("---")
    st.caption("Université Mohammed V — Rabat\nMaster ML & Intelligence Logistique\n2024–2025")

# ──────────────────────────────────────────────────────────────
#  EN-TETE
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🔍 Système d'Alerte Précoce — Risque Fournisseurs Maroc</h1>
  <p>Pipeline ML : UMAP → HDBSCAN → Isolation Forest → VAE → SHAP</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
#  UPLOAD DU FICHIER
# ──────────────────────────────────────────────────────────────
st.markdown("### 📁 Étape 1 — Chargez votre dataset")

fichier = st.file_uploader(
    "Importez votre fichier Excel (.xlsx) ou CSV (.csv)",
    type=["xlsx", "csv"],
)

if fichier is None:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.info("**Étape 1** — Chargez votre dataset ci-dessus")
    c2.info("**Étape 2** — Ajustez les paramètres (barre gauche)")
    c3.info("**Étape 3** — Cliquez sur Lancer l'analyse")
    st.stop()

@st.cache_data(show_spinner=False)
def lire(contenu, nom):
    if nom.endswith(".xlsx"):
        return pd.read_excel(io.BytesIO(contenu))
    return pd.read_csv(io.BytesIO(contenu))

with st.spinner("Lecture du fichier…"):
    df = lire(fichier.read(), fichier.name)

st.success(f"✅  {fichier.name} — {df.shape[0]} fournisseurs × {df.shape[1]} colonnes")

with st.expander("👁️  Aperçu (5 premières lignes)"):
    st.dataframe(df.head(), use_container_width=True)

features = [c for c in df.columns
            if c not in EXCLURE
            and df[c].dtype in ["float64", "int64", "float32", "int32"]]

with st.expander(f"🔢  {len(features)} variables numériques pour le ML"):
    cols = st.columns(4)
    for i, f in enumerate(features):
        cols[i % 4].markdown(f"- `{f}`")

X_brut = df[features].copy()
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  BOUTON LANCEMENT
# ──────────────────────────────────────────────────────────────
st.markdown("### 🚀 Étape 2 — Lancez l'analyse")

if not st.button("▶  Lancer l'analyse complète", type="primary", use_container_width=True):
    st.info("👆 Cliquez sur le bouton — durée estimée : 3 à 5 minutes")
    st.stop()

# ──────────────────────────────────────────────────────────────
#  PIPELINE ML
# ──────────────────────────────────────────────────────────────
barre   = st.progress(0, text="Démarrage…")
log     = st.empty()
t_debut = time.time()

def avancer(msg, pct, ok=False):
    ic = "✅" if ok else "⏳"
    barre.progress(pct, text=f"{ic}  {msg}")
    log.info(f"{ic}  {msg}")

# 1. Prétraitement
avancer("Étape 1/7 — Prétraitement", 5)
X_imp = SimpleImputer(strategy="median").fit_transform(X_brut)
X_sc  = RobustScaler().fit_transform(X_imp)
avancer("Prétraitement terminé", 12, ok=True)

# 2. PCA
avancer("Étape 2/7 — PCA", 15)
pca   = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_sc)
n_pca = pca.n_components_
avancer(f"PCA : {X_sc.shape[1]}D → {n_pca}D", 22, ok=True)

# 3. UMAP
avancer("Étape 3/7 — UMAP (30 à 40 secondes…)", 25)
X_u3 = umap.UMAP(n_components=3, n_neighbors=p_neighbors,
                   min_dist=p_min_dist, random_state=42).fit_transform(X_pca)
X_u2 = umap.UMAP(n_components=2, n_neighbors=p_neighbors,
                   min_dist=p_min_dist, random_state=42).fit_transform(X_pca)
avancer("UMAP terminé", 42, ok=True)

# 4. HDBSCAN
avancer("Étape 4/7 — HDBSCAN", 46)
cl     = hdbscan.HDBSCAN(min_cluster_size=p_min_cluster,
                           min_samples=p_min_samples,
                           metric="euclidean", prediction_data=True)
labels = cl.fit_predict(X_u3)
n_cl   = len(set(labels)) - (1 if -1 in labels else 0)
n_br   = int((labels == -1).sum())
mask   = labels != -1
sil    = silhouette_score(X_u3[mask], labels[mask]) if mask.sum() > 10 else 0.0
dbi    = davies_bouldin_score(X_u3[mask], labels[mask]) if mask.sum() > 10 else 9.9
avancer(f"HDBSCAN : {n_cl} clusters + {n_br} anomalies", 57, ok=True)

# 5. Isolation Forest
avancer("Étape 5/7 — Isolation Forest", 60)
iso    = IsolationForest(n_estimators=200, contamination=p_contam,
                          random_state=42, n_jobs=-1)
iso.fit(X_sc)
if_r   = iso.decision_function(X_sc)
if_sc  = 1 - (if_r - if_r.min()) / (if_r.max() - if_r.min())
n_anom = int((iso.predict(X_sc) == -1).sum())
avancer(f"Isolation Forest : {n_anom} anomalies", 68, ok=True)

# 6. VAE
avancer("Étape 6/7 — VAE", 70)

class VAE(nn.Module):
    def __init__(self, d, l=8):
        super().__init__()
        h = max(d // 2, l * 4)
        self.enc = nn.Sequential(
            nn.Linear(d, h), nn.BatchNorm1d(h), nn.LeakyReLU(0.1),
            nn.Dropout(0.2), nn.Linear(h, h // 2), nn.LeakyReLU(0.1))
        self.mu  = nn.Linear(h // 2, l)
        self.lv  = nn.Linear(h // 2, l)
        self.dec = nn.Sequential(
            nn.Linear(l, h // 2), nn.LeakyReLU(0.1),
            nn.Linear(h // 2, h), nn.BatchNorm1d(h),
            nn.LeakyReLU(0.1), nn.Linear(h, d))
    def forward(self, x):
        h = self.enc(x); mu, lv = self.mu(h), self.lv(h)
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
        return self.dec(z), mu, lv

Xn  = torch.FloatTensor(X_sc[labels != -1])
ld  = DataLoader(TensorDataset(Xn), batch_size=32, shuffle=True)
vae = VAE(X_sc.shape[1])
opt = optim.Adam(vae.parameters(), lr=1e-3)
vae.train()
for _ in range(60):
    for (b,) in ld:
        opt.zero_grad()
        r, mu, lv = vae(b)
        loss = nn.functional.mse_loss(r, b, reduction="sum") \
             - 5e-4 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
        loss.backward(); opt.step()

vae.eval()
with torch.no_grad():
    r_all, _, _ = vae(torch.FloatTensor(X_sc))
    ve = nn.functional.mse_loss(r_all, torch.FloatTensor(X_sc),
                                 reduction="none").mean(dim=1).numpy()
vae_sc = (ve - ve.min()) / (ve.max() - ve.min())
anom   = (if_sc + vae_sc) / 2
avancer("VAE terminé", 80, ok=True)

# 7. Score composite
avancer("Étape 7/7 — Score composite", 84)

cr = {c: float(if_sc[labels == c].mean()) if c != -1 else 1.0 for c in set(labels)}
cc = np.array([cr[c] for c in labels])

def gcol(n):
    return df[n].fillna(0).astype(float).values if n in df.columns else np.zeros(len(df))

derive = (0.4 * np.clip(gcol("PSI_Score") / 0.5, 0, 1)
        + 0.4 * gcol("Changepoint_PELT")
        + 0.2 * np.clip(np.abs(gcol("Tendance_OTD_6M")) / 10, 0, 1))

score_100 = np.clip(
    (p_w_cl * cc + p_w_an * anom + p_w_te * derive + p_w_dt * vae_sc) * 100,
    0, 100)

def alerte(s):
    if s <= p_seuil_vert:   return "🟢 Vert"
    if s <= p_seuil_orange: return "🟠 Orange"
    return "🔴 Rouge"

alertes  = np.array([alerte(s) for s in score_100])
n_vert   = int((alertes == "🟢 Vert").sum())
n_orange = int((alertes == "🟠 Orange").sum())
n_rouge  = int((alertes == "🔴 Rouge").sum())

df_res = df.copy()
df_res["Cluster_HDBSCAN"] = labels
df_res["Score_IF"]        = np.round(if_sc * 100, 1)
df_res["Score_VAE"]       = np.round(vae_sc * 100, 1)
df_res["Score_Anomalie"]  = np.round(anom * 100, 1)
df_res["Score_Risque_ML"] = np.round(score_100, 1)
df_res["Alerte_ML"]       = alertes
df_res["Priorite_ML"]     = [
    "IMMÉDIAT" if a == "🔴 Rouge" else "SURVEILLANCE" if a == "🟠 Orange" else "STANDARD"
    for a in alertes]

t_total = time.time() - t_debut
avancer(f"Pipeline complet en {t_total:.0f} secondes", 90, ok=True)

# SHAP
shap_values = None
shap_df     = None
if p_shap:
    avancer("Calcul SHAP…", 93)
    try:
        exp         = shap.TreeExplainer(iso)
        shap_values = exp.shap_values(X_sc)
        ma          = np.abs(shap_values).mean(axis=0)
        shap_df     = pd.DataFrame({
            "Variable": features,
            "SHAP_abs": ma,
            "SHAP_pct": (ma / ma.sum() * 100).round(1),
        }).sort_values("SHAP_abs", ascending=False).reset_index(drop=True)
        df_res["Top_Variable_SHAP"] = [
            features[int(np.argmax(np.abs(shap_values[i])))] for i in range(len(df))]
        avancer("SHAP terminé", 98, ok=True)
    except Exception as e:
        st.warning(f"SHAP non disponible : {e}")

barre.progress(100, text="✅  Analyse terminée !")
log.success(f"✅  {len(df)} fournisseurs analysés en {t_total:.0f} secondes")
st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  CARTES KPI
# ──────────────────────────────────────────────────────────────
st.markdown("### 📊 Résultats")

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.markdown(f'<div class="kpi-card"><div class="kpi-val">{len(df)}</div><div class="kpi-lab">Fournisseurs</div></div>', unsafe_allow_html=True)
k2.markdown(f'<div class="kpi-card"><div class="kpi-val">{n_cl}</div><div class="kpi-lab">Clusters ML</div></div>', unsafe_allow_html=True)
k3.markdown(f'<div class="kpi-card" style="border-color:#4CAF50"><div class="kpi-val" style="color:#375623">{n_vert}</div><div class="kpi-lab">🟢 Vert</div></div>', unsafe_allow_html=True)
k4.markdown(f'<div class="kpi-card" style="border-color:#FF9800"><div class="kpi-val" style="color:#C55A11">{n_orange}</div><div class="kpi-lab">🟠 Orange</div></div>', unsafe_allow_html=True)
k5.markdown(f'<div class="kpi-card" style="border-color:#C00000"><div class="kpi-val" style="color:#C00000">{n_rouge}</div><div class="kpi-lab">🔴 Rouge</div></div>', unsafe_allow_html=True)
k6.markdown(f'<div class="kpi-card"><div class="kpi-val">{score_100.mean():.1f}</div><div class="kpi-lab">Score moyen</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📏  Métriques de validation ML"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Silhouette",     f"{sil:.3f}", delta="✅ > 0.50" if sil > 0.50 else "⚠️")
    m2.metric("Davies-Bouldin", f"{dbi:.3f}", delta="✅ < 1.50" if dbi < 1.50 else "⚠️")
    m3.metric("Anomalies IF",   n_anom)
    m4.metric("Anomalies HDBSCAN", n_br)

st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  ONGLETS
# ──────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "📌 Clusters UMAP",
    "📊 Distribution",
    "🔬 Anomalies",
    "🧠 SHAP",
    "📋 Tableau",
])

dp = df.copy()
dp["UMAP1"]   = X_u2[:, 0]
dp["UMAP2"]   = X_u2[:, 1]
dp["Cluster"] = ["Anomalie" if c == -1 else f"Cluster {c}" for c in labels]
dp["Alerte"]  = alertes
dp["Score"]   = np.round(score_100, 1)
hv = [c for c in ["ID_Fournisseur", "Secteur", "Region_Maroc"] if c in dp.columns]

with t1:
    st.subheader("Projection UMAP 2D")
    choix = st.radio("Colorier par :", ["Alerte", "Cluster", "Score"], horizontal=True)
    if choix == "Alerte":
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Alerte",
                          color_discrete_map=COULEURS, hover_data=hv + ["Score"],
                          title="UMAP 2D — Niveaux d'alerte",
                          template="plotly_white", height=520)
    elif choix == "Cluster":
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Cluster",
                          hover_data=hv + ["Score"],
                          title="UMAP 2D — Clusters HDBSCAN",
                          template="plotly_white", height=520)
    else:
        fig = px.scatter(dp, x="UMAP1", y="UMAP2", color="Score",
                          color_continuous_scale="RdYlGn_r", hover_data=hv,
                          title="UMAP 2D — Score de risque",
                          template="plotly_white", height=520)
    fig.update_traces(marker=dict(size=8, opacity=0.82, line=dict(width=0.5, color="white")))
    fig.update_layout(font=dict(family="Arial", size=11), title_font=dict(size=14, color="#1F3864"))
    st.plotly_chart(fig, use_container_width=True)

with t2:
    st.subheader("Distribution des alertes")
    ca, cb = st.columns(2)
    fig_h = px.histogram(x=score_100, nbins=40, color_discrete_sequence=["#2E5496"],
                          labels={"x": "Score (0–100)", "count": "Fréquence"},
                          title="Distribution du score composite", template="plotly_white")
    fig_h.add_vline(x=p_seuil_vert,    line_dash="dash", line_color="#4CAF50",
                     annotation_text=f"Vert ({p_seuil_vert})")
    fig_h.add_vline(x=p_seuil_orange,  line_dash="dash", line_color="#FF9800",
                     annotation_text=f"Orange ({p_seuil_orange})")
    fig_h.add_vline(x=score_100.mean(), line_dash="dot",  line_color="#C00000",
                     annotation_text=f"Moy {score_100.mean():.1f}")
    ca.plotly_chart(fig_h, use_container_width=True)
    fig_p = px.pie(names=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
                    values=[n_vert, n_orange, n_rouge],
                    color=["🟢 Vert", "🟠 Orange", "🔴 Rouge"],
                    color_discrete_map=COULEURS,
                    title="Répartition des alertes", hole=0.4)
    fig_p.update_traces(textinfo="label+percent+value", textposition="outside")
    cb.plotly_chart(fig_p, use_container_width=True)
    df_box = pd.DataFrame({
        "Cluster": ["Anomalie" if c == -1 else f"C{c}" for c in labels],
        "Score":   score_100})
    fig_b = px.box(df_box, x="Cluster", y="Score", color="Cluster",
                    title="Score par cluster", template="plotly_white",
                    height=400, points="outliers")
    fig_b.add_hline(y=p_seuil_vert,   line_dash="dash", line_color="#4CAF50")
    fig_b.add_hline(y=p_seuil_orange, line_dash="dash", line_color="#FF9800")
    fig_b.update_layout(showlegend=False, font=dict(family="Arial", size=11))
    st.plotly_chart(fig_b, use_container_width=True)

with t3:
    st.subheader("Scores d'anomalie")
    cc1, cc2 = st.columns(2)
    fig_if = px.scatter(x=X_u2[:, 0], y=X_u2[:, 1], color=if_sc * 100,
                         color_continuous_scale="RdYlGn_r",
                         labels={"x": "UMAP1", "y": "UMAP2", "color": "Score IF"},
                         title="Score Isolation Forest", template="plotly_white", height=420)
    fig_if.update_traces(marker=dict(size=7, opacity=0.8))
    cc1.plotly_chart(fig_if, use_container_width=True)
    fig_vae = px.scatter(x=X_u2[:, 0], y=X_u2[:, 1], color=vae_sc * 100,
                          color_continuous_scale="RdYlGn_r",
                          labels={"x": "UMAP1", "y": "UMAP2", "color": "Score VAE"},
                          title="Score VAE", template="plotly_white", height=420)
    fig_vae.update_traces(marker=dict(size=7, opacity=0.8))
    cc2.plotly_chart(fig_vae, use_container_width=True)
    fig_sc = px.scatter(x=if_sc * 100, y=vae_sc * 100, color=alertes,
                         color_discrete_map=COULEURS,
                         labels={"x": "Score IF (×100)", "y": "Score VAE (×100)"},
                         title="Corrélation IF vs VAE", template="plotly_white", height=420)
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=100, y1=100,
                      line=dict(dash="dash", color="gray", width=1))
    fig_sc.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=0.4, color="white")))
    st.plotly_chart(fig_sc, use_container_width=True)

with t4:
    st.subheader("Importance SHAP des variables")
    if shap_df is None:
        st.info("Cochez **Calculer SHAP** dans la barre latérale et relancez.")
    else:
        top = shap_df.head(p_shap_top)
        colors = ["#C00000" if v > 10 else "#C55A11" if v > 5 else "#2E5496"
                  for v in top["SHAP_pct"]]
        fig_s = go.Figure(go.Bar(
            x=top["SHAP_pct"], y=top["Variable"], orientation="h",
            marker_color=colors,
            text=[f"{v:.1f}%" for v in top["SHAP_pct"]], textposition="outside"))
        fig_s.update_layout(
            title=dict(text=f"Top {p_shap_top} variables", font=dict(size=14, color="#1F3864")),
            xaxis_title="Contribution (%)", yaxis=dict(autorange="reversed"),
            height=max(420, p_shap_top * 27), template="plotly_white",
            font=dict(family="Arial", size=11), margin=dict(l=10, r=80, t=50, b=40))
        st.plotly_chart(fig_s, use_container_width=True)
        st.dataframe(top[["Variable", "SHAP_pct"]].rename(
            columns={"SHAP_pct": "Contribution (%)"}),
            use_container_width=True, height=350)

with t5:
    st.subheader("Tableau des fournisseurs")
    f1, f2, f3 = st.columns(3)
    sel  = f1.multiselect("Alertes", ["🔴 Rouge", "🟠 Orange", "🟢 Vert"],
                            default=["🔴 Rouge", "🟠 Orange"])
    smin = f2.slider("Score min", 0, 100, 0)
    smax = f3.slider("Score max", 0, 100, 100)
    df_f = df_res[
        df_res["Alerte_ML"].isin(sel)
        & (df_res["Score_Risque_ML"] >= smin)
        & (df_res["Score_Risque_ML"] <= smax)
    ].sort_values("Score_Risque_ML", ascending=False)
    st.info(f"**{len(df_f)} fournisseurs** affichés sur {len(df_res)}")
    afficher = [c for c in [
        "ID_Fournisseur", "Secteur", "Region_Maroc",
        "Alerte_ML", "Score_Risque_ML", "Score_IF", "Score_VAE",
        "Cluster_HDBSCAN", "Priorite_ML", "Top_Variable_SHAP",
    ] if c in df_f.columns]
    st.dataframe(df_f[afficher], use_container_width=True, height=420)
    csv = df_res.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️  Télécharger les résultats (CSV)",
                        data=csv, file_name="resultats_alertes_fournisseurs.csv",
                        mime="text/csv", type="primary")

st.markdown("---")

# ──────────────────────────────────────────────────────────────
#  RAPPORT INDIVIDUEL
# ──────────────────────────────────────────────────────────────
st.markdown("### 📄 Rapport individuel fournisseur")

options = df_res["ID_Fournisseur"].tolist() if "ID_Fournisseur" in df_res.columns else list(df_res.index)
id_sel  = st.selectbox("Sélectionnez un fournisseur", options)
idx     = df_res[df_res["ID_Fournisseur"] == id_sel].index[0] if "ID_Fournisseur" in df_res.columns else id_sel
ligne   = df_res.loc[idx]
sc_v    = float(ligne["Score_Risque_ML"])
al_v    = str(ligne["Alerte_ML"])
fid     = ligne.get("ID_Fournisseur", f"Index {idx}")
fsect   = ligne.get("Secteur", "N/A")
freg    = ligne.get("Region_Maroc", "N/A")
fcl     = int(ligne["Cluster_HDBSCAN"])

css = "alerte-rouge" if "Rouge" in al_v else "alerte-orange" if "Orange" in al_v else "alerte-vert"
st.markdown(f'<div class="{css}">🏭 <b>{fid}</b> · {fsect} · {freg}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="{css}">Alerte : <b>{al_v}</b> · Score : <b>{sc_v:.1f}/100</b> · Cluster : <b>{"Anomalie" if fcl==-1 else f"C{fcl}"}</b></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
r1.metric("Score de risque", f"{sc_v:.1f}/100")
r2.metric("Score IF",        f"{ligne['Score_IF']:.1f}/100")
r3.metric("Score VAE",       f"{ligne['Score_VAE']:.1f}/100")
r4.metric("Score Anomalie",  f"{ligne['Score_Anomalie']:.1f}/100")

if shap_values is not None:
    st.markdown("#### 🧠 Facteurs de risque (SHAP)")
    pos  = df_res.index.get_loc(idx)
    sv   = shap_values[pos]
    top5 = np.argsort(np.abs(sv))[::-1][:5]
    tot  = np.abs(sv).sum() + 1e-10
    rows = ""
    for rk, j in enumerate(top5, 1):
        ct  = abs(sv[j]) / tot * 100
        di  = "↑ Aggrave" if sv[j] > 0 else "↓ Réduit"
        dc  = "#C00000" if sv[j] > 0 else "#375623"
        vn  = features[j]
        try:    vb = round(float(df.loc[idx, vn]), 3)
        except: vb = "N/A"
        bw  = int(ct * 3)
        bc  = "#C00000" if sv[j] > 0 else "#375623"
        bg  = "#FFF8F8" if sv[j] > 0 else "#F8FFF8"
        rows += (
            f"<tr style='background:{bg};border-bottom:1px solid #eee'>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{rk}</td>"
            f"<td style='padding:8px;font-weight:700;color:#1F3864'>{vn}</td>"
            f"<td style='padding:8px;text-align:center'><code>{vb}</code></td>"
            f"<td style='padding:8px;text-align:center;font-weight:700'>{ct:.1f}%</td>"
            f"<td style='padding:8px;color:{dc};font-weight:600'>{di}</td>"
            f"<td style='padding:8px'>"
            f"<span class='shap-bar' style='width:{bw}px;background:{bc}'></span></td></tr>")
    st.markdown(
        "<table style='width:100%;border-collapse:collapse;font-family:Arial;font-size:13px'>"
        "<tr style='background:#1F3864;color:white'>"
        "<th style='padding:8px'>Rang</th><th>Variable</th><th>Valeur</th>"
        "<th>Contribution</th><th>Direction</th><th>Barre</th></tr>"
        + rows + "</table>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

if "Rouge" in al_v:
    rec_css = "alerte-rouge"
    rec_txt = "🔴 <b>ACTION IMMÉDIATE</b> — Activer le plan de contingence dans les 2 semaines."
elif "Orange" in al_v:
    rec_css = "alerte-orange"
    rec_txt = "🟠 <b>SURVEILLANCE RENFORCÉE</b> — Planifier un audit dans le mois."
else:
    rec_css = "alerte-vert"
    rec_txt = "🟢 <b>SURVEILLANCE STANDARD</b> — Monitoring mensuel, aucune action immédiate."

st.markdown(f'<div class="{rec_css}"><b>Recommandation :</b> {rec_txt}</div>',
            unsafe_allow_html=True)

rv = [v for v in ["OTD_Pct", "Altman_ZScore", "Score_ESG", "Stabilite_Politique",
                   "Score_IF", "Current_Ratio", "Dependance_Mono"] if v in df_res.columns]
if len(rv) >= 4:
    with st.expander("📡  Profil radar"):
        vn = []
        for v in rv:
            d = df_res[v].dropna()
            vn.append(float(np.clip(
                (float(df_res.loc[idx, v]) - d.min()) / (d.max() - d.min() + 1e-10),
                0, 1) * 100))
        fig_r = go.Figure(go.Scatterpolar(
            r=vn + [vn[0]], theta=rv + [rv[0]], fill="toself",
            fillcolor="rgba(46,84,150,0.15)", line=dict(color="#2E5496", width=2)))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=dict(text=f"Profil — {fid}", font=dict(color="#1F3864", size=13)),
            height=420, showlegend=False, font=dict(family="Arial", size=10))
        st.plotly_chart(fig_r, use_container_width=True)

st.markdown("---")
st.caption("Université Mohammed V — Rabat · Master ML & Intelligence Logistique · 2024–2025")

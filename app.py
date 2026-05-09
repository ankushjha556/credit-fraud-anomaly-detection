import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib, gdown, json, os, io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class FraudAutoencoderV2(nn.Module):
    def __init__(self, input_dim=29, bottleneck=4, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),        nn.Dropout(dropout),
            nn.Linear(32, 16),        nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),        nn.Dropout(dropout),
            nn.Linear(16, 8),         nn.BatchNorm1d(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 8),  nn.BatchNorm1d(8),  nn.LeakyReLU(0.1),
            nn.Linear(8, 16),          nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Linear(16, 32),         nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )
    def forward(self, x): return self.decoder(self.encoder(x))

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #030712 !important;
    color: #f1f5f9 !important;
}
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"] { visibility: hidden !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1380px !important; }
[data-testid="stAppViewContainer"]::before {
    content: ''; position: fixed; inset: 0;
    background:
        radial-gradient(ellipse 900px 600px at 10% 5%,  rgba(239,68,68,0.06)  0%, transparent 70%),
        radial-gradient(ellipse 700px 500px at 90% 85%, rgba(245,158,11,0.05) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0a1628 0%, #06101e 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.25rem !important; }
[data-testid="stFileUploader"] section {
    background: rgba(239,68,68,0.03) !important;
    border: 1.5px dashed rgba(239,68,68,0.18) !important;
    border-radius: 16px !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(239,68,68,0.4) !important;
}
[data-testid="stFileUploader"] button {
    background: rgba(239,68,68,0.1) !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    color: #ef4444 !important;
    border-radius: 8px !important;
}
.stSpinner > div { border-top-color: #ef4444 !important; }
div[data-testid="stRadio"] label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ──────────────────────────────────────────────
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DRIVE_IDS = {
    "isolation_forest.pkl" : "14gStlAKjeW0RSBPzyzekEi-n76few9UF",
    "autoencoder_v2.pth"   : "14T8lpaBJhXxkCk_OvSYetBJ9C9MflLqV",
    "scaler.pkl"           : "1e26eBRe2R8i7lMgzh69g4Qjxi2A1op8g",
    "final_config.json"    : "1iSms28x7HVM5iEuXGBWG0tysYUPxN3f2",
}

@st.cache_resource
def load_all_models():
    for fname, fid in DRIVE_IDS.items():
        path = f"{MODELS_DIR}/{fname}"
        if not os.path.exists(path):
            with st.spinner(f"Downloading {fname}..."):
                gdown.download(
                    f"https://drive.google.com/uc?id={fid}",
                    path, quiet=True)
    with open(f"{MODELS_DIR}/final_config.json") as f:
        cfg = json.load(f)
    if_model = joblib.load(f"{MODELS_DIR}/isolation_forest.pkl")
    scaler   = joblib.load(f"{MODELS_DIR}/scaler.pkl")
    device   = torch.device("cpu")
    ae_model = FraudAutoencoderV2(
        29, int(cfg["ae_params"]["bottleneck"]),
        cfg["ae_params"]["dropout"])
    ae_model.load_state_dict(torch.load(
        f"{MODELS_DIR}/autoencoder_v2.pth", map_location=device))
    ae_model.eval()
    return if_model, ae_model, scaler, cfg, device

if_model, ae_model, scaler, cfg, device = load_all_models()

def normalize(s): return (s - s.min()) / (s.max() - s.min() + 1e-9)

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120,
                bbox_inches="tight", facecolor="#030712")
    return base64.b64encode(buf.getvalue()).decode()

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#ef4444,#f59e0b);
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem;">🛡️</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#f1f5f9;">FraudShield AI</div>
                <div style="font-size:0.62rem;color:#475569;font-family:'JetBrains Mono',monospace;">
                    ENSEMBLE DETECTOR · v1.0
                </div>
            </div>
        </div>
    </div>
    """ + "".join(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">{k}</span>
        <span style="font-size:0.75rem;font-weight:600;color:{c};font-family:'JetBrains Mono',monospace;">{v}</span>
    </div>""" for k, v, c in [
        ("Test ROC-AUC",  "0.9439",    "#ef4444"),
        ("Test PR-AUC",   "0.4555",    "#ef4444"),
        ("Normal Acc",    "99.89%",    "#10b981"),
        ("Fraud Recall",  "50.0%",     "#f59e0b"),
        ("False Alarms",  "0.11%",     "#10b981"),
        ("IF Configs",    "27 tested", "#7c3aed"),
        ("AE Configs",    "27 tested", "#7c3aed"),
        ("Total Expts",   "63",        "#7c3aed"),
        ("Training data", "199,134",   "#94a3b8"),
        ("Fraud labels",  "None ✅",   "#10b981"),
    ]) + """
    <div style="margin-top:1.4rem;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">⬡ Key Insight</div>
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.1);
                border-radius:12px;padding:1rem;font-size:0.73rem;color:#64748b;line-height:1.65;">
        Trained on <span style="color:#f1f5f9;font-weight:600;">normal transactions only</span>
        — no fraud labels used. V17 shows
        <span style="color:#ef4444;font-weight:600;">435× higher</span>
        reconstruction error in fraud vs normal.
    </div>
    """, unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────
# ── Hero ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:3.5rem 0 2rem;">
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.22);
                border-radius:50px;padding:6px 18px;margin-bottom:1.6rem;">
        <span style="width:7px;height:7px;border-radius:50%;background:#ef4444;
                     box-shadow:0 0 10px #ef4444;display:inline-block;"></span>
        <span style="font-family:'JetBrains Mono',monospace;font-size:0.66rem;
                     color:#ef4444;letter-spacing:0.12em;text-transform:uppercase;">
            Unsupervised Ensemble Detection
        </span>
    </div>
    <h1 style="font-size:clamp(2.4rem,4vw,4rem);font-weight:800;line-height:1.08;
               letter-spacing:-0.04em;margin:0 0 1rem;
               background:linear-gradient(135deg,#ffffff 0%,#fca5a5 40%,#fbbf24 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">
        Credit Card Fraud<br>Detection
    </h1>
    <p style="font-size:0.95rem;color:#64748b;max-width:500px;
              margin:0 auto 2.5rem;line-height:1.8;">
        Isolation Forest + Autoencoder ensemble trained on
        <span style="color:#ef4444;font-weight:600;">284,807 transactions</span>.
        Zero fraud labels used. Achieves
        <span style="color:#f59e0b;font-weight:600;">ROC-AUC 0.9439</span>.
    </p>
    <div style="display:inline-grid;grid-template-columns:repeat(4,1fr);
                background:rgba(255,255,255,0.02);
                border:1px solid rgba(255,255,255,0.07);
                border-radius:18px;overflow:hidden;max-width:680px;width:100%;">
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#ef4444;
                        font-family:'JetBrains Mono',monospace;">0.9439</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">ROC-AUC</div>
        </div>
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#f59e0b;
                        font-family:'JetBrains Mono',monospace;">0.4555</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">PR-AUC</div>
        </div>
        <div style="padding:1.1rem 1rem;border-right:1px solid rgba(255,255,255,0.07);">
            <div style="font-size:1.5rem;font-weight:700;color:#10b981;
                        font-family:'JetBrains Mono',monospace;">99.89%</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">Normal Acc</div>
        </div>
        <div style="padding:1.1rem 1rem;">
            <div style="font-size:1.5rem;font-weight:700;color:#7c3aed;
                        font-family:'JetBrains Mono',monospace;">63</div>
            <div style="font-size:0.6rem;color:#334155;text-transform:uppercase;
                        letter-spacing:0.1em;margin-top:5px;
                        font-family:'JetBrains Mono',monospace;">Experiments</div>
        </div>
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,transparent,
    rgba(239,68,68,0.2),rgba(245,158,11,0.2),transparent);
    margin:0 0 2rem;"></div>
""", unsafe_allow_html=True)

# ── Mode Toggle ────────────────────────────────────────────────
mode = st.radio("Select input mode",
                ["📋 Manual Input", "📁 Upload CSV"],
                horizontal=True,
                label_visibility="collapsed")

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ── Manual Input ───────────────────────────────────────────────
if mode == "📋 Manual Input":
    st.markdown("""
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                border-radius:14px;padding:1rem 1.4rem 0.5rem;margin-bottom:1rem;">
        <div style="font-size:0.62rem;font-weight:600;color:#64748b;text-transform:uppercase;
                    letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;">
            Transaction Features — V1 to V28 are PCA-transformed by the bank
        </div>
    </div>
    """, unsafe_allow_html=True)

    preset = st.selectbox("Load a preset transaction:", [
        "Custom",
        "Typical Normal ($22)",
        "High Value Normal ($500)",
        "Suspicious Pattern",
        "Known Fraud Pattern"
    ])

    presets = {
        "Typical Normal ($22)": [
            -1.3598,-0.0728, 2.5363, 1.3782,-0.3383,
             0.4624, 0.2396, 0.0987, 0.3638, 0.0908,
            -0.5516,-0.6178,-0.9913,-0.3111, 1.4681,
            -0.4704, 0.2076, 0.1579, 0.1288,-0.2893,
            -0.0733,-0.3276,-0.9605,-0.1855, 0.3130,
             0.0197, 0.2774,-0.1104, 22.00],
        "High Value Normal ($500)": [
             1.191, 0.266,-0.166,-0.448,-0.601,
            -0.420,-0.327, 0.052, 0.065,-1.576,
             0.836,-0.401,-0.357,-0.550,-0.107,
            -0.420,-0.451,-0.208,-0.027, 0.082,
            -0.018,-0.238,-0.181, 0.055,-0.053,
             0.021,-0.009, 0.012,500.00],
        "Suspicious Pattern": [
            -2.312, 1.952,-1.610, 3.997,-0.522,
             1.800,-0.375,-0.246,-1.420, 0.804,
            -0.292,-1.585,-0.885,-2.176, 0.471,
            -0.681,-0.018,-1.082,-0.440, 0.734,
             0.574,-0.098,-0.194,-1.181, 0.648,
            -0.222, 0.082,-0.073,149.62],
        "Known Fraud Pattern": [
            -3.043,-3.157, 1.088, 2.288, 4.745,
             3.682,-0.489,-0.217,-4.779,-2.627,
            -4.545, 1.178,-3.427,-1.336,-0.243,
            -4.391,-0.260,-0.307,-0.512, 0.408,
             0.027, 0.161,-0.139,-0.067,-0.202,
            -0.167,-0.354,-0.040,  1.00],
    }

    default_vals = presets.get(preset, [0.0] * 29)
    feature_vals = []

    for row_start in range(0, 28, 5):
        row_feats = feature_names[row_start:row_start + 5]
        row_defs  = default_vals[row_start:row_start + 5]
        cols = st.columns(len(row_feats))
        for col, fname, dval in zip(cols, row_feats, row_defs):
            with col:
                feature_vals.append(
                    st.number_input(fname, value=float(dval),
                                    format="%.4f", key=f"feat_{fname}"))

    amount = st.number_input("Amount ($)",
                              value=float(default_vals[-1]),
                              min_value=0.0, format="%.2f",
                              key="amount_input")
    feature_vals.append(amount)
    features = np.array([feature_vals])

    if st.button("🔍 Analyze Transaction", use_container_width=True):
        with st.spinner("Running ensemble inference..."):
            feat = features.copy()
            feat[0, -1] = scaler.transform([[feat[0, -1]]])[0][0]
            try:
               if_score = float(-if_model.decision_function(feat))
            except Exception:
               if_score = 0.0   # fallback if sklearn version mismatch
            t = torch.FloatTensor(feat)
            with torch.no_grad():
                recon    = ae_model(t)
                ae_score = float(((t - recon) ** 2).mean())

            if_norm   = min(max((if_score + 0.1) / 0.3, 0), 1)
            ae_norm   = min(max(ae_score / 2.0, 0), 1)
            ens_score = cfg["w_if"] * if_norm + cfg["w_ae"] * ae_norm
            is_fraud  = ens_score >= cfg["threshold"]
            risk_pct  = round(min(ens_score * 100, 100), 1)

        sc      = "#ef4444" if is_fraud else "#10b981"
        verdict = "🚨 FRAUD DETECTED" if is_fraud else "✅ LEGITIMATE TRANSACTION"
        v_bg    = "rgba(239,68,68,0.1)"   if is_fraud else "rgba(16,185,129,0.1)"
        v_bdr   = "rgba(239,68,68,0.3)"   if is_fraud else "rgba(16,185,129,0.3)"

        st.markdown(f"""
        <div style="text-align:center;padding:2rem;background:{v_bg};
                    border:2px solid {v_bdr};border-radius:18px;margin:1.5rem 0;">
            <div style="font-size:2rem;font-weight:800;color:{sc};
                        letter-spacing:-0.02em;">{verdict}</div>
            <div style="font-size:0.78rem;color:#475569;margin-top:0.5rem;
                        font-family:'JetBrains Mono',monospace;">
                Ensemble Score: {ens_score:.4f} &nbsp;·&nbsp;
                Threshold: {cfg["threshold"]:.4f}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:1.5rem;">
        """ + "".join(f"""
            <div style="background:rgba(255,255,255,0.02);
                        border:1px solid rgba(255,255,255,0.07);
                        border-radius:14px;padding:1.3rem;
                        position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                            background:{bc};border-radius:14px 14px 0 0;"></div>
                <div style="font-size:1.3rem;margin-bottom:0.4rem;">{ic}</div>
                <div style="font-size:1.7rem;font-weight:800;color:{bc};
                            font-family:'JetBrains Mono',monospace;line-height:1;">{vl}</div>
                <div style="font-size:0.6rem;color:#475569;text-transform:uppercase;
                            letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                            margin:4px 0 0.8rem;">{nm}</div>
                <div style="height:3px;background:rgba(255,255,255,0.05);border-radius:3px;">
                    <div style="height:100%;width:{pw}%;background:{bc};border-radius:3px;"></div>
                </div>
            </div>""" for ic, nm, vl, bc, pw in [
            ("🌲", "Isolation Forest", f"{if_norm:.4f}", "#f59e0b", min(if_norm*100, 100)),
            ("🧠", "Autoencoder Score", f"{ae_norm:.4f}", "#7c3aed", min(ae_norm*100, 100)),
            ("⚡", "Ensemble Risk",    f"{risk_pct}%",   sc,         risk_pct),
        ]) + "</div>", unsafe_allow_html=True)

# ── CSV Upload ─────────────────────────────────────────────────
elif mode == "📁 Upload CSV":
    st.markdown("""
    <div style="font-size:0.75rem;color:#475569;font-family:'JetBrains Mono',monospace;
                margin-bottom:1rem;">
        CSV must contain columns: V1, V2, ..., V28, Amount
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload transactions CSV",
                                 type=["csv"],
                                 label_visibility="collapsed")
    if uploaded:
        df = pd.read_csv(uploaded)
        req_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
        if not all(c in df.columns for c in req_cols):
            st.error("❌ Missing required columns. CSV needs V1–V28 + Amount")
        else:
            df = df[req_cols].head(500)
            with st.spinner(f"Analyzing {len(df)} transactions..."):
                X = df.values.copy().astype(float)
                X[:, -1] = scaler.transform(
                    X[:, -1].reshape(-1, 1)).flatten()
                if_scores = -if_model.decision_function(X)
                t = torch.FloatTensor(X)
                with torch.no_grad():
                    recon     = ae_model(t)
                    ae_scores = ((t - recon) ** 2).mean(dim=1).numpy()
                ens = (cfg["w_if"] * normalize(if_scores) +
                       cfg["w_ae"] * normalize(ae_scores))
                preds        = (ens >= cfg["threshold"]).astype(int)
                df["Risk"]   = np.round(ens, 4)
                df["Verdict"] = ["🚨 FRAUD" if p else "✅ Normal"
                                  for p in preds]

            n_fraud  = int(preds.sum())
            n_normal = len(preds) - n_fraud

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(3,1fr);
                        gap:1rem;margin:1rem 0;">
            """ + "".join(f"""
                <div style="background:rgba(255,255,255,0.02);
                            border:1px solid rgba(255,255,255,0.07);
                            border-radius:14px;padding:1.2rem;text-align:center;">
                    <div style="font-size:1.8rem;font-weight:800;color:{c};
                                font-family:'JetBrains Mono',monospace;">{v}</div>
                    <div style="font-size:0.62rem;color:#475569;text-transform:uppercase;
                                letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;
                                margin-top:4px;">{k}</div>
                </div>""" for k, v, c in [
                ("Total Analyzed", f"{len(df):,}", "#94a3b8"),
                ("Flagged Fraud",  f"{n_fraud:,}", "#ef4444"),
                ("Legitimate",     f"{n_normal:,}","#10b981"),
            ]) + "</div>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(10, 3), facecolor="#030712")
            ax.set_facecolor("#0d1117")
            ax.hist(ens[preds==0], bins=50, color="#1f6feb",
                    alpha=0.7, density=True, label="Normal")
            if n_fraud > 0:
                ax.hist(ens[preds==1], bins=20, color="#ef4444",
                        alpha=0.9, density=True, label="Flagged Fraud")
            ax.axvline(cfg["threshold"], color="#ffa657",
                       linestyle="--", linewidth=2,
                       label=f"Threshold={cfg['threshold']:.3f}")
            ax.set_title("Risk Score Distribution",
                         color="#c9d1d9", fontsize=11)
            ax.set_xlabel("Ensemble Score", color="#8b949e")
            ax.legend(facecolor="#161b22", edgecolor="#30363d",
                      labelcolor="#c9d1d9")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values(): sp.set_edgecolor("#21262d")
            b64 = fig_to_b64(fig); plt.close(fig)
            st.markdown(
                f'<img src="data:image/png;base64,{b64}"'
                f' style="width:100%;border-radius:12px;margin-bottom:1rem;"/>',
                unsafe_allow_html=True)

            show_cols = ["V1","V2","V3","Amount","Risk","Verdict"]
            st.dataframe(df[show_cols], use_container_width=True, height=300)
            st.download_button("⬇️ Download Results CSV",
                                df.to_csv(index=False),
                                "fraud_results.csv", "text/csv",
                                use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:1.5rem;">
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#ef4444,#f59e0b);
                        display:flex;align-items:center;justify-content:center;font-size:1.2rem;">🛡️</div>
            <div>
                <div style="font-size:1rem;font-weight:800;color:#f1f5f9;">FraudShield AI</div>
                <div style="font-size:0.62rem;color:#475569;font-family:'JetBrains Mono',monospace;">
                    ENSEMBLE DETECTOR · v1.0
                </div>
            </div>
        </div>
    </div>
    <div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">
        ⬡ Model Performance
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Test ROC-AUC</span>
        <span style="font-size:0.75rem;font-weight:600;color:#ef4444;font-family:'JetBrains Mono',monospace;">0.9439</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Test PR-AUC</span>
        <span style="font-size:0.75rem;font-weight:600;color:#ef4444;font-family:'JetBrains Mono',monospace;">0.4555</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Normal Acc</span>
        <span style="font-size:0.75rem;font-weight:600;color:#10b981;font-family:'JetBrains Mono',monospace;">99.89%</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Fraud Recall</span>
        <span style="font-size:0.75rem;font-weight:600;color:#f59e0b;font-family:'JetBrains Mono',monospace;">50.0%</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">False Alarms</span>
        <span style="font-size:0.75rem;font-weight:600;color:#10b981;font-family:'JetBrains Mono',monospace;">0.11%</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Total Experiments</span>
        <span style="font-size:0.75rem;font-weight:600;color:#7c3aed;font-family:'JetBrains Mono',monospace;">63</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Training data</span>
        <span style="font-size:0.75rem;font-weight:600;color:#94a3b8;font-family:'JetBrains Mono',monospace;">199,134</span>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.03);">
        <span style="font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">Fraud labels</span>
        <span style="font-size:0.75rem;font-weight:600;color:#10b981;font-family:'JetBrains Mono',monospace;">None ✅</span>
    </div>
    <div style="margin-top:1.4rem;font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;
                color:#1e3a5f;font-family:'JetBrains Mono',monospace;margin-bottom:0.8rem;">⬡ Key Insight</div>
    <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.1);
                border-radius:12px;padding:1rem;font-size:0.73rem;color:#64748b;line-height:1.65;">
        Trained on <span style="color:#f1f5f9;font-weight:600;">normal transactions only</span>
        — no fraud labels used. V17 shows
        <span style="color:#ef4444;font-weight:600;">435× higher</span>
        reconstruction error in fraud vs normal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    threshold = st.slider("Detection Threshold", 0.05, 0.95, 0.50, 0.05)
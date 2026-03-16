import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ── Robust path resolution ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "career_model.joblib")
LE_PATH    = os.path.join(MODEL_DIR, "label_encoder.joblib")
DATA_PATH  = os.path.join(BASE_DIR,  "career_dataset.xlsx")

FEATURES = ["math","logic","creativity","communication","leadership",
            "problem_solving","programming","data_analysis","design",
            "networking","management","writing"]

# ── Auto-build model if .joblib files are missing ──────────────────────────────
def build_and_save_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    df   = pd.read_excel(DATA_PATH)
    le   = LabelEncoder()
    y    = le.fit_transform(df["Career"])
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=100, max_depth=12,
                                          random_state=42, n_jobs=-1))
    ])
    pipe.fit(df[FEATURES], y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipe, MODEL_PATH, compress=3)
    joblib.dump(le,   LE_PATH,   compress=3)
    return pipe, le

@st.cache_resource(show_spinner="🤖 Loading AI model…")
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LE_PATH):
        return build_and_save_model()
    try:
        return joblib.load(MODEL_PATH), joblib.load(LE_PATH)
    except Exception:
        return build_and_save_model()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Career Predictor AI", page_icon="🚀",
                   layout="wide", initial_sidebar_state="collapsed")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
:root{--bg:#07070e;--surface:#0f0f1a;--card:#14141f;--card2:#1b1b2a;--border:#252538;
      --accent:#7c3aed;--a2:#06b6d4;--a3:#f59e0b;--a4:#10b981;
      --text:#e2e8f0;--muted:#5a6478;--glow:rgba(124,58,237,.4);}
html,body,[class*="css"]{font-family:'Syne',sans-serif!important;
  background-color:var(--bg)!important;color:var(--text)!important;}
.stApp{background:var(--bg)!important;}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:1rem 2.5rem 3rem!important;max-width:1500px;}

.bg-anim{position:fixed;top:0;left:0;width:100vw;height:100vh;pointer-events:none;z-index:0;overflow:hidden;}
.orb{position:absolute;border-radius:50%;filter:blur(80px);opacity:.1;
  animation:drift 18s ease-in-out infinite alternate;}
.orb1{width:500px;height:500px;background:#7c3aed;top:-100px;left:-100px;}
.orb2{width:400px;height:400px;background:#06b6d4;bottom:-100px;right:-100px;animation-delay:-7s;}
.orb3{width:280px;height:280px;background:#f59e0b;top:45%;left:48%;animation-delay:-13s;}
@keyframes drift{from{transform:translate(0,0)scale(1);}to{transform:translate(55px,38px)scale(1.12);}}

.hero{text-align:center;padding:2.8rem 1rem 1.5rem;}
.hero-badge{display:inline-block;font-family:'Space Mono',monospace;font-size:.68rem;
  letter-spacing:.2em;text-transform:uppercase;color:var(--a2);
  border:1px solid rgba(6,182,212,.3);border-radius:999px;padding:.25rem .9rem;margin-bottom:1.2rem;
  animation:pulse-border 3s ease-in-out infinite;}
@keyframes pulse-border{0%,100%{border-color:rgba(6,182,212,.3);}
  50%{border-color:rgba(6,182,212,.7);box-shadow:0 0 14px rgba(6,182,212,.2);}}
.hero-title{font-size:clamp(2.8rem,5vw,4.5rem);font-weight:800;line-height:1.05;margin:0 0 .7rem;
  background:linear-gradient(135deg,#a78bfa 0%,#06b6d4 50%,#f59e0b 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{color:var(--muted);font-size:1rem;max-width:520px;margin:0 auto 1.5rem;line-height:1.8;}
.stats-row{display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;margin-bottom:1.5rem;}
.stat-pill{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:.6rem 1.3rem;text-align:center;transition:transform .2s,border-color .2s;}
.stat-pill:hover{transform:translateY(-3px);border-color:var(--accent);}
.stat-num{font-family:'Space Mono',monospace;font-size:1.3rem;font-weight:700;color:var(--a2);}
.stat-label{font-size:.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}

.live-badge{display:inline-flex;align-items:center;gap:.4rem;font-family:'Space Mono',monospace;
  font-size:.7rem;color:var(--a4);background:rgba(16,185,129,.1);
  border:1px solid rgba(16,185,129,.3);border-radius:999px;padding:.2rem .8rem;margin-bottom:.8rem;}
.live-dot{width:6px;height:6px;border-radius:50%;background:var(--a4);
  animation:blink 1.2s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1;}50%{opacity:.2;}}

.result-card{background:linear-gradient(135deg,rgba(124,58,237,.12),rgba(6,182,212,.07));
  border:1px solid rgba(124,58,237,.4);border-radius:22px;padding:1.8rem 2rem;text-align:center;
  position:relative;overflow:hidden;animation:fadeIn .5s ease;}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px);}to{opacity:1;transform:translateY(0);}}
.result-card::before{content:'';position:absolute;top:-70px;right:-70px;width:200px;height:200px;
  border-radius:50%;background:radial-gradient(circle,rgba(124,58,237,.3),transparent 70%);}
.result-icon{font-size:3rem;margin-bottom:.3rem;}
.result-career{font-size:2rem;font-weight:800;background:linear-gradient(90deg,#a78bfa,#06b6d4);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;margin:.2rem 0 .4rem;}
.result-conf{font-family:'Space Mono',monospace;font-size:.85rem;color:var(--a2);}
.conf-meter-bg{background:var(--surface);border-radius:999px;height:8px;overflow:hidden;margin:.8rem 0;}
.conf-meter-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--accent),var(--a2));}

.alt-row{display:flex;gap:.5rem;flex-wrap:wrap;justify-content:center;margin-top:.9rem;}
.alt-chip{background:var(--surface);border:1px solid var(--border);border-radius:999px;
  padding:.28rem .9rem;font-size:.75rem;font-family:'Space Mono',monospace;color:var(--muted);}

.section-title{font-size:1.1rem;font-weight:700;margin-bottom:1rem;
  display:flex;align-items:center;gap:.5rem;}
.dot{display:inline-block;width:8px;height:8px;border-radius:50%;
  background:var(--accent);box-shadow:0 0 8px var(--accent);}

.insight-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.7rem;margin-top:1rem;}
.insight-card{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:.9rem 1rem;text-align:center;transition:transform .2s,border-color .2s;}
.insight-card:hover{transform:translateY(-3px);border-color:var(--a3);}
.insight-icon{font-size:1.4rem;margin-bottom:.3rem;}
.insight-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;}
.insight-val{font-size:1rem;font-weight:700;color:var(--text);margin-top:.15rem;}

.quiz-progress{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--muted);
  text-align:right;margin-bottom:.5rem;}
.progress-bar-wrap{background:var(--surface);border-radius:999px;height:4px;margin-bottom:1.2rem;}
.progress-bar{height:100%;border-radius:999px;
  background:linear-gradient(90deg,var(--accent),var(--a2));transition:width .4s ease;}
.quiz-card{background:var(--card);border:1px solid var(--border);border-radius:18px;
  padding:1.8rem 2rem;margin-bottom:1rem;animation:fadeIn .4s ease;}
.quiz-q{font-size:1.15rem;font-weight:700;margin-bottom:1.2rem;line-height:1.5;}

div.stButton>button{width:100%;
  background:linear-gradient(135deg,var(--accent),var(--a2))!important;
  color:#fff!important;font-family:'Syne',sans-serif!important;font-size:1rem!important;
  font-weight:700!important;letter-spacing:.04em!important;border:none!important;
  border-radius:14px!important;padding:.8rem 2rem!important;
  transition:all .2s!important;box-shadow:0 4px 20px var(--glow)!important;}
div.stButton>button:hover{opacity:.85!important;transform:translateY(-2px)!important;}

.stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;
  border-radius:12px;padding:.3rem;gap:.2rem;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;
  border-radius:8px!important;font-family:'Syne',sans-serif!important;
  font-size:.85rem!important;font-weight:600!important;}
.stTabs [aria-selected="true"]{background:var(--card2)!important;color:var(--text)!important;}
.stTabs [data-baseweb="tab-panel"]{padding:1.2rem 0!important;}

[data-testid="metric-container"]{background:var(--card)!important;
  border:1px solid var(--border)!important;border-radius:14px!important;
  padding:1rem 1.2rem!important;}

.footer{text-align:center;padding:2rem 0 .5rem;font-family:'Space Mono',monospace;
  font-size:.68rem;color:var(--muted);border-top:1px solid var(--border);margin-top:3rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="bg-anim">
  <div class="orb orb1"></div><div class="orb orb2"></div><div class="orb orb3"></div>
</div>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
pipeline, le = load_models()

# ── Data constants ─────────────────────────────────────────────────────────────
CAREER_ICONS = {
    "Software Engineer":"💻","Data Scientist":"📊","Web Developer":"🌐",
    "Cybersecurity Analyst":"🛡️","AI/ML Engineer":"🤖","Database Administrator":"🗄️",
    "Network Engineer":"🔗","UI/UX Designer":"🎨","Product Manager":"📋","DevOps Engineer":"⚙️",
}
CAREER_DESC = {
    "Software Engineer":      "Design, build & maintain software systems and applications at scale.",
    "Data Scientist":         "Extract powerful insights from data using statistics, ML & storytelling.",
    "Web Developer":          "Build fast, responsive, beautiful web applications and sites.",
    "Cybersecurity Analyst":  "Defend systems and networks against digital threats and attacks.",
    "AI/ML Engineer":         "Design and deploy intelligent, self-learning systems & models.",
    "Database Administrator": "Manage, optimize, secure and scale databases across systems.",
    "Network Engineer":       "Design, implement and troubleshoot computer network infrastructure.",
    "UI/UX Designer":         "Craft intuitive, delightful user interfaces and experiences.",
    "Product Manager":        "Define product vision and lead cross-functional teams to launch.",
    "DevOps Engineer":        "Automate and bridge software development and IT operations.",
}
CAREER_SKILLS = {
    "Software Engineer":      dict(math=8,logic=9,creativity=6,communication=6,leadership=5,problem_solving=9,programming=9,data_analysis=6,design=4,networking=4,management=4,writing=5),
    "Data Scientist":         dict(math=10,logic=8,creativity=6,communication=6,leadership=5,problem_solving=8,programming=8,data_analysis=10,design=3,networking=3,management=4,writing=6),
    "Web Developer":          dict(math=6,logic=7,creativity=8,communication=6,leadership=4,problem_solving=7,programming=9,data_analysis=5,design=7,networking=5,management=3,writing=5),
    "Cybersecurity Analyst":  dict(math=7,logic=9,creativity=5,communication=6,leadership=5,problem_solving=9,programming=7,data_analysis=7,design=3,networking=9,management=5,writing=6),
    "AI/ML Engineer":         dict(math=10,logic=9,creativity=7,communication=5,leadership=4,problem_solving=9,programming=9,data_analysis=9,design=3,networking=4,management=4,writing=5),
    "Database Administrator": dict(math=7,logic=9,creativity=4,communication=6,leadership=5,problem_solving=8,programming=7,data_analysis=8,design=3,networking=6,management=6,writing=5),
    "Network Engineer":       dict(math=6,logic=8,creativity=4,communication=6,leadership=5,problem_solving=8,programming=5,data_analysis=5,design=3,networking=10,management=5,writing=5),
    "UI/UX Designer":         dict(math=4,logic=6,creativity=10,communication=8,leadership=5,problem_solving=7,programming=5,data_analysis=5,design=10,networking=3,management=4,writing=7),
    "Product Manager":        dict(math=6,logic=7,creativity=8,communication=10,leadership=9,problem_solving=8,programming=4,data_analysis=7,design=6,networking=6,management=9,writing=8),
    "DevOps Engineer":        dict(math=6,logic=8,creativity=5,communication=6,leadership=6,problem_solving=8,programming=8,data_analysis=6,design=3,networking=8,management=6,writing=5),
}
CAREER_SALARY = {
    "Software Engineer":"₹8L – ₹35L","Data Scientist":"₹10L – ₹40L",
    "Web Developer":"₹5L – ₹22L","Cybersecurity Analyst":"₹8L – ₹30L",
    "AI/ML Engineer":"₹12L – ₹50L","Database Administrator":"₹6L – ₹25L",
    "Network Engineer":"₹5L – ₹20L","UI/UX Designer":"₹5L – ₹25L",
    "Product Manager":"₹12L – ₹45L","DevOps Engineer":"₹8L – ₹32L",
}
CAREER_DEMAND = {
    "Software Engineer":"🔥 Very High","Data Scientist":"🔥 Very High",
    "Web Developer":"✅ High","Cybersecurity Analyst":"🔥 Very High",
    "AI/ML Engineer":"🚀 Explosive","Database Administrator":"✅ Moderate",
    "Network Engineer":"✅ Moderate","UI/UX Designer":"✅ High",
    "Product Manager":"✅ High","DevOps Engineer":"🔥 Very High",
}
SKILL_LABELS = {
    "math":"🔢 Mathematics","logic":"🧠 Logic","creativity":"🎨 Creativity",
    "communication":"💬 Communication","leadership":"👑 Leadership",
    "problem_solving":"🔍 Problem Solving","programming":"💻 Programming",
    "data_analysis":"📊 Data Analysis","design":"✏️ UI/UX Design",
    "networking":"🔗 Networking","management":"📋 Management","writing":"✍️ Writing",
}
PRESETS = {
    "🔬 Analyst":  dict(math=9,logic=8,creativity=5,communication=6,leadership=4,problem_solving=8,programming=7,data_analysis=9,design=3,networking=3,management=4,writing=6),
    "🎨 Creative": dict(math=4,logic=6,creativity=10,communication=8,leadership=5,problem_solving=7,programming=5,data_analysis=5,design=10,networking=3,management=4,writing=7),
    "👑 Leader":   dict(math=6,logic=7,creativity=8,communication=10,leadership=10,problem_solving=8,programming=4,data_analysis=7,design=6,networking=6,management=9,writing=8),
    "⚙️ Engineer": dict(math=8,logic=9,creativity=5,communication=6,leadership=5,problem_solving=9,programming=9,data_analysis=6,design=3,networking=7,management=5,writing=5),
}
QUIZ_QUESTIONS = [
    {"q":"🏃 You have a free weekend. What do you spend it on?",
     "opts":["Building a personal project or coding","Analysing data from a hobby","Designing graphics or UI mockups","Reading about business & strategy"],
     "skill_map":[{"programming":2,"logic":1},{"data_analysis":2,"math":1},{"design":2,"creativity":1},{"management":2,"leadership":1}]},
    {"q":"🧩 Which problem excites you most?",
     "opts":["Making an app 10× faster","Finding hidden patterns in messy data","Making a website beautiful & intuitive","Protecting a system from hackers"],
     "skill_map":[{"programming":2,"problem_solving":1},{"data_analysis":2,"math":1},{"design":2,"creativity":1},{"networking":2,"logic":1}]},
    {"q":"📚 You're picking a course. Which do you choose?",
     "opts":["Advanced Algorithms","Machine Learning & Statistics","Human-Centred Design","Cloud Infrastructure & DevOps"],
     "skill_map":[{"programming":2,"logic":2},{"math":2,"data_analysis":2},{"design":2,"creativity":1},{"networking":2,"management":1}]},
    {"q":"🤝 In a team project, your natural role is?",
     "opts":["The one writing the core code","The one mining insights from data","The one designing the experience","The one leading & coordinating"],
     "skill_map":[{"programming":2,"problem_solving":1},{"data_analysis":2,"math":1},{"design":2,"communication":1},{"leadership":2,"management":2}]},
    {"q":"💡 What describes your thinking style best?",
     "opts":["Methodical and systematic","Curious and data-driven","Visual and aesthetic","Strategic and people-focused"],
     "skill_map":[{"logic":2,"problem_solving":1},{"math":1,"data_analysis":2},{"creativity":2,"design":1},{"leadership":1,"communication":2}]},
    {"q":"🌐 Which tech area sounds most appealing?",
     "opts":["Building scalable backend systems","AI and predictive models","Network security & protocols","Agile product roadmaps"],
     "skill_map":[{"programming":2,"logic":1},{"math":2,"data_analysis":1},{"networking":2,"logic":1},{"management":2,"leadership":1}]},
]

# ── Session state initialisation ───────────────────────────────────────────────
# KEY FIX: slider values live in st.session_state["skills"] dict.
# Sliders use value= from that dict but NO key= argument.
# Presets/Reset update the dict → st.rerun() → sliders re-render with new value=.
# This avoids the StreamlitAPIException caused by writing to a widget-bound key.
if "skills" not in st.session_state:
    st.session_state["skills"] = {k: 5 for k in FEATURES}
if "quiz_step"      not in st.session_state: st.session_state["quiz_step"]      = 0
if "quiz_scores"    not in st.session_state: st.session_state["quiz_scores"]    = {k: 5 for k in FEATURES}
if "quiz_done"      not in st.session_state: st.session_state["quiz_done"]      = False
if "explorer_career" not in st.session_state: st.session_state["explorer_career"] = None
if "compare_a"      not in st.session_state: st.session_state["compare_a"]      = "Software Engineer"
if "compare_b"      not in st.session_state: st.session_state["compare_b"]      = "Data Scientist"

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict(vals: dict) -> list:
    inp   = pd.DataFrame([vals])[FEATURES]
    probs = pipeline.predict_proba(inp)[0]
    idx   = np.argsort(probs)[::-1]
    return [(le.inverse_transform([i])[0], float(probs[i]) * 100) for i in idx]

def radar_trace(vals, label, color, fill):
    cats = [SKILL_LABELS[k].split(" ", 1)[1] for k in FEATURES]
    r    = [vals[k] for k in FEATURES]
    return go.Scatterpolar(r=r+[r[0]], theta=cats+[cats[0]], fill="toself",
                           name=label, line=dict(color=color, width=2), fillcolor=fill)

def radar_layout(h=320):
    return dict(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,10],
                            tickfont=dict(color="#64748b", size=8), gridcolor="#252538"),
            angularaxis=dict(tickfont=dict(color="#94a3b8", size=9), gridcolor="#252538"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
        legend=dict(font=dict(color="#94a3b8", size=11), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=40, t=20, b=20), height=h,
    )

def prob_bars(results, height=320, title="All Career Match Probabilities"):
    careers = [c for c,_ in results]
    probs   = [p for _,p in results]
    colors  = ["#7c3aed" if i==0 else "#252538" for i in range(len(results))]
    fig = go.Figure(go.Bar(
        x=probs, y=[f"{CAREER_ICONS[c]} {c}" for c in careers],
        orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs], textposition="inside",
        textfont=dict(color="white", size=11, family="Space Mono"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(range=[0,100], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8", size=11), autorange="reversed"),
        margin=dict(l=10, r=20, t=28, b=10), height=height,
        title=dict(text=title, font=dict(color="#64748b", size=12), x=0),
    )
    return fig

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">✦ AI-Powered · Self-Training · Live Prediction · 4 Modes</div>
  <h1 class="hero-title">Career Predictor AI</h1>
  <p class="hero-sub">Discover your ideal tech career. Drag sliders for instant ML predictions,
  take the quiz, explore paths, and compare careers side-by-side.</p>
  <div class="stats-row">
    <div class="stat-pill"><div class="stat-num">88%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-pill"><div class="stat-num">10</div><div class="stat-label">Careers</div></div>
    <div class="stat-pill"><div class="stat-num">12</div><div class="stat-label">Skill Dims</div></div>
    <div class="stat-pill"><div class="stat-num">1K</div><div class="stat-label">Samples</div></div>
    <div class="stat-pill"><div class="stat-num">4</div><div class="stat-label">Modes</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🚀  Live Predictor", "🎯  Skill Quiz",
    "🗺️  Career Explorer", "⚖️  Compare Careers",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_L, col_R = st.columns([1, 1.1], gap="large")

    with col_L:
        st.markdown(
            '<div class="section-title"><span class="dot"></span> Rate Your Skills'
            '<span style="color:var(--muted);font-size:.8rem;font-weight:400">'
            ' — drag any slider for an instant AI result</span></div>',
            unsafe_allow_html=True,
        )

        # ── Preset buttons — update the skills dict then rerun ─────────────
        # SAFE: we write to st.session_state["skills"], NOT to any widget key
        pc = st.columns(4)
        for col, (pname, pvals) in zip(pc, PRESETS.items()):
            with col:
                if st.button(pname, key=f"preset_{pname}"):
                    st.session_state["skills"].update(pvals)
                    st.rerun()

        st.markdown("")

        # ── Sliders — NO key= argument; value= comes from the skills dict ──
        # Reading back slider values via on_change callback stored in skills dict
        current = st.session_state["skills"]
        values  = {}
        rA, rB  = st.columns(2)

        for i, feat in enumerate(FEATURES):
            with (rA if i % 2 == 0 else rB):
                new_val = st.slider(
                    SKILL_LABELS[feat],
                    min_value=1, max_value=10,
                    value=int(current[feat]),
                    key=f"sl_{feat}",          # unique display key, never written to externally
                )
                values[feat] = new_val
                # Keep the dict in sync so Explorer tab reads correct values
                st.session_state["skills"][feat] = new_val

        st.markdown("")
        # ── Reset button — SAFE: writes to skills dict, not widget key ─────
        if st.button("↺  Reset All to 5", key="reset_all"):
            st.session_state["skills"] = {k: 5 for k in FEATURES}
            # Force sliders to re-render with new value by clearing their keys
            for feat in FEATURES:
                if f"sl_{feat}" in st.session_state:
                    del st.session_state[f"sl_{feat}"]
            st.rerun()

    # ── RIGHT: live results ────────────────────────────────────────────────
    with col_R:
        results    = predict(values)
        top_career = results[0][0]
        top_conf   = results[0][1]
        ideal      = CAREER_SKILLS[top_career]

        alt_html = "".join(
            f'<span class="alt-chip">{CAREER_ICONS[c]} {c} · {p:.0f}%</span>'
            for c, p in results[1:5]
        )
        st.markdown(f"""
        <div class="live-badge"><span class="live-dot"></span> LIVE PREDICTION — updates on every slider drag</div>
        <div class="result-card">
          <div class="result-icon">{CAREER_ICONS[top_career]}</div>
          <div class="result-career">{top_career}</div>
          <div class="result-conf">Confidence: {top_conf:.1f}%</div>
          <div class="conf-meter-bg">
            <div class="conf-meter-fill" style="width:{top_conf:.1f}%"></div>
          </div>
          <p style="color:#94a3b8;font-size:.85rem;margin:.4rem 0 .6rem">{CAREER_DESC[top_career]}</p>
          <div class="alt-row">{alt_html}</div>
        </div>
        """, unsafe_allow_html=True)

        avg_s   = np.mean(list(values.values()))
        top_sk  = max(values, key=values.get)
        weak_sk = min(values, key=values.get)
        st.markdown(f"""
        <div class="insight-grid">
          <div class="insight-card"><div class="insight-icon">⭐</div>
            <div class="insight-label">Top Skill</div>
            <div class="insight-val">{SKILL_LABELS[top_sk].split(' ',1)[1]}</div></div>
          <div class="insight-card"><div class="insight-icon">📈</div>
            <div class="insight-label">Avg Score</div>
            <div class="insight-val">{avg_s:.1f} / 10</div></div>
          <div class="insight-card"><div class="insight-icon">🎯</div>
            <div class="insight-label">Grow This</div>
            <div class="insight-val">{SKILL_LABELS[weak_sk].split(' ',1)[1]}</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        fig_r = go.Figure()
        fig_r.add_trace(radar_trace(values, "You", "#7c3aed", "rgba(124,58,237,0.15)"))
        fig_r.add_trace(radar_trace(ideal, f"Ideal {top_career}", "#06b6d4", "rgba(6,182,212,0.08)"))
        fig_r.update_layout(**radar_layout(300))
        st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})

        st.plotly_chart(prob_bars(results), use_container_width=True, config={"displayModeBar": False})

        with st.expander(f"💰 Salary & Market Demand — {top_career}"):
            m1, m2 = st.columns(2)
            m1.metric("💸 Salary Range (India)", CAREER_SALARY[top_career])
            m2.metric("📈 Job Demand",           CAREER_DEMAND[top_career])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SKILL QUIZ
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        '<div class="section-title"><span class="dot"></span>'
        " Skill Discovery Quiz — 6 questions that auto-build your profile</div>",
        unsafe_allow_html=True,
    )

    if not st.session_state.quiz_done:
        step  = st.session_state.quiz_step
        total = len(QUIZ_QUESTIONS)

        st.markdown(f"""
        <div class="quiz-progress">Question {step + 1} of {total}</div>
        <div class="progress-bar-wrap">
          <div class="progress-bar" style="width:{int(step / total * 100)}%"></div>
        </div>
        """, unsafe_allow_html=True)

        q = QUIZ_QUESTIONS[step]
        st.markdown(f'<div class="quiz-card"><div class="quiz-q">{q["q"]}</div></div>', unsafe_allow_html=True)

        oc1, oc2 = st.columns(2)
        for i, opt in enumerate(q["opts"]):
            with (oc1 if i % 2 == 0 else oc2):
                if st.button(opt, key=f"qopt_{step}_{i}", use_container_width=True):
                    for sk, boost in q["skill_map"][i].items():
                        st.session_state.quiz_scores[sk] = min(10, st.session_state.quiz_scores[sk] + boost)
                    if step + 1 >= total:
                        st.session_state.quiz_done = True
                    else:
                        st.session_state.quiz_step += 1
                    st.rerun()

        if step > 0:
            st.markdown("")
            if st.button("← Back", key="quiz_back"):
                st.session_state.quiz_step -= 1
                st.rerun()
    else:
        qs       = st.session_state.quiz_scores
        qresults = predict(qs)
        qtop, qconf = qresults[0]

        alt_q = "".join(
            f'<span class="alt-chip">{CAREER_ICONS[c]} {c} · {p:.0f}%</span>'
            for c, p in qresults[1:5]
        )
        st.markdown(f"""
        <div class="result-card">
          <div style="font-size:.72rem;font-family:Space Mono,monospace;color:var(--a4);margin-bottom:.5rem">✅ QUIZ COMPLETE</div>
          <div class="result-icon">{CAREER_ICONS[qtop]}</div>
          <div class="result-career">{qtop}</div>
          <div class="result-conf">Quiz Confidence: {qconf:.1f}%</div>
          <div class="conf-meter-bg">
            <div class="conf-meter-fill" style="width:{qconf:.1f}%"></div>
          </div>
          <p style="color:#94a3b8;font-size:.85rem;margin:.5rem 0">{CAREER_DESC[qtop]}</p>
          <div class="alt-row">{alt_q}</div>
        </div>
        """, unsafe_allow_html=True)

        qc1, qc2 = st.columns(2)
        with qc1:
            st.markdown('<div class="section-title" style="margin-top:1.2rem"><span class="dot"></span> Your Quiz Profile vs Ideal</div>', unsafe_allow_html=True)
            fig_qr = go.Figure()
            fig_qr.add_trace(radar_trace(qs,                  "Your Profile",  "#7c3aed", "rgba(124,58,237,0.18)"))
            fig_qr.add_trace(radar_trace(CAREER_SKILLS[qtop], f"Ideal {qtop}", "#f59e0b", "rgba(245,158,11,0.08)"))
            fig_qr.update_layout(**radar_layout(300))
            st.plotly_chart(fig_qr, use_container_width=True, config={"displayModeBar": False})

        with qc2:
            st.markdown('<div class="section-title" style="margin-top:1.2rem"><span class="dot"></span> Top 5 Career Matches</div>', unsafe_allow_html=True)
            for c, p in qresults[:5]:
                st.markdown(f"""
                <div style="margin-bottom:.65rem">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.25rem">
                    <span style="font-size:.85rem">{CAREER_ICONS[c]} {c}</span>
                    <span style="font-family:Space Mono,monospace;font-size:.8rem;color:var(--a2)">{p:.1f}%</span>
                  </div>
                  <div style="background:var(--surface);border-radius:999px;height:7px">
                    <div style="width:{int(p)}%;height:100%;border-radius:999px;
                      background:linear-gradient(90deg,var(--accent),var(--a2))"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄  Retake Quiz", key="retake"):
                st.session_state.quiz_step   = 0
                st.session_state.quiz_done   = False
                st.session_state.quiz_scores = {k: 5 for k in FEATURES}
                st.rerun()
        with b2:
            if st.button("📋  Copy Scores → Live Predictor", key="copy_scores"):
                # SAFE: write to the skills dict, not to any widget key
                st.session_state["skills"] = {k: int(v) for k, v in qs.items()}
                # Clear slider widget state so they re-render with new values
                for feat in FEATURES:
                    if f"sl_{feat}" in st.session_state:
                        del st.session_state[f"sl_{feat}"]
                st.success("✅ Scores copied! Switch to the Live Predictor tab.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CAREER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="section-title"><span class="dot"></span>'
        " Career Explorer — click any card for a deep-dive & skill gap analysis</div>",
        unsafe_allow_html=True,
    )

    careers_list = list(CAREER_ICONS.keys())
    for ri, row in enumerate([careers_list[:5], careers_list[5:]]):
        cols = st.columns(5)
        for ci, career in enumerate(row):
            with cols[ci]:
                if st.button(
                    f"{CAREER_ICONS[career]}\n\n**{career}**\n\n{CAREER_DEMAND[career]}",
                    key=f"exp_{ri}_{ci}", use_container_width=True,
                ):
                    st.session_state.explorer_career = career
                    st.rerun()

    st.markdown("")
    sel = st.session_state.explorer_career

    if sel:
        ideal = CAREER_SKILLS[sel]
        cur   = {k: int(st.session_state["skills"].get(k, 5)) for k in FEATURES}

        st.markdown(f"""
        <div class="result-card" style="text-align:left;padding:1.5rem 2rem">
          <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:.5rem">
            <span style="font-size:3rem">{CAREER_ICONS[sel]}</span>
            <div style="flex:1">
              <div style="font-size:1.6rem;font-weight:800;
                background:linear-gradient(90deg,#a78bfa,#06b6d4);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text">{sel}</div>
              <div style="color:#64748b;font-size:.85rem">{CAREER_DESC[sel]}</div>
            </div>
            <div style="text-align:right">
              <div style="font-family:Space Mono,monospace;color:var(--a3);font-size:.72rem">SALARY (INDIA)</div>
              <div style="font-size:1.1rem;font-weight:700">{CAREER_SALARY[sel]}</div>
              <div style="color:#64748b;font-size:.8rem;margin-top:.2rem">{CAREER_DEMAND[sel]}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown('<div class="section-title" style="margin-top:1rem"><span class="dot"></span> Skill Gap: You vs Ideal</div>', unsafe_allow_html=True)
            for sk in FEATURES:
                iv   = ideal[sk]; cv = cur[sk]; diff = iv - cv
                color = "#10b981" if diff <= 0 else ("#f59e0b" if diff <= 2 else "#ef4444")
                tag   = "✅ On target" if diff <= 0 else f"📈 +{diff} needed"
                st.markdown(f"""
                <div style="margin-bottom:.6rem">
                  <div style="display:flex;justify-content:space-between;margin-bottom:.2rem">
                    <span style="font-size:.8rem">{SKILL_LABELS[sk]}</span>
                    <span style="font-size:.72rem;color:{color};font-family:Space Mono,monospace">{tag}</span>
                  </div>
                  <div style="background:var(--surface);border-radius:999px;height:7px;position:relative">
                    <div style="width:{cv*10}%;height:100%;border-radius:999px;
                      background:linear-gradient(90deg,var(--accent),var(--a2))"></div>
                    <div style="position:absolute;top:-3px;left:{iv*10}%;width:3px;height:13px;
                      background:var(--a3);border-radius:2px"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('<div style="font-size:.7rem;color:var(--muted)">🟡 Amber mark = ideal target score</div>', unsafe_allow_html=True)

        with ec2:
            st.markdown('<div class="section-title" style="margin-top:1rem"><span class="dot"></span> You vs Ideal Radar</div>', unsafe_allow_html=True)
            fig_e = go.Figure()
            fig_e.add_trace(radar_trace(cur,   "Your Skills",  "#7c3aed", "rgba(124,58,237,0.18)"))
            fig_e.add_trace(radar_trace(ideal, f"Ideal {sel}", "#f59e0b", "rgba(245,158,11,0.1)"))
            fig_e.update_layout(**radar_layout(340))
            st.plotly_chart(fig_e, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown("""
        <div style="background:var(--card);border:1px dashed var(--border);border-radius:18px;
                    padding:2.5rem;text-align:center;margin-top:.5rem">
          <div style="font-size:2.5rem;margin-bottom:.8rem">🗺️</div>
          <div style="color:var(--muted);font-size:.95rem;line-height:1.7">
            Click any career card above to explore its skill requirements,<br>
            salary range, job demand, and your personal skill gap.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — COMPARE CAREERS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        '<div class="section-title"><span class="dot"></span> Side-by-Side Career Comparison</div>',
        unsafe_allow_html=True,
    )

    cc1, cc2, cc3 = st.columns([1, 0.15, 1])
    with cc1:
        career_a = st.selectbox("Career A", list(CAREER_ICONS.keys()), key="cmp_a",
                                index=list(CAREER_ICONS.keys()).index(st.session_state.compare_a))
    with cc2:
        st.markdown('<div style="text-align:center;padding-top:2.3rem;font-size:1.5rem;color:var(--muted)">⚔️</div>', unsafe_allow_html=True)
    with cc3:
        career_b = st.selectbox("Career B", list(CAREER_ICONS.keys()), key="cmp_b",
                                index=list(CAREER_ICONS.keys()).index(st.session_state.compare_b))
    # SAFE: cmp_a / cmp_b are selectbox keys, but we only write to compare_a / compare_b
    # which are separate plain state keys (not bound to any widget)
    st.session_state.compare_a = career_a
    st.session_state.compare_b = career_b

    if career_a == career_b:
        st.warning("⚠️ Please select two different careers to compare.")
    else:
        hc1, hc2 = st.columns(2)
        for col, career in zip([hc1, hc2], [career_a, career_b]):
            with col:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-radius:16px;
                            padding:1.2rem 1.4rem;text-align:center;margin-bottom:1rem">
                  <div style="font-size:2.5rem">{CAREER_ICONS[career]}</div>
                  <div style="font-size:1.2rem;font-weight:800;margin:.3rem 0">{career}</div>
                  <div style="font-family:Space Mono,monospace;color:var(--a2);font-size:.78rem">{CAREER_SALARY[career]}</div>
                  <div style="color:var(--muted);font-size:.75rem;margin-top:.3rem">{CAREER_DEMAND[career]}</div>
                </div>
                """, unsafe_allow_html=True)

        fig_cmp = go.Figure()
        fig_cmp.add_trace(radar_trace(CAREER_SKILLS[career_a], career_a, "#7c3aed", "rgba(124,58,237,0.18)"))
        fig_cmp.add_trace(radar_trace(CAREER_SKILLS[career_b], career_b, "#06b6d4", "rgba(6,182,212,0.12)"))
        fig_cmp.update_layout(**radar_layout(380))
        st.plotly_chart(fig_cmp, use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-title"><span class="dot"></span> Skill-by-Skill Breakdown</div>', unsafe_allow_html=True)
        sa, sb = CAREER_SKILLS[career_a], CAREER_SKILLS[career_b]
        h0, h1, h2, h3 = st.columns([2, 1, .4, 1])
        h0.markdown("**Skill**")
        h1.markdown(f"**{CAREER_ICONS[career_a]} {career_a}**")
        h2.markdown("**vs**")
        h3.markdown(f"**{CAREER_ICONS[career_b]} {career_b}**")
        st.markdown("---")

        for sk in FEATURES:
            va, vb = sa[sk], sb[sk]
            arrow  = "🔺" if va > vb else ("🔻" if va < vb else "➖")
            bar_a  = f'<div style="height:7px;width:{va*10}%;background:linear-gradient(90deg,#7c3aed,#a78bfa);border-radius:999px;margin-top:4px"></div>'
            bar_b  = f'<div style="height:7px;width:{vb*10}%;background:linear-gradient(90deg,#06b6d4,#67e8f9);border-radius:999px;margin-top:4px"></div>'
            r0,r1,r2,r3 = st.columns([2, 1, .4, 1])
            r0.markdown(SKILL_LABELS[sk])
            r1.markdown(f"**{va}/10**{bar_a}", unsafe_allow_html=True)
            r2.markdown(arrow)
            r3.markdown(f"**{vb}/10**{bar_b}", unsafe_allow_html=True)

        va_arr = np.array([sa[k] for k in FEATURES])
        vb_arr = np.array([sb[k] for k in FEATURES])
        sim = 100 - (np.mean(np.abs(va_arr - vb_arr)) / 9 * 100)
        msg = ("Very similar — skills transfer well! 🤝" if sim > 70 else
               "Quite different — distinct strengths needed 🔀" if sim < 50 else
               "Moderate overlap — complementary paths 🔄")
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:14px;
                    padding:1rem 1.4rem;text-align:center;margin-top:1rem">
          <div style="color:var(--muted);font-size:.75rem;font-family:Space Mono,monospace;
            text-transform:uppercase;letter-spacing:.1em">Career Similarity Score</div>
          <div style="font-size:2rem;font-weight:800;color:var(--a3)">{sim:.0f}%</div>
          <div style="color:var(--muted);font-size:.82rem">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

# ── ABOUT ──────────────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️  About this model & dataset"):
    a1, a2, a3 = st.columns(3)
    a1.markdown("**🤖 Algorithm**\n\nRandom Forest (100 trees, depth 12) in a scikit-learn Pipeline with StandardScaler. Saved with **Joblib** (compressed). Auto-rebuilds from Excel if missing.")
    a2.markdown("**📊 Dataset**\n\n1 000 synthetic profiles in **Excel (.xlsx)** — 10 tech careers × 12 skill dimensions. Loaded via `openpyxl`.")
    a3.markdown("**🎯 Performance**\n\n~88% test accuracy, ~85% 5-fold CV. Full probability distribution returned for all 10 career classes.")

st.markdown("""
<div class="footer">
  Career Predictor AI · Streamlit + scikit-learn + Joblib · Excel Dataset<br>
  🚀 Live Predictor · 🎯 Skill Quiz · 🗺️ Career Explorer · ⚖️ Compare Careers
</div>
""", unsafe_allow_html=True)

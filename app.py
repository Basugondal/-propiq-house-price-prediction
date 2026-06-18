import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="PropIQ — Pakistan Real Estate AI",
    page_icon="",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&display=swap');

html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
    font-family: 'Roboto', sans-serif !important;
    background: #f1f3ff !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important; visibility: hidden !important;
}

.block-container {
    padding: 0 !important;
    max-width: 900px !important;
}

/* ── HERO ── */
.propiq-hero {
    width: 100%;
    border-radius: 24px;
    overflow: hidden;
    position: relative;
    min-height: 380px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 32px;
    background: #1e1b4b;
}

.propiq-hero-img {
    position: absolute; inset: 0;
    background: url('https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=1200&q=80') center/cover;
    opacity: 0.22;
}

.propiq-hero-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(30,27,75,0.95) 0%, rgba(67,56,202,0.85) 60%, rgba(109,40,217,0.8) 100%);
}

.propiq-hero-grid {
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(129,140,248,0.08) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(129,140,248,0.08) 1px, transparent 1px);
    background-size: 44px 44px;
}

.propiq-hero-content {
    position: relative; z-index: 3;
    text-align: center;
    padding: 48px 32px;
}

.propiq-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(129,140,248,0.18);
    border: 1px solid rgba(129,140,248,0.35);
    border-radius: 50px;
    padding: 5px 16px;
    font-size: 11px; font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 2px; text-transform: uppercase;
    margin-bottom: 18px;
}

.propiq-title {
    font-family: 'Roboto', sans-serif;
    font-size: 2rem; font-weight: 900;
    color: #fff;
    line-height: 1.2; letter-spacing: -0.5px;
    margin-bottom: 12px;
    word-break: keep-all;
    overflow-wrap: break-word;
}

.propiq-title .g {
    background: linear-gradient(90deg, #a5b4fc, #c084fc, #f9a8d4);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.propiq-sub {
    font-size: 0.95rem; color: #94a3b8;
    max-width: 460px; margin: 0 auto 28px;
    line-height: 1.7; font-weight: 400;
}

.propiq-stats {
    display: flex; justify-content: center;
    gap: 32px; flex-wrap: wrap;
}

.propiq-stat-num {
    font-family: 'Roboto', sans-serif;
    font-size: 1.7rem; font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1;
}

.propiq-stat-lbl {
    font-size: 10px; color: #475569;
    text-transform: uppercase; letter-spacing: 1.2px; margin-top: 3px;
}

/* ── SECTION HEADING ── */
.propiq-sh {
    text-align: center;
    margin: 40px 0 20px;
}

.propiq-sh-tag {
    display: inline-block;
    background: linear-gradient(135deg, #ede9fe, #fce7f3);
    color: #7c3aed; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 4px 14px; border-radius: 50px;
    margin-bottom: 8px;
}

.propiq-sh-title {
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 1.7rem; font-weight: 800;
    color: #1e1b4b; letter-spacing: -0.8px;
}

/* ── PREDICTOR CARD ── */
.propiq-card {
    background: #ffffff;
    border-radius: 20px;
    padding: 32px 28px;
    box-shadow: 0 2px 8px rgba(99,102,241,0.07),
                0 12px 40px rgba(99,102,241,0.10);
    border: 1px solid rgba(99,102,241,0.1);
    margin-bottom: 8px;
}

/* ── WIDGETS ── */
label[data-testid="stWidgetLabel"] p {
    color: #1e1b4b !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    font-family: 'Roboto', sans-serif !important;
}

[data-testid="stNumberInput"] input {
    background: #f5f3ff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 600 !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.12) !important;
    background: #fff !important;
}

[data-baseweb="select"] > div {
    background: #f5f3ff !important;
    border: 1.5px solid #ddd6fe !important;
    border-radius: 10px !important;
    color: #1e1b4b !important;
    font-family: 'Roboto', sans-serif !important;
    font-weight: 500 !important;
}

/* ── BUTTON ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #d946ef) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Roboto', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    width: 100% !important;
    padding: 0.8rem !important;
    box-shadow: 0 4px 18px rgba(99,102,241,0.35) !important;
    letter-spacing: 0.2px !important;
    margin-top: 6px !important;
}

[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 28px rgba(99,102,241,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── RESULT ── */
.propiq-result-wrap {
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #d946ef);
    border-radius: 20px; padding: 3px;
    box-shadow: 0 12px 40px rgba(99,102,241,0.28);
    margin: 20px 0 8px;
}

.propiq-result-inner {
    background: linear-gradient(135deg, #fafbff, #f5f3ff);
    border-radius: 18px; padding: 30px 24px;
    text-align: center;
}

.propiq-result-badge {
    display: inline-block;
    background: linear-gradient(135deg, #ede9fe, #fce7f3);
    color: #7c3aed; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 4px 14px; border-radius: 50px; margin-bottom: 12px;
}

.propiq-result-price {
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #6366f1, #8b5cf6, #d946ef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1; margin-bottom: 6px;
}

.propiq-result-desc {
    font-size: 0.85rem; color: #64748b; margin-bottom: 20px;
    font-family: 'Roboto', sans-serif;
}

.propiq-range {
    display: flex; border-radius: 12px;
    overflow: hidden; border: 1px solid #e0e7ff;
    background: #fff;
}

.propiq-range-col {
    flex: 1; padding: 14px 8px; text-align: center;
    border-right: 1px solid #e0e7ff;
}

.propiq-range-col:last-child { border-right: none; }
.propiq-range-col.mid { background: linear-gradient(135deg, #f5f3ff, #fdf4ff); }

.propiq-range-val {
    font-family: 'Roboto', sans-serif;
    font-size: 1rem; font-weight: 700; color: #4f46e5;
}

.propiq-range-col.mid .propiq-range-val {
    background: linear-gradient(135deg, #6366f1, #d946ef);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

.propiq-range-lbl {
    font-size: 9px; color: #94a3b8;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
    font-family: 'Roboto', sans-serif;
}

/* ── HOW IT WORKS ── */
.propiq-hiw {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border-radius: 20px; padding: 36px 28px;
    margin: 32px 0;
}

.propiq-hiw-title {
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 1.5rem; font-weight: 800;
    color: #fff; text-align: center;
    letter-spacing: -0.8px; margin-bottom: 6px;
}

.propiq-hiw-sub {
    text-align: center; color: #64748b;
    font-size: 0.85rem; margin-bottom: 28px;
    font-family: 'Roboto', sans-serif;
}

.propiq-hiw-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px;
}

.propiq-hiw-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 20px 16px;
}

.propiq-hiw-icon { font-size: 1.6rem; margin-bottom: 8px; }

.propiq-hiw-step {
    font-size: 9px; color: #818cf8; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px;
    font-family: 'Roboto', sans-serif;
}

.propiq-hiw-ctitle {
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 0.95rem; font-weight: 700; color: #fff; margin-bottom: 4px;
}

.propiq-hiw-cdesc {
    font-size: 0.8rem; color: #475569; line-height: 1.5;
    font-family: 'Roboto', sans-serif;
}

/* ── CITIES ── */
.propiq-cities-grid {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 12px; margin-bottom: 32px;
}

.propiq-city {
    border-radius: 14px; overflow: hidden;
    position: relative; height: 140px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
}

.propiq-city img {
    width: 100%; height: 100%; object-fit: cover;
    filter: brightness(0.55) saturate(1.2);
    display: block;
}

.propiq-city-lbl {
    position: absolute; bottom: 0; left: 0; right: 0;
    padding: 16px 12px 10px;
    background: linear-gradient(transparent, rgba(0,0,0,0.82));
    color: #fff;
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 0.95rem; font-weight: 700;
}

.propiq-city-sub {
    font-size: 10px; color: #a5b4fc;
    font-family: 'Roboto', sans-serif;
    font-weight: 400; margin-top: 1px;
}

/* ── CONTACT ── */
.propiq-contact {
    border-radius: 20px; overflow: hidden;
    position: relative; margin: 8px 0 40px;
    background: #1e1b4b;
}

.propiq-contact-bg {
    position: absolute; inset: 0;
    background: url('https://images.unsplash.com/photo-1486325212027-8081e485255e?w=1200&q=80') center/cover;
    opacity: 0.1;
}

.propiq-contact-overlay {
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(30,27,75,0.95), rgba(67,56,202,0.9));
}

.propiq-contact-inner {
    position: relative; z-index: 2;
    padding: 44px 28px; text-align: center;
}

.propiq-contact-tag {
    display: inline-block;
    background: rgba(129,140,248,0.18);
    border: 1px solid rgba(129,140,248,0.3);
    color: #a5b4fc; font-size: 10px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    padding: 4px 14px; border-radius: 50px; margin-bottom: 14px;
    font-family: 'Roboto', sans-serif;
}

.propiq-contact-title {
    font-family: 'Roboto', 'Roboto', sans-serif;
    font-size: 1.8rem; font-weight: 800;
    color: #fff; letter-spacing: -1px; margin-bottom: 8px;
}

.propiq-contact-sub {
    font-size: 0.88rem; color: #64748b;
    line-height: 1.7; margin-bottom: 28px;
    font-family: 'Roboto', sans-serif;
}

.propiq-contact-btns {
    display: flex; justify-content: center; gap: 12px; flex-wrap: wrap;
}

.propiq-btn {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 10px 22px; border-radius: 10px;
    font-family: 'Roboto', sans-serif;
    font-weight: 600; font-size: 0.88rem;
    text-decoration: none; transition: all 0.2s;
    cursor: pointer;
}

.propiq-btn-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35);
}

.propiq-btn-secondary {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #e2e8f0;
}

.propiq-footer {
    margin-top: 28px; padding-top: 20px;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: #334155; font-size: 11px;
    font-family: 'Roboto', sans-serif;
}
</style>
""", unsafe_allow_html=True)
@st.cache_resource
def load_model():
    model    = pickle.load(open("model.pkl", "rb"))
    features = pickle.load(open("model_features.pkl", "rb"))
    return model, features

try:
    model, model_features = load_model()
except Exception as e:
    st.error(f"model.pkl / model_features.pkl same folder mein rakhein: {e}")
    st.stop()

cities    = sorted([f.replace("location_city_","")     for f in model_features if f.startswith("location_city_")])
provinces = sorted([f.replace("location_province_","") for f in model_features if f.startswith("location_province_")])
purposes  = sorted([f.replace("purpose_","")           for f in model_features if f.startswith("purpose_")])

def fmt(val):
    if val >= 1e7:   return f"PKR {val/1e7:.2f} Cr"
    elif val >= 1e5: return f"PKR {val/1e5:.1f} Lakh"
    else:            return f"PKR {val:,.0f}"

st.markdown("""
<div class="propiq-hero">
    <div class="propiq-hero-img"></div>
    <div class="propiq-hero-overlay"></div>
    <div class="propiq-hero-grid"></div>
    <div class="propiq-hero-content">
        <div class="propiq-eyebrow">🤖 AI-Powered · Pakistan Real Estate</div>
        <div class="propiq-title">
            Smarter Property<br>
            <span class="g">Price Estimates</span>
        </div>
        <div class="propiq-sub">
            Instant AI valuations powered by 53,000+ real Zameen.com listings
            and a Random Forest model across 100+ Pakistani cities.
        </div>
        <div class="propiq-stats">
            <div><div class="propiq-stat-num">53K+</div><div class="propiq-stat-lbl">Listings</div></div>
            <div><div class="propiq-stat-num">73%</div><div class="propiq-stat-lbl">R² Score</div></div>
            <div><div class="propiq-stat-num">100+</div><div class="propiq-stat-lbl">Cities</div></div>
            <div><div class="propiq-stat-num">300</div><div class="propiq-stat-lbl">RF Trees</div></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="propiq-sh">
    <div class="propiq-sh-tag">✦ AI Price Engine</div>
    <div class="propiq-sh-title">Estimate Your Property Value</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    area     = st.number_input(" Area (Marla)", min_value=1.0, max_value=5000.0, value=10.0, step=0.5)
    bedrooms = st.number_input(" Bedrooms",     min_value=1,   max_value=10,     value=3)
with c2:
    bathrooms = st.number_input(" Bathrooms",   min_value=1,   max_value=10,     value=2)
    city      = st.selectbox(" City",           ["Select City"] + cities)
with c3:
    province = st.selectbox(" Province",        ["Select Province"] + provinces)
    purpose  = st.selectbox(" Purpose",         ["Select Purpose"] + purposes)

predict = st.button("⚡ Get AI Price Estimate")

if predict:
    if "Select" in [city, province, purpose]:
        st.warning("⚠️ Please select City, Province, and Purpose.")
    else:
        try:
            row = pd.DataFrame([{col: 0 for col in model_features}])
            row["area"]    = area
            row["bedroom"] = bedrooms
            row["bath"]    = bathrooms
            for val, pfx in [(city,"location_city_"),(province,"location_province_"),(purpose,"purpose_")]:
                k = f"{pfx}{val}"
                if k in row.columns: row[k] = 1

            predicted = max(0, model.predict(row)[0])
            low, high = predicted*0.88, predicted*1.12

            st.markdown(f"""
            <div class="propiq-result-wrap">
                <div class="propiq-result-inner">
                    <div class="propiq-result-badge">✦ AI Estimated Market Value</div>
                    <div class="propiq-result-price">{fmt(predicted)}</div>
                    <div class="propiq-result-desc">
                        {area} Marla &nbsp;·&nbsp; {bedrooms} Bed &nbsp;·&nbsp;
                        {bathrooms} Bath &nbsp;·&nbsp; {city}, {province}
                    </div>
                    <div class="propiq-range">
                        <div class="propiq-range-col">
                            <div class="propiq-range-val">{fmt(low)}</div>
                            <div class="propiq-range-lbl">Low</div>
                        </div>
                        <div class="propiq-range-col mid">
                            <div class="propiq-range-val">{fmt(predicted)}</div>
                            <div class="propiq-range-lbl">AI Estimate</div>
                        </div>
                        <div class="propiq-range-col">
                            <div class="propiq-range-val">{fmt(high)}</div>
                            <div class="propiq-range-lbl">High</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("""
<div class="propiq-hiw">
    <div class="propiq-hiw-title">How PropIQ Works</div>
    <div class="propiq-hiw-sub">From raw data to accurate estimates — in milliseconds</div>
    <div class="propiq-hiw-grid">
        <div class="propiq-hiw-card">
            <div class="propiq-hiw-icon"></div>
            <div class="propiq-hiw-step">Step 01</div>
            <div class="propiq-hiw-ctitle">Data Collection</div>
            <div class="propiq-hiw-cdesc">53,000+ real listings scraped from Zameen.com across Pakistan</div>
        </div>
        <div class="propiq-hiw-card">
            <div class="propiq-hiw-icon"></div>
            <div class="propiq-hiw-step">Step 02</div>
            <div class="propiq-hiw-ctitle">AI Preprocessing</div>
            <div class="propiq-hiw-cdesc">Price parsing, area conversion, outlier removal & feature engineering</div>
        </div>
        <div class="propiq-hiw-card">
            <div class="propiq-hiw-icon"></div>
            <div class="propiq-hiw-step">Step 03</div>
            <div class="propiq-hiw-ctitle">Model Training</div>
            <div class="propiq-hiw-cdesc">Random Forest with 300 trees trained on 160 engineered features</div>
        </div>
        <div class="propiq-hiw-card">
            <div class="propiq-hiw-icon"></div>
            <div class="propiq-hiw-step">Step 04</div>
            <div class="propiq-hiw-ctitle">Instant Prediction</div>
            <div class="propiq-hiw-cdesc">Enter details and get your AI-powered price estimate in real time</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="propiq-sh">
    <div class="propiq-sh-tag">✦ Coverage</div>
    <div class="propiq-sh-title">Major Markets We Cover</div>
</div>
<div class="propiq-cities-grid">
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1599940824399-b87987ceb72a?w=500&q=80"/>
        <div class="propiq-city-lbl">Lahore<div class="propiq-city-sub">Punjab's Real Estate Hub</div></div>
    </div>
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1567157577867-05ccb1388e66?w=500&q=80"/>
        <div class="propiq-city-lbl">Karachi<div class="propiq-city-sub">Financial Capital</div></div>
    </div>
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1578662996442-48f60103fc96?w=500&q=80"/>
        <div class="propiq-city-lbl">Islamabad<div class="propiq-city-sub">The Capital Territory</div></div>
    </div>
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1486325212027-8081e485255e?w=500&q=80"/>
        <div class="propiq-city-lbl">Rawalpindi<div class="propiq-city-sub">Twin City Market</div></div>
    </div>
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=500&q=80"/>
        <div class="propiq-city-lbl">Faisalabad<div class="propiq-city-sub">Industrial Hub</div></div>
    </div>
    <div class="propiq-city">
        <img src="https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500&q=80"/>
        <div class="propiq-city-lbl">+ 100 More<div class="propiq-city-sub">Across All Provinces</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="propiq-contact">
    <div class="propiq-contact-bg"></div>
    <div class="propiq-contact-overlay"></div>
    <div class="propiq-contact-inner">
        <div class="propiq-contact-tag">✦ Get In Touch</div>
        <div class="propiq-contact-title">Built by Abdul Basit</div>
        <div class="propiq-contact-sub">
            BSc Software Engineering · University of Sargodha · 2026<br>
            Flutter Developer &amp; AI/ML Enthusiast · Pakistan
        </div>
        <div class="propiq-contact-btns">
            <a class="propiq-btn propiq-btn-primary" href="chabdalbasit@gmail.com">✉ Email Me</a>
            <a class="propiq-btn propiq-btn-secondary" href="https://github.com/yourusername" target="_blank">⌥ GitHub</a>
            <a class="propiq-btn propiq-btn-secondary" href="www.linkedin.com/in/muhammad-abdul-basit-43a460406" target="_blank">in LinkedIn</a>
        </div>
        <div class="propiq-footer">
            PropIQ · Zameen.com Data · © 2026 Basit
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
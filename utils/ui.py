import html
import streamlit as st

from config import DISCLAIMER


def apply_global_styles():
    st.markdown(
        """
        <style>
        :root {
            --navy: #12355b;
            --blue: #2563eb;
            --sky: #eef6ff;
            --ink: #14213d;
            --muted: #64748b;
            --line: rgba(30, 64, 175, .12);
            --success: #15803d;
            --warning: #b45309;
        }

        .stApp {
            background:
                radial-gradient(circle at 90% 5%, rgba(59,130,246,.08), transparent 24rem),
                linear-gradient(180deg, #f8fbff 0%, #ffffff 45%, #f8fafc 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3.5rem;
            max-width: 1220px;
        }

        h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.2rem 2.35rem;
            border-radius: 28px;
            background: linear-gradient(135deg, #0d2c54 0%, #174c88 52%, #2f75ca 100%);
            color: white;
            margin-bottom: 1.4rem;
            box-shadow: 0 18px 45px rgba(15, 45, 85, .18);
        }
        .hero::after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            right: -85px;
            top: -110px;
            background: rgba(255,255,255,.09);
        }
        .hero::before {
            content: "";
            position: absolute;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            right: 80px;
            bottom: -120px;
            background: rgba(255,255,255,.06);
        }
        .hero h1 {
            color: white;
            margin: 0 0 .5rem 0;
            font-size: clamp(2rem, 4vw, 3.15rem);
            line-height: 1.03;
        }
        .hero p {
            margin: 0;
            opacity: .92;
            font-size: 1.08rem;
            max-width: 720px;
        }
        .hero-badge {
            display: inline-block;
            margin-bottom: .8rem;
            padding: .38rem .72rem;
            border-radius: 999px;
            background: rgba(255,255,255,.14);
            border: 1px solid rgba(255,255,255,.22);
            font-size: .82rem;
            font-weight: 700;
            letter-spacing: .02em;
        }

        .section-kicker {
            font-size: .8rem;
            font-weight: 800;
            letter-spacing: .10em;
            text-transform: uppercase;
            color: #3b82f6;
            margin-bottom: .25rem;
        }

        .soft-card {
            padding: 1.25rem 1.35rem;
            border-radius: 20px;
            border: 1px solid var(--line);
            background: rgba(255,255,255,.84);
            min-height: 128px;
            box-shadow: 0 9px 28px rgba(15, 23, 42, .055);
        }
        .soft-card h3, .soft-card h4 { margin-top: 0; }
        .soft-card p { color: #475569; margin-bottom: 0; }

        .feature-card {
            padding: 1.2rem 1.25rem;
            border-radius: 20px;
            background: rgba(255,255,255,.86);
            border: 1px solid var(--line);
            box-shadow: 0 8px 24px rgba(15,23,42,.05);
            min-height: 145px;
        }
        .feature-icon { font-size: 1.45rem; margin-bottom: .55rem; }
        .feature-title { font-size: 1.04rem; font-weight: 800; color: var(--ink); }
        .feature-text { color: #64748b; font-size: .92rem; margin-top: .35rem; }

        .model-card {
            padding: 1.15rem 1.2rem;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(15,23,42,.05);
            min-height: 126px;
        }
        .model-name { font-weight: 850; font-size: 1.05rem; color: var(--ink); }
        .status-pill {
            display: inline-block;
            margin-top: .65rem;
            padding: .32rem .64rem;
            border-radius: 999px;
            font-size: .80rem;
            font-weight: 800;
        }
        .status-ready { background: #ecfdf3; color: #15803d; border: 1px solid #bbf7d0; }
        .status-missing { background: #fff7ed; color: #b45309; border: 1px solid #fed7aa; }

        .step-title {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin: .45rem 0 .95rem 0;
        }
        .step-circle {
            width: 34px;
            height: 34px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #dbeafe;
            color: #1d4ed8;
            font-weight: 900;
        }
        .step-text { font-size: 1.16rem; font-weight: 850; color: var(--ink); }

        .flow-wrap {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: .7rem;
            margin: .7rem 0 .2rem 0;
        }
        .flow-item {
            text-align: center;
            padding: .9rem .6rem;
            background: #ffffff;
            border: 1px solid var(--line);
            border-radius: 16px;
            color: #334155;
            font-size: .88rem;
            font-weight: 700;
        }

        .result-banner {
            padding: 1.45rem 1.5rem;
            border-radius: 22px;
            margin: .7rem 0 1rem 0;
            border: 1px solid;
            box-shadow: 0 10px 26px rgba(15,23,42,.06);
        }
        .result-normal { background: #f0fdf4; border-color: #bbf7d0; }
        .result-pneumonia { background: #fff7ed; border-color: #fed7aa; }
        .result-label { color: #64748b; font-size: .82rem; font-weight: 800; text-transform: uppercase; letter-spacing: .08em; }
        .result-main { font-size: 1.72rem; font-weight: 900; color: var(--ink); margin-top: .15rem; }
        .result-note { color: #475569; margin-top: .3rem; }

        .tiny-muted { color: #64748b; font-size: .86rem; }
        .callout {
            padding: 1rem 1.1rem;
            border-radius: 16px;
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            color: #334155;
        }

        [data-testid="stMetric"] {
            background: rgba(255,255,255,.88);
            border: 1px solid rgba(15, 23, 42, .08);
            padding: .92rem;
            border-radius: 16px;
            box-shadow: 0 7px 20px rgba(15,23,42,.04);
        }
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,.72);
            border-radius: 18px;
        }
        .stButton > button, [data-testid="stPageLink-NavLink"] {
            border-radius: 13px !important;
            font-weight: 750 !important;
        }
        .stButton > button[kind="primary"] {
            min-height: 3rem;
            font-size: 1rem;
            box-shadow: 0 8px 20px rgba(37,99,235,.17);
        }

        @media (max-width: 850px) {
            .flow-wrap { grid-template-columns: 1fr 1fr; }
            .hero { padding: 1.65rem 1.45rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badge: str = "FYP2 • Medical AI Research Prototype"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{html.escape(badge)}</div>
            <h1>{html.escape(title)}</h1>
            <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(step: int | None, title: str, kicker: str | None = None):
    if kicker:
        st.markdown(f'<div class="section-kicker">{html.escape(kicker)}</div>', unsafe_allow_html=True)
    if step is None:
        st.subheader(title)
        return
    st.markdown(
        f'<div class="step-title"><span class="step-circle">{step}</span><span class="step-text">{html.escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def model_status_card(model_name: str, ready: bool, detail: str = ""):
    label = "Ready to analyse" if ready else "Model file missing"
    status_class = "status-ready" if ready else "status-missing"
    detail_html = f'<div class="tiny-muted" style="margin-top:.5rem">{html.escape(detail)}</div>' if detail else ""
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-name">{html.escape(model_name)}</div>
            <span class="status-pill {status_class}">{label}</span>
            {detail_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def result_banner(display_class: str, confidence: float):
    is_pneumonia = display_class == "Pneumonia-like"
    css = "result-pneumonia" if is_pneumonia else "result-normal"
    note = (
        "The image contains visual patterns the selected model associates more strongly with the Pneumonia class."
        if is_pneumonia
        else "The image contains visual patterns the selected model associates more strongly with the Normal class."
    )
    st.markdown(
        f"""
        <div class="result-banner {css}">
            <div class="result-label">AI analysis result</div>
            <div class="result-main">{html.escape(display_class)}</div>
            <div class="result-note">Model confidence: <b>{confidence * 100:.1f}%</b><br>{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def disclaimer():
    st.warning(DISCLAIMER, icon="⚠️")

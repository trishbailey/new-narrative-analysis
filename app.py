import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import json
import time
import re
import textwrap
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import concurrent.futures
# NEW imports for distinctness & QA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics.pairwise import cosine_similarity

from difflib import SequenceMatcher

# ML / NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.cluster import MiniBatchKMeans

# ---------------------------
# Configuration & Constants
# ---------------------------
XAI_API_KEY_ENV_VAR = 'XAI_API_KEY'
GROK_MODEL = 'grok-4-fast-reasoning'
CLASSIFICATION_MODEL = 'grok-3-mini'   # used for seed labeling
API_BASE_URL = "https://api.x.ai/v1/chat/completions"

# Theme extraction sampling targets
THEMES_TOTAL_SAMPLE_TARGET = 800          # initial coverage target (small/med datasets)
BATCH_SIZE_FOR_THEMES = 400               # map-reduce batch size
SAMPLING_SPLIT = (0.40, 0.40, 0.20)       # (time-random, top-engagement, novelty)
NOVELTY_TFIDF_MAX_DOCS = 20000            # cap TF-IDF corpus size for novelty
OTHER_RATE_THRESHOLD = 0.35               # trigger refresh if >35% Other
MAX_REFRESH_ROUNDS = 1

# --- Theme distinctness & QA thresholds ---
THEME_SIM_THRESHOLD = 0.80     # cosine similarity threshold to flag near-duplicates
THEME_TERM_JACCARD = 0.60      # token overlap threshold (secondary check)
CLUSTER_K_MIN = 6              # min clusters per batch
CLUSTER_K_MAX = 12             # max clusters per batch
CLUSTER_EXEMPLARS = 12         # exemplar posts sent to LLM per cluster
TIEBREAK_MAX = 300             # cap LLM tie-break calls on uncertain classifications
ABSTAIN_PROB = 0.45            # below this max-prob => abstain to LLM
ABSTAIN_MARGIN = 0.10          # if (p1 - p2) < margin => abstain to LLM

# Hybrid classification
AI_SEED_SAMPLE_SIZE = 200                 # better diversity for LinearSVC
MAX_POSTS_FOR_ANALYSIS = 150              # (kept for corpus snippets if needed)
CLASSIFICATION_DEFAULT = "Other/Unrelated"

# Meltwater column mapping (confirmed)
TEXT_COLUMNS = ['Opening Text', 'Headline', 'Hit Sentence']
ENGAGEMENT_COLUMN = 'Likes'
AUTHOR_COLUMN = 'Influencer'
DATE_COLUMN = 'Date'
TIME_COLUMN = 'Time'

# If Date already includes time (e.g., "23-Sep-2025 04:56PM")
DATE_TIME_FORMAT = '%d-%b-%Y %I:%M%p'

# UI styling
PRIMARY_COLOR = "#1E88E5"
BG_COLOR = "#ffffff"
SIDEBAR_BG = "#f7f7f7"
TEXT_COLOR = "#333333"
HEADER_COLOR = "#111111"

# Toxicity keywords (abridged header; full list retained for scoring)
TOXIC_KEYWORDS = [
    'hate','hateful','hatred','racist','sexist','homophobe','bigot','fascist',
    'kill','murder','assassinate','slaughter','genocide','exterminate',
    'violent','violence','rage','angry','fury','wrath','attack','beat',
    'bomb','explode','shoot','stab','fight','war','terror','terrorist',
    'die','dead','death','suffer','pain','hurt','bleed','burn',
    'bitch','cunt','pussy','dick','asshole','fuck','shit','damn',
    'nigger','chink','spic','kike','faggot','dyke','tranny',
    'rape','molest','abuse','torment',
    # (… full list from your prior version retained …)
    'corporatism'
]
TOXICITY_THRESHOLD = 0.01

# ---------------------------
# Streamlit Page & Theme
# ---------------------------
st.set_page_config(
    page_title="Narrative Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Narrative Analyzer for Meltwater Data"}
)

st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG}; color: {TEXT_COLOR}; }}
.stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR};
         font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }}
h1, h2, h3, h4, h5, h6 {{ color: {HEADER_COLOR} !important; }}
p, span, label, div {{ color: {TEXT_COLOR} !important; }}
.stButton > button.primary {{ background-color: {PRIMARY_COLOR} !important; border-color: {PRIMARY_COLOR} !important; color: white !important; }}
.stButton > button.primary:hover {{ background-color: #1565C0 !important; border-color: #1565C0 !important; }}
.stButton > button:not(.primary) {{ color: {HEADER_COLOR} !important; border-color: #d0d0d0 !important; background-color: #fafafa !important; }}
div[data-testid="stAlert"] * p, div[data-testid="stAlert"] * h5 {{ color: {TEXT_COLOR} !important; }}
code {{ background-color: #f2f2f2; color: #d63384; padding: 2px 4px; border-radius: 4px; }}
.js-plotly-plot {{ background-color: {BG_COLOR} !important; }}
.chart-subtitle {{ font-size: 0.95rem; color: #6b7280; margin-top: -10px; margin-bottom: 12px; }}
.chart-source {{ font-size: 0.85rem; color: #9ca3af; margin-top: 8px; }}
</style>
""",
    unsafe_allow_html=True
)

VIBRANT_QUAL = [
    "#1E88E5","#E53935","#8E24AA","#43A047","#FB8C00","#00ACC1",
    "#F4511E","#3949AB","#6D4C41","#7CB342","#D81B60","#00897B"
]
vibrant_layout = pio.templates["plotly_white"].layout.to_plotly_json()
vibrant_layout.update({
    "font": {"family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "size": 14, "color": "#111111"},
    "colorway": VIBRANT_QUAL,
    "title": {"x": 0.0, "xanchor": "left", "y": 0.95, "yanchor": "top", "font": {"size": 20}},
    "margin": {"l": 80, "r": 40, "t": 60, "b": 60},
    "paper_bgcolor": "white","plot_bgcolor": "white",
    "xaxis": {"showgrid": True,"gridcolor": "#e9eaec","gridwidth": 1,"tickformat": ",","ticks": "","tickfont": {"size": 12,"color": "#111111"}},
    "yaxis": {"showgrid": True,"gridcolor": "#e9eaec","gridwidth": 1,"tickformat": ",","ticks": "","tickfont": {"size": 12,"color": "#111111"}, "automargin": True},
    "legend": {"orientation": "h","yanchor": "bottom","y": -0.25,"xanchor": "left","x": 0.0,"title": {"text": ""}, "itemclick": "toggleothers"},
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": "white","bordercolor": "#d1d5db","font": {"family": "Inter","size": 12,"color": "#111111"}}
})
pio.templates["vibrant"] = pio.templates["plotly_white"]
pio.templates["vibrant"].layout.update(vibrant_layout)

# ---------------------------
# Helpers
# ---------------------------
def finalize_figure(fig, title: str, subtitle: str | None = None, source: str | None = None, height: int | None = None):
    fig.update_layout(template="vibrant", title={"text": title})
    if height: fig.update_layout(height=height)
    if subtitle:
        fig.add_annotation(text=f"<span style='color:#6b7280'>{subtitle}</span>",
                           xref="paper", yref="paper", x=0, y=1.07, showarrow=False, align="left")
    return fig

def wrap_text(s: str, width: int = 16) -> str:
    return "<br>".join(textwrap.wrap(s, width))

# Adaptive time axis for daily → yearly spans
def configure_time_axis(fig, dates: pd.Series):
    if dates.empty: return fig
    dmin = pd.to_datetime(dates.min())
    dmax = pd.to_datetime(dates.max())
    span_days = max((dmax - dmin).days, 1)

    if span_days <= 10:
        dtick = "D1";  tickformat = "%b %d"
    elif span_days <= 60:
        dtick = "D7";  tickformat = "%b %d"
    elif span_days <= 180:
        dtick = "M1";  tickformat = "%b %Y"
    elif span_days <= 365:
        dtick = "M1";  tickformat = "%b %Y"
    elif span_days <= 730:
        dtick = "M3";  tickformat = "%b %Y"
    else:
        dtick = "M12"; tickformat = "%Y"

    fig.update_xaxes(type="date", dtick=dtick, tickformat=tickformat)
    return fig

# Normalize LLM narratives
def normalize_narratives(raw):
    if not isinstance(raw, list): return []
    title_keys = {'narrative_title','title','narrative','theme','name'}
    summary_keys = {'summary','desc','description','overview','rationale','explanation'}
    out = []
    for item in raw:
        if isinstance(item, dict):
            title, summary = None, ''
            for k in title_keys:
                if k in item and item[k]: title = str(item[k]).strip(); break
            for k in summary_keys:
                if k in item and item[k]: summary = str(item[k]).strip(); break
            if title: out.append({'narrative_title': title, 'summary': summary})
        elif isinstance(item, str) and item.strip():
            out.append({'narrative_title': item.strip(), 'summary': ''})
    return out

# Toxicity scoring
def compute_toxicity_scores(df_viz):
    def post_density(text):
        if pd.isna(text) or not text: return 0.0
        words = text.lower().split()
        if not words: return 0.0
        matches = sum(1 for kw in TOXIC_KEYWORDS if kw.lower() in text.lower())
        return matches / len(words)
    df_viz['toxicity_density'] = df_viz['POST_TEXT'].apply(post_density)
    df_viz['is_toxic'] = df_viz['toxicity_density'] > TOXICITY_THRESHOLD
    theme_toxicity = df_viz.groupby('NARRATIVE_TAG').agg(
        Avg_Density=('toxicity_density','mean'),
        Pct_Toxic_Posts=('is_toxic','mean')
    ).reset_index()
    theme_toxicity['Pct_Toxic_Posts'] *= 100
    return df_viz, theme_toxicity.sort_values('Avg_Density', ascending=False)

# ---------------------------
# API Call with backoff
# ---------------------------
def call_grok_with_backoff(payload, api_key, max_retries=5):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(max_retries):
        try:
            r = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=300)
            r.raise_for_status()
            j = r.json()
            if j.get('choices') and j['choices'][0].get('message'):
                return j['choices'][0]['message']['content']
            st.error("Unexpected API response structure.")
            return None
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
            return None
    st.error("Max retries reached. API call failed.")
    return None

# ---------------------------
# LLM: Theme extraction & explanations
# ---------------------------
def analyze_narratives(corpus, api_key):
    system_prompt = (
        "You are a world-class social media narrative analyst. Analyze the provided corpus and identify "
        "the 6–10 most significant, high-level themes or sub-narratives. Include at least TWO low-volume or "
        "emerging themes that might be underrepresented. Return a single JSON array of objects with keys "
        "narrative_title and summary. Titles must be specific and chart-ready."
    )
    user_query = f"Corpus sample:\n{corpus}"
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role":"system","content": system_prompt},
                     {"role":"user","content": user_query}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type":"ARRAY","items":{"type":"OBJECT","properties":{
                    "narrative_title":{"type":"STRING"},
                    "summary":{"type":"STRING"}}}
            }
        },
        "temperature": 0.5
    }
    with st.spinner("Extracting narrative themes..."):
        resp = call_grok_with_backoff(payload, api_key)
    if not resp: return None
    try:
        parsed = json.loads(resp)
    except json.JSONDecodeError:
        st.error("Failed to parse JSON response from Grok.")
        st.code(resp)
        return None
    normalized = normalize_narratives(parsed)
    return normalized if normalized else None

def tie_break_llm(post_text: str, themes: list[dict], api_key: str) -> str:
    """
    Ask LLM to assign a theme using inclusion/exclusion rules.
    Returns a theme title or CLASSIFICATION_DEFAULT.
    """
    themes_brief = [
        {
            "narrative_title": t.get("narrative_title",""),
            "inclusion_rule": t.get("inclusion_rule",""),
            "exclusion_rules": t.get("exclusion_rules",[]),
            "summary": t.get("summary","")
        } for t in themes
    ]
    system = ("You assign a single best-fitting theme for the post using the provided inclusion/exclusion rules. "
              f"If none match, return exactly '{CLASSIFICATION_DEFAULT}'. Respond with ONLY the theme title or that exact fallback.")
    user = json.dumps({"post": post_text, "themes": themes_brief}, ensure_ascii=False)
    payload = {
        "model": CLASSIFICATION_MODEL,
        "messages": [{"role":"system","content": system},{"role":"user","content": user}],
        "temperature": 0.0, "max_tokens": 50
    }
    r = call_grok_with_backoff(payload, api_key)
    if not r: 
        return CLASSIFICATION_DEFAULT
    label = r.strip()
    titles = [t.get('narrative_title') for t in themes if t.get('narrative_title')]
    return label if label in titles or label == CLASSIFICATION_DEFAULT else CLASSIFICATION_DEFAULT

def train_and_classify_hybrid(df_full, theme_titles, api_key):
    """
    Phase 1: LLM seed labels (unchanged logic).
    Phase 2: Calibrated LinearSVC for probabilities.
    Phase 3: Predict; abstain to LLM tie-break when uncertain.
    """
    st.info(f"Phase 1: Labeling {AI_SEED_SAMPLE_SIZE} examples with {CLASSIFICATION_MODEL}...")
    actual_seed = min(AI_SEED_SAMPLE_SIZE, len(df_full))
    df_sample = df_full.sample(actual_seed, random_state=42).copy()
    df_rest = df_full.drop(df_sample.index).copy()

    # LLM seed
    seed_tags = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(classify_post_for_seed, t, theme_titles, api_key): t for t in df_sample['POST_TEXT']}
        progress = st.progress(0)
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            seed_tags.append(fut.result())
            progress.progress((i+1)/actual_seed)
    progress.empty()
    df_sample['NARRATIVE_TAG'] = seed_tags

    if df_sample['NARRATIVE_TAG'].nunique() < 2:
        st.error("Seed labels lacked diversity. Try regenerating themes or increasing seed size.")
        return None

    # Phase 2: Calibrated classifier
    st.success("Phase 2: Training Calibrated LinearSVC...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=5000), method='sigmoid', cv=3))
    ])
    model.fit(df_sample['POST_TEXT'], df_sample['NARRATIVE_TAG'])

    # Phase 3: Predict with abstention + LLM tie-break
    st.info(f"Phase 3: Classifying remaining {len(df_rest):,} posts with abstention...")
    proba = model.predict_proba(df_rest['POST_TEXT'])
    classes = list(model.named_steps['clf'].classes_)
    top_idx = proba.argmax(axis=1)
    top_prob = proba.max(axis=1)
    # second best margin
    sorted_idx = np.argsort(proba, axis=1)
    second_best = proba[np.arange(proba.shape[0]), sorted_idx[:,-2]]
    margin = top_prob - second_best

    # initial labels
    labels = np.array([classes[i] for i in top_idx])

    # decide uncertain
    uncertain_mask = (top_prob < ABSTAIN_PROB) | (margin < ABSTAIN_MARGIN)
    uncertain_indices = np.where(uncertain_mask)[0]
    if len(uncertain_indices) > 0:
        st.warning(f"Running LLM tie-break on {min(len(uncertain_indices), TIEBREAK_MAX)} uncertain posts...")
        # Pull the current themes from session (contains rules)
        themes_full = st.session_state.narrative_data
        # Limit for cost; you can raise TIEBREAK_MAX if needed
        to_fix = uncertain_indices[:TIEBREAK_MAX]
        texts = df_rest['POST_TEXT'].iloc[to_fix].tolist()
        fixed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futs = [ex.submit(tie_break_llm, t, themes_full, api_key) for t in texts]
            for fut in concurrent.futures.as_completed(futs):
                fixed.append(fut.result())
        for local_i, lab in enumerate(fixed):
            abs_i = to_fix[local_i]
            labels[abs_i] = lab

    df_rest['NARRATIVE_TAG'] = labels
    st.success("Full dataset classification complete.")
    return pd.concat([df_sample, df_rest], ignore_index=True)

# ---------------------------
# Data crunching & takeaways
# ---------------------------
def perform_data_crunching_and_summary(df_classified: pd.DataFrame) -> str:
    theme_metrics = df_classified.groupby('NARRATIVE_TAG').agg(
        Volume=('POST_TEXT','size'),
        Total_Likes=(ENGAGEMENT_COLUMN,'sum')
    ).reset_index()
    theme_metrics['Avg_Likes_Per_Post'] = theme_metrics['Total_Likes'] / theme_metrics['Volume']
    theme_metrics = theme_metrics.sort_values('Volume', ascending=False)

    narrative_summary = "Narrative Metrics (Volume, Total Likes, Avg Likes Per Post):\n"
    narrative_summary += theme_metrics.to_string(index=False, float_format="%.2f") + "\n\n"

    overall_top_authors = df_classified.groupby(AUTHOR_COLUMN).agg(
        Total_Likes=(ENGAGEMENT_COLUMN,'sum'),
        Post_Count=('POST_TEXT','size')
    ).nlargest(3, 'Total_Likes').reset_index()

    author_summary = "Overall Top 3 Influencers (by Total Likes):\n"
    author_summary += overall_top_authors.to_string(index=False) + "\n\n"

    valid_dates = df_classified['DATETIME'].dropna()
    if not valid_dates.empty:
        start_date = valid_dates.min().strftime('%Y-%m-%d')
        end_date = valid_dates.max().strftime('%Y-%m-%d')
    else:
        start_date = end_date = "N/A"
    total_posts = len(df_classified)
    total_likes_all = df_classified[ENGAGEMENT_COLUMN].sum()

    context_summary = (
        f"Dataset Context:\n"
        f" Total Posts Analyzed: {total_posts:,}\n"
        f" Total Likes Across All Posts: {total_likes_all:,}\n"
        f" Timeframe: {start_date} to {end_date}\n"
    )
    return narrative_summary + author_summary + context_summary

def generate_takeaways(summary_data, api_key):
    system_prompt = (
        "You are a senior social media analyst. Provide exactly five concise, data-driven takeaways "
        "based only on the provided summary. Return a JSON array of strings."
    )
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role":"system","content": system_prompt},
                     {"role":"user","content": f"Summary:\n{summary_data}"}],
        "generationConfig": {"responseMimeType":"application/json",
                             "responseSchema":{"type":"ARRAY","items":{"type":"STRING"}}},
        "temperature": 0.4
    }
    with st.spinner("Generating executive takeaways..."):
        r = call_grok_with_backoff(payload, st.session_state.api_key)
    if not r: return None
    try:
        return json.loads(r)
    except json.JSONDecodeError:
        st.error("Failed to parse takeaways JSON.")
        return None

# ---------------------------
# Visualizations
# ---------------------------
def plot_toxicity_by_theme(theme_toxicity):
    fig = px.bar(
        theme_toxicity,
        x='NARRATIVE_TAG', y='Avg_Density',
        title='Toxicity Density by Narrative Theme',
        labels={'Avg_Density':'Average Toxic Keyword Density','NARRATIVE_TAG':'Narrative Theme'},
        height=500, color='Avg_Density', color_continuous_scale='Reds'
    )
    def wrap_labels_bar(text): return '<br>'.join(textwrap.wrap(text, 15))
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        xaxis={'categoryorder':'total descending','tickangle':0,'automargin':True,
               'tickvals': theme_toxicity['NARRATIVE_TAG'].tolist(),
               'ticktext':[wrap_labels_bar(t) for t in theme_toxicity['NARRATIVE_TAG']]},
        showlegend=False, coloraxis_colorbar=dict(title="Density")
    )
    return finalize_figure(fig, "Toxicity density by narrative theme",
                           "Average proportion of angry/hateful/violent keywords per post; sorted descending", 500)

def plot_theme_influencer_share(df_viz, theme, author_col, engagement_col, top_n=5):
    df_theme = df_viz[df_viz['NARRATIVE_TAG'] == theme].copy()
    if df_theme.empty: return None
    df_grouped = df_theme.groupby(author_col)[engagement_col].sum().reset_index(name='Author_Likes')
    df_top = df_grouped.nlargest(top_n, 'Author_Likes')
    theme_total = df_theme[engagement_col].sum()
    df_top['Percentage'] = np.where(theme_total > 0, (df_top['Author_Likes']/theme_total)*100, 0.0)
    fig = px.bar(df_top, x='Author_Likes', y=author_col, orientation='h',
                 labels={'Author_Likes':'Likes', author_col:'Influencer'},
                 height=400, color=author_col, color_discrete_sequence=VIBRANT_QUAL)
    fig.update_traces(hovertemplate='<b>%{y}</b><br>Likes: %{x:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>',
                      customdata=df_top[['Percentage']], marker_line_width=0)
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title='Total Likes')
    return finalize_figure(fig, f"{theme}: top {top_n} influencers by likes",
                           "Share computed against total theme likes; only top authors shown", 380)

def plot_overall_author_ranking(df_classified, author_col, engagement_col, top_n=10):
    df_classified['Theme_Rank'] = df_classified.groupby([author_col,'NARRATIVE_TAG'])['POST_TEXT'].transform('count')
    df_classified = df_classified.sort_values(by=['Theme_Rank'], ascending=False)
    df_primary_theme = df_classified.drop_duplicates(subset=[author_col], keep='first')[[author_col,'NARRATIVE_TAG']].rename(columns={'NARRATIVE_TAG':'Primary_Theme'})
    overall_metrics = df_classified.groupby(author_col).agg(Total_Likes=(engagement_col,'sum'),
                                                            Total_Posts=('POST_TEXT','size')).reset_index()
    overall_metrics = overall_metrics.merge(df_primary_theme, on=author_col, how='left')
    overall_metrics = overall_metrics.sort_values('Total_Likes', ascending=False).head(top_n)
    fig = px.bar(overall_metrics, x='Total_Likes', y=author_col, color='Primary_Theme',
                 orientation='h', height=550, color_discrete_sequence=VIBRANT_QUAL)
    fig.update_traces(hovertemplate='<b>%{y}</b><br>Likes: %{x:,}<br>Posts: %{customdata[0]}<br>Primary Theme: %{customdata[1]}<extra></extra>',
                      customdata=overall_metrics[['Total_Posts','Primary_Theme']], marker_line_width=0)
    fig.update_layout(yaxis={'title':None,'automargin':True}, xaxis={'tickformat':',','title':'Total Likes'})
    return finalize_figure(fig, f"Top {top_n} influencers by total likes", height=550)

# ---------------------------
# Data loading (robust CSV/XLSX + datetime)
# ---------------------------
def _has_required(df: pd.DataFrame) -> bool:
    cols = set(df.columns.astype(str).str.strip())
    required = set(TEXT_COLUMNS + [ENGAGEMENT_COLUMN, AUTHOR_COLUMN, DATE_COLUMN, TIME_COLUMN])
    return required.issubset(cols)

def load_meltwater(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    is_csv = name.endswith(".csv")
    if is_csv:
        uploaded.seek(0); df0 = pd.read_csv(uploaded, dtype=str)
    else:
        uploaded.seek(0); df0 = pd.read_excel(uploaded, engine='openpyxl')
    if not _has_required(df0):
        if is_csv:
            uploaded.seek(0)
            df1 = pd.read_csv(uploaded, header=None, dtype=str)
            df1.columns = df1.iloc[0].astype(str).str.strip()
            df1 = df1.iloc[1:].reset_index(drop=True)
        else:
            uploaded.seek(0)
            df1 = pd.read_excel(uploaded, engine='openpyxl', skiprows=1)
        if _has_required(df1):
            df = df1
        else:
            have = set(df0.columns.astype(str).str.strip())
            need = set(TEXT_COLUMNS + [ENGAGEMENT_COLUMN, AUTHOR_COLUMN, DATE_COLUMN, TIME_COLUMN])
            missing = [c for c in need if c not in have]
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
    else:
        df = df0

    df.columns = df.columns.astype(str).str.strip()
    if any(str(c).startswith("Unnamed") for c in df.columns):
        st.warning("Detected 'Unnamed' columns—header may be malformed.")

    # Build POST_TEXT robustly
    def _safe_join(row):
        parts = []
        for col in TEXT_COLUMNS:
            val = row.get(col, None)
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        return " | ".join(parts)
    df["POST_TEXT"] = df.apply(_safe_join, axis=1)

    # Likes -> numeric
    df[ENGAGEMENT_COLUMN] = pd.to_numeric(df[ENGAGEMENT_COLUMN], errors='coerce').fillna(0).astype(int)

    # Datetime robust
    df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).str.strip()
    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()
    dt_primary = pd.to_datetime(df[DATE_COLUMN], format=DATE_TIME_FORMAT, errors='coerce')
    needs_b = dt_primary.isna()
    if needs_b.any():
        combined = (df.loc[needs_b, DATE_COLUMN] + " " + df.loc[needs_b, TIME_COLUMN]).str.strip()
        dt_fallback = pd.to_datetime(combined, errors='coerce')  # infer formats
        dt_primary = dt_primary.where(~needs_b, dt_fallback)
    df["DATETIME"] = dt_primary

    parse_success = df["DATETIME"].notna().mean() * 100
    if parse_success < 90:
        st.warning(f"Datetime parsing succeeded for only {parse_success:.1f}% of rows.")
    else:
        st.success(f"Datetime parsing complete: {parse_success:.1f}% success rate ({df['DATETIME'].notna().sum():,} valid).")

    # Drop rows with truly empty text
    df = df[df["POST_TEXT"].str.strip().astype(bool)]

    return df

# ---------------------------
# Sampling utilities (stratified mixed)
# ---------------------------
def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    # Use whatever exists; default heavily to Likes
    score = pd.Series(0.0, index=df.index, dtype=float)
    weights = {
        'Likes': 1.0, 'Like': 1.0,
        'Comments': 1.5, 'Comment': 1.5, 'Replies': 1.5,
        'Reposts': 2.0, 'Retweets': 2.0, 'Shares': 2.0, 'Quote Tweets': 2.0,
        'Engagements': 1.0
    }
    for col, w in weights.items():
        if col in df.columns:
            score = score + pd.to_numeric(df[col], errors='coerce').fillna(0) * w
    # Ensure we at least use Likes if nothing else
    if score.eq(0).all():
        score = pd.to_numeric(df.get(ENGAGEMENT_COLUMN, 0), errors='coerce').fillna(0)
    return score

def assign_time_bins(df: pd.DataFrame) -> pd.Series:
    """Return a bin label per row: day, week, or month based on span."""
    valid = df['DATETIME'].dropna()
    if valid.empty:
        return pd.Series("bin_all", index=df.index)
    span_days = (valid.max() - valid.min()).days
    if span_days <= 30:
        bins = df['DATETIME'].dt.to_period('D').astype(str)  # daily
    elif span_days <= 180:
        bins = df['DATETIME'].dt.to_period('W').astype(str)  # weekly
    else:
        bins = df['DATETIME'].dt.to_period('M').astype(str)  # monthly
    return bins

def stratified_mixed_sample_indices(df: pd.DataFrame,
                                    target_n:int = THEMES_TOTAL_SAMPLE_TARGET,
                                    split:tuple = SAMPLING_SPLIT,
                                    random_state:int = 42) -> list:
    """40% time-stratified random, 40% top-engagement, 20% novelty/outliers."""
    n_total = min(target_n, len(df))
    n_rand = int(n_total * split[0])
    n_top  = int(n_total * split[1])
    n_nov  = n_total - n_rand - n_top

    # De-duplication for sampling only (keep originals for analytics)
    df_samp = df.copy()
    df_samp['TEXT_NORM'] = df_samp['POST_TEXT'].str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    df_samp = df_samp.drop_duplicates(subset=['TEXT_NORM'])

    # Time bins
    bins = assign_time_bins(df_samp)
    df_samp = df_samp.assign(_bin=bins)
    bin_values = df_samp['_bin'].unique().tolist()
    bin_values.sort()

    # Random within bins
    rand_idx = []
    per_bin = max(1, n_rand // max(1, len(bin_values)))
    for b in bin_values:
        idx = df_samp[df_samp['_bin']==b].sample(min(per_bin, sum(df_samp['_bin']==b)), random_state=random_state, replace=False).index.tolist()
        rand_idx.extend(idx)
    rand_idx = rand_idx[:n_rand]

    # Top engagement within bins
    df_samp['ENG_SCORE'] = compute_engagement_score(df_samp)
    remaining = df_samp.drop(index=rand_idx, errors='ignore')
    top_idx = []
    per_bin_top = max(1, n_top // max(1, len(bin_values)))
    for b in bin_values:
        sub = remaining[remaining['_bin']==b].sort_values('ENG_SCORE', ascending=False)
        idx = sub.head(per_bin_top).index.tolist()
        top_idx.extend(idx)
    top_idx = top_idx[:n_top]

    # Novelty selection on remaining
    remaining2 = remaining.drop(index=top_idx, errors='ignore')
    nov_idx = select_novelty_indices(remaining2, n_nov, random_state=random_state)

    # Combine & pad if needed
    chosen = list(dict.fromkeys(rand_idx + top_idx + nov_idx))
    if len(chosen) < n_total:
        # Fill with random leftovers
        leftovers = df_samp.drop(index=chosen, errors='ignore')
        fill = leftovers.sample(n=min(n_total-len(chosen), len(leftovers)),
                                random_state=random_state, replace=False).index.tolist()
        chosen += fill
    return chosen[:n_total]

def select_novelty_indices(df_remain: pd.DataFrame, k: int, random_state:int=42) -> list:
    if k <= 0 or df_remain.empty:
        return []
    docs = df_remain['POST_TEXT'].fillna("").astype(str).tolist()
    # Cap docs for TF-IDF
    if len(docs) > NOVELTY_TFIDF_MAX_DOCS:
        df_remain = df_remain.sample(NOVELTY_TFIDF_MAX_DOCS, random_state=random_state)
        docs = df_remain['POST_TEXT'].fillna("").astype(str).tolist()
    # Vectorize
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(docs)
    n_clusters = max(10, min(50, X.shape[0] // 100))  # heuristic
    if n_clusters <= 1:
        # pick shortest/longest texts for diversity if clustering is trivial
        lengths = df_remain['POST_TEXT'].str.len().fillna(0)
        idx_short = lengths.nsmallest(max(1, k//2)).index.tolist()
        idx_long  = lengths.nlargest(k - len(idx_short)).index.tolist()
        return (idx_short + idx_long)[:k]
    km = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, n_init=3, batch_size=1024)
    labels = km.fit_predict(X)
    df_tmp = df_remain.copy()
    df_tmp['_cluster'] = labels
    # Rank clusters by size ascending (smallest = novel)
    sizes = df_tmp['_cluster'].value_counts().sort_values(ascending=True)
    select = []
    for cid in sizes.index:
        cand = df_tmp[df_tmp['_cluster']==cid].index.tolist()
        take = min(max(1, k // len(sizes)), len(cand))
        select.extend(cand[:take])
        if len(select) >= k: break
    return select[:k]

# --- Cluster → label: build clusters and ask LLM to label each cluster only ---

def _tfidf_features(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(texts)
    return tfidf, X

def build_cluster_views(df: pd.DataFrame, index_list: list[int], k: int) -> list[dict]:
    """
    Cluster a subset of posts and return a list of cluster dicts:
      { 'post_indices': [abs_idx...], 'top_terms': [...], 'exemplars': [abs_idx...] }
    """
    sub = df.loc[index_list]
    texts = sub['POST_TEXT'].astype(str).tolist()
    if len(texts) < k:
        k = max(2, min(len(texts), CLUSTER_K_MIN))

    tfidf, X = _tfidf_features(texts)
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=1024)
    labels = km.fit_predict(X)

    # top terms per cluster from centroids
    terms = np.array(tfidf.get_feature_names_out())
    centroids = km.cluster_centers_
    clusters = []
    for cid in range(k):
        mask = (labels == cid)
        if not mask.any(): 
            continue
        cluster_abs_idx = list(sub.index[mask])
        # top 10 terms
        centroid = centroids[cid]
        top_idx = np.argsort(centroid)[-10:][::-1]
        top_terms = terms[top_idx].tolist()
        # pick exemplars: closest to centroid
        # cosine distance ~ 1 - cosine_similarity
        Xc = X[mask]
        sim = cosine_similarity(Xc, centroids[cid].reshape(1, -1)).ravel()
        order = np.argsort(sim)[::-1]
        chosen = order[:CLUSTER_EXEMPLARS]
        exemplars = [cluster_abs_idx[i] for i in chosen]
        clusters.append({
            'post_indices': cluster_abs_idx,
            'top_terms': top_terms,
            'exemplars': exemplars
        })
    return clusters

def label_cluster_with_llm(cluster: dict, df: pd.DataFrame, api_key: str) -> dict | None:
    """
    Ask LLM to label THIS cluster only, with strict mutual-exclusivity scaffolding.
    Returns a theme dict with fields used later in QA and classification.
    """
    ex_texts = df.loc[cluster['exemplars'], 'POST_TEXT'].astype(str).tolist()
    ex_json = json.dumps(ex_texts[:CLUSTER_EXEMPLARS], ensure_ascii=False)
    terms_json = json.dumps(cluster['top_terms'], ensure_ascii=False)

    system_prompt = (
        "You label ONE cluster of posts into a distinct sub-theme. "
        "Your theme MUST NOT be an umbrella that could subsume other plausible sub-themes. "
        "Return JSON with: narrative_title (specific), summary (2 sentences), "
        "inclusion_rule (1 sentence decision rule), exclusion_rules (array of 1–3 rules naming what to avoid), "
        "representative_terms (3–6 phrases), positive_examples (1–2 snippets from provided), near_miss_examples (1–2 snippets). "
        "Do not invent content beyond the provided examples."
    )
    user_prompt = (
        f"Cluster top terms:\n{terms_json}\n\n"
        f"Cluster exemplar posts (JSON array):\n{ex_json}\n\n"
        "Return a single JSON object with the specified keys."
    )
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role":"system","content": system_prompt},
            {"role":"user","content": user_prompt}
        ],
        "generationConfig": {
            "responseMimeType":"application/json",
            "responseSchema": {
                "type":"OBJECT",
                "properties":{
                    "narrative_title":{"type":"STRING"},
                    "summary":{"type":"STRING"},
                    "inclusion_rule":{"type":"STRING"},
                    "exclusion_rules":{"type":"ARRAY","items":{"type":"STRING"}},
                    "representative_terms":{"type":"ARRAY","items":{"type":"STRING"}},
                    "positive_examples":{"type":"ARRAY","items":{"type":"STRING"}},
                    "near_miss_examples":{"type":"ARRAY","items":{"type":"STRING"}}
                },
                "required":["narrative_title","summary","inclusion_rule","exclusion_rules","representative_terms"]
            }
        },
        "temperature": 0.4
    }
    resp = call_grok_with_backoff(payload, api_key)
    if not resp:
        return None
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError:
        return None
    # minimal sanity
    title = str(obj.get("narrative_title","")).strip()
    if not title:
        return None
    # pack theme
    return {
        "narrative_title": title,
        "summary": str(obj.get("summary","")).strip(),
        "inclusion_rule": str(obj.get("inclusion_rule","")).strip(),
        "exclusion_rules": obj.get("exclusion_rules", []) or [],
        "representative_terms": obj.get("representative_terms", []) or [],
        "positive_examples": obj.get("positive_examples", []) or [],
        "near_miss_examples": obj.get("near_miss_examples", []) or []
    }

# ---------------------------
# Map-Reduce theme extraction
# ---------------------------
def extract_themes_map_reduce(df: pd.DataFrame, indices: list[int], batch_size:int, api_key:str) -> list[dict]:
    """
    Map: split indices into batches; for each batch, cluster posts and label each cluster (LLM).
    Reduce: merge near-duplicates across batches, then audit & refine.
    """
    all_themes = []

    # Map
    for i in range(0, len(indices), batch_size):
        sub_idx = indices[i:i+batch_size]
        # Pick cluster count dynamically: ~1 theme per 60–80 posts, bounded
        k_guess = int(np.clip(round(len(sub_idx) / 70), CLUSTER_K_MIN, CLUSTER_K_MAX))
        clusters = build_cluster_views(df, sub_idx, k_guess)

        # Label clusters concurrently
        out = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = [ex.submit(label_cluster_with_llm, c, df, api_key) for c in clusters]
            for fut in concurrent.futures.as_completed(futures):
                theme = fut.result()
                if theme:
                    out.append(theme)
        if out:
            all_themes.extend(out)

    if not all_themes:
        return []

    # Reduce v1: merge close duplicates by title tokens / fuzzy ratio (reuse your existing merge)
    merged = merge_theme_lists([all_themes], max_out=20)  # temporary max

    # Reduce v2: audit + refine
    refined = audit_and_refine_themes(merged, api_key)
def theme_similarity_df(themes: list[dict]) -> pd.DataFrame:
    """
    Build a cosine-similarity matrix between themes using their title/summary/rules/examples.
    Requires _theme_vector_strings(...) and _tfidf_features(...) to be defined above.
    """
    if not themes or len(themes) < 2:
        return pd.DataFrame()

    blobs = _theme_vector_strings(themes)          # from Patch 5
    tfidf, X = _tfidf_features(blobs)              # reuses your TF-IDF helper
    sims = cosine_similarity(X)
    titles = [t.get('narrative_title', '') for t in themes]

    return pd.DataFrame(sims, index=titles, columns=titles)

def _theme_vector_strings(themes: list[dict]) -> list[str]:
    """Flatten each theme into a text string for vectorization."""
    blobs = []
    for t in themes:
        parts = [
            t.get('narrative_title',''),
            t.get('summary',''),
            t.get('inclusion_rule',''),
            " ".join(t.get('exclusion_rules',[]) or []),
            " ".join(t.get('representative_terms',[]) or []),
            " ".join(t.get('positive_examples',[]) or []),
        ]
        blobs.append(" ".join([p for p in parts if p]))
    return blobs

def refine_pair_with_llm(A: dict, B: dict, api_key: str) -> dict | tuple[dict, dict] | None:
    """
    Ask LLM to decide: keep A, keep B, or MERGE into a sharper single theme.
    Returns: merged dict OR (A_refined, B_refined) OR None on failure.
    """
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role":"system","content":
             "You are refining two theme definitions. **Your primary goal is to identify distinct, high-level subtopics.** "
             "If themes are about the same core event, **MERGE** them aggressively into one sharper, more comprehensive theme. "
             "Only **KEEP BOTH** if they cover two entirely different aspects (e.g., 'Financial Impact' vs. 'Cultural Backlash'). "
             "Ensure final themes are MUTUALLY EXCLUSIVE."
             "Return JSON with either {'action':'merge', 'theme':{...}} or "
             "{'action':'keep_both','theme_a':{...},'theme_b':{...}}. "
             "Each theme object must have: narrative_title, summary, inclusion_rule, exclusion_rules (array), representative_terms (array)."},
        ],
        "generationConfig": {
            "responseMimeType":"application/json",
            "responseSchema": {"type":"OBJECT"}
        },
        "temperature": 0.2
    }
    resp = call_grok_with_backoff(payload, api_key)
    if not resp: 
        return None
    try:
        obj = json.loads(resp)
    except json.JSONDecodeError:
        return None
    action = (obj.get("action") or "").lower()
    if action == "merge" and isinstance(obj.get("theme"), dict):
        return obj["theme"]
    if action == "keep_both" and isinstance(obj.get("theme_a"), dict) and isinstance(obj.get("theme_b"), dict):
        return (obj["theme_a"], obj["theme_b"])
    return None

def audit_and_refine_themes(themes: list[dict], api_key: str) -> list[dict]:
    """Flag near-duplicate themes and refine via LLM; fallback to heuristic merge if LLM fails."""
    if len(themes) <= 1:
        return themes

    blobs = _theme_vector_strings(themes)
    tfidf, X = _tfidf_features(blobs)
    sims = cosine_similarity(X)

    keep = themes.copy()
    removed = set()
    # Greedy pass: compare upper triangle
    for i in range(len(keep)):
        if i in removed: 
            continue
        for j in range(i+1, len(keep)):
            if j in removed: 
                continue
            s = sims[i, j]
            if s >= THEME_SIM_THRESHOLD:
                # try refine
                ref = refine_pair_with_llm(keep[i], keep[j], api_key)
                if isinstance(ref, dict):
                    keep[i] = ref
                    removed.add(j)
                elif isinstance(ref, tuple) and len(ref) == 2:
                    keep[i], keep[j] = ref
                else:
                    # fallback: keep the more specific (longer inclusion rule)
                    len_i = len(keep[i].get('inclusion_rule',''))
                    len_j = len(keep[j].get('inclusion_rule',''))
                    if len_i >= len_j:
                        removed.add(j)
                    else:
                        removed.add(i)
                        break
    refined = [t for k, t in enumerate(keep) if k not in removed]
    return refined


# ---------------------------
# Session State
# ---------------------------
if 'df_full' not in st.session_state: st.session_state.df_full = None
if 'narrative_data' not in st.session_state: st.session_state.narrative_data = None
if 'theme_titles' not in st.session_state: st.session_state.theme_titles = []
if 'theme_explanations' not in st.session_state: st.session_state.theme_explanations = None
if 'classified_df' not in st.session_state: st.session_state.classified_df = None
if 'takeaways_list' not in st.session_state: st.session_state.takeaways_list = None
if 'sampling_indices' not in st.session_state: st.session_state.sampling_indices = None
if 'refresh_done' not in st.session_state: st.session_state.refresh_done = 0

# API key
XAI_KEY = os.getenv(XAI_API_KEY_ENV_VAR)
st.session_state.api_key = XAI_KEY

# Title
st.title("X Narrative Analysis Dashboard")
st.markdown("Automated thematic extraction and quantitative analysis of Meltwater data.")

# Sidebar
with st.sidebar:
    if not st.session_state.api_key:
        st.error(f"FATAL ERROR: Grok API Key not found. Please set '{XAI_API_KEY_ENV_VAR}'.")
    uploaded_file = st.file_uploader(
        "Upload Meltwater Data (.xlsx or .csv)",
        type=['xlsx','csv'],
        help="Upload your Meltwater export as Excel (.xlsx) or CSV (.csv)."
    )
    if st.session_state.classified_df is not None and not st.session_state.classified_df.empty:
        st.markdown("---")
        csv = st.session_state.classified_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Tagged Data (CSV)", data=csv,
                           file_name='meltwater_narrative_analysis.csv',
                           mime='text/csv', type="primary")

# Early exit
if not st.session_state.api_key or uploaded_file is None:
    if uploaded_file is None:
        st.info("Upload your Meltwater Data (.xlsx or .csv) in the sidebar to begin.")
    st.stop()

# ---------------------------
# Main Auto-Chain Pipeline
# ---------------------------
with st.container():
    # 0) Load once
    if st.session_state.df_full is None:
        try:
            st.info("Loading file...")
            df = load_meltwater(uploaded_file)
            st.session_state.df_full = df.copy()

            # Summary
            data_rows = df.shape[0]
            valid_dates = df['DATETIME'].dropna()
            date_min = valid_dates.min().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"
            date_max = valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"
            st.success("File uploaded successfully!")
            st.markdown("#### File Data Summary")
            st.markdown(f"- **Total Rows Processed:** {data_rows:,}\n- **Date Span:** {date_min} to {date_max}")
            st.markdown("---")

            # Build stratified mixed sampling indices
            st.info("Building stratified sampling (random/top-engagement/novelty)...")
            st.session_state.sampling_indices = stratified_mixed_sample_indices(
                st.session_state.df_full,
                target_n=min(THEMES_TOTAL_SAMPLE_TARGET, len(st.session_state.df_full))
            )

            st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.stop()

    # 1) Extract themes via map-reduce (auto)
    if st.session_state.df_full is not None and st.session_state.narrative_data is None:
        themes_merged = extract_themes_map_reduce(
            st.session_state.df_full,
            st.session_state.sampling_indices,
            batch_size=BATCH_SIZE_FOR_THEMES,
            api_key=st.session_state.api_key
        )
        if themes_merged:
            # Run a second audit just to be safe (idempotent)
            audited = audit_and_refine_themes(themes_merged, st.session_state.api_key)
            st.session_state.narrative_data = audited
            st.session_state.theme_titles = [t['narrative_title'] for t in audited if t.get('narrative_title')]
            st.rerun() # Ensure this is also indented if you want it conditional

    # 1a) Theme explanations (auto)
    if st.session_state.narrative_data is not None and st.session_state.theme_explanations is None:
        # corpus sample from selected indices
        corpus_sample = ' | '.join(st.session_state.df_full.loc[st.session_state.sampling_indices, 'POST_TEXT'].astype(str).tolist())
        st.session_state.theme_explanations = generate_theme_explanations(corpus_sample, st.session_state.theme_titles, st.session_state.api_key)
        st.rerun()

# Present themes + explanations
    if st.session_state.narrative_data is not None:
        st.header("Narratives Extraction")
        st.subheader("Identified Narrative Themes")
        exp_map = {d.get('narrative_title',''): d.get('explanation','') for d in (st.session_state.theme_explanations or [])}
        
        # All lines below this MUST be indented by 4 spaces to be inside the loop
        for i, narrative in enumerate(st.session_state.narrative_data):
            title = narrative.get('narrative_title', f"Theme {i+1}")
            st.markdown(f"**{i+1}. {title}**")
            
            # Prefer explicit explanation if you generated them; else show theme summary
            expl = (st.session_state.theme_explanations and
                    next((e.get('explanation') for e in st.session_state.theme_explanations if e.get('narrative_title')==title), "")) or ""
            summary = narrative.get('summary','')
            
            if expl:
                st.markdown(expl)
            elif summary:
                st.markdown(summary)
                
            # NEW: show rules to make distinctness visible (first instance)
            inc = narrative.get('inclusion_rule','')
            exc = narrative.get('exclusion_rules',[]) or []
            if inc:
                st.caption(f"_Inclusion rule_: {inc}")
            if exc:
                st.caption(f"_Exclusion rules_: {', '.join(exc[:3])}")
                
            # (NOTE: This second block is a DUPLICATE and likely unnecessary, 
            # but is kept here and indented correctly to fix the syntax error.)
            # NEW: show rules to make distinctness visible (duplicate instance)
            inc = narrative.get('inclusion_rule','')
            exc = narrative.get('exclusion_rules',[]) or []
            if inc:
                st.caption(f"_Inclusion rule_: {inc}")
            if exc:
                st.caption(f"_Exclusion rules_: {', '.join(exc[:3])}")
# --- Theme QA: similarity heatmap and top overlaps ---
simdf = theme_similarity_df(st.session_state.narrative_data)
if not simdf.empty:
    st.subheader("Theme QA: Similarity Heatmap")
    fig_sim = px.imshow(simdf, text_auto=False, aspect="auto",
                        title="Cosine similarity between themes (higher = more overlap)")
    fig_sim.update_layout(margin=dict(l=40, r=20, t=50, b=40))
    st.plotly_chart(
    finalize_figure(fig_sim, "Theme overlap diagnostics"),
    use_container_width=True,
    config={"displayModeBar": False},
    # This is the patch: use a simple, unique string
    key="theme_overlap_fig_sim_chart" 
)

    # list the top 5 most similar pairs (excluding diagonal)
    pairs = []
    for i in range(len(simdf)):
        for j in range(i + 1, len(simdf)):
            pairs.append((simdf.index[i], simdf.columns[j], simdf.iloc[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = [p for p in pairs if p[2] >= 0.50][:5]
    if top_pairs:
        st.markdown("**Most similar theme pairs (flagged for review):**")
        for a, b, s in top_pairs:
            st.markdown(f"- {a} ⟷ {b}: {s:.2f}")

    # --- Theme QA: similarity heatmap and top overlaps ---
simdf = theme_similarity_df(st.session_state.narrative_data)
if not simdf.empty:
    st.subheader("Theme QA: Similarity Heatmap")
    fig_sim = px.imshow(simdf, text_auto=False, aspect="auto", title="Cosine similarity between themes (higher = more overlap)")
    fig_sim.update_layout(margin=dict(l=40,r=20,t=50,b=40))
    st.plotly_chart(finalize_figure(fig_sim, "Theme overlap diagnostics"), use_container_width=True, config={"displayModeBar": False})
    # list the top 5 most similar pairs (excluding diagonal)
    pairs = []
    for i in range(len(simdf)):
        for j in range(i+1, len(simdf)):
            pairs.append((simdf.index[i], simdf.columns[j], simdf.iloc[i,j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = [p for p in pairs if p[2] >= 0.50][:5]
    if top_pairs:
        st.markdown("**Most similar theme pairs (flagged for review):**")
        for a,b,s in top_pairs:
            st.markdown(f"- {a} ⟷ {b}: {s:.2f}")
        st.markdown("---")
        st.success("Themes identified. Proceeding to tagging and analytics...")

    # 2) Classification (auto)
    if (st.session_state.narrative_data is not None
        and st.session_state.classified_df is None
        and st.session_state.theme_titles):
        df_classified = train_and_classify_hybrid(st.session_state.df_full, st.session_state.theme_titles, st.session_state.api_key)
        if df_classified is not None:
            st.session_state.classified_df = df_classified
        st.rerun()

    # 2b) Iterative refresh if Other too high (one round)
    if (st.session_state.classified_df is not None
        and st.session_state.refresh_done < MAX_REFRESH_ROUNDS):
        other_rate = (st.session_state.classified_df['NARRATIVE_TAG'].eq(CLASSIFICATION_DEFAULT)).mean()
        if other_rate > OTHER_RATE_THRESHOLD:
            st.warning(f"'Other' is {other_rate*100:.1f}% > {OTHER_RATE_THRESHOLD*100:.0f}%. Running a targeted refresh on 'Other' posts.")
            # sample from 'Other' for new themes
            df_other = st.session_state.classified_df[st.session_state.classified_df['NARRATIVE_TAG']==CLASSIFICATION_DEFAULT]
            if not df_other.empty:
                targets = stratified_mixed_sample_indices(df_other, target_n=min(400, len(df_other)))
                # indices here are relative to df_other; convert to df_full indices
                other_abs_idx = df_other.iloc[targets].index.tolist()
                new_themes = extract_themes_map_reduce(st.session_state.df_full, other_abs_idx, batch_size=300, api_key=st.session_state.api_key)
                if new_themes:
                    # merge with existing themes
                    merged = merge_theme_lists([st.session_state.narrative_data, new_themes], max_out=14)
                    st.session_state.narrative_data = merged
                    st.session_state.theme_titles = [t['narrative_title'] for t in merged if t.get('narrative_title')]
                    # reclassify with expanded taxonomy
                    df_re = train_and_classify_hybrid(st.session_state.df_full, st.session_state.theme_titles, st.session_state.api_key)
                    if df_re is not None:
                        st.session_state.classified_df = df_re
            st.session_state.refresh_done += 1
            st.rerun()

    # 3) Visualizations & Insights (auto)
    if st.session_state.classified_df is not None and not st.session_state.classified_df.empty:
        st.header("Data Analysis by Narrative")
        dfc = st.session_state.classified_df

        # Filter for viz: exclude Other & require dates
        df_viz = dfc[(dfc['NARRATIVE_TAG'] != CLASSIFICATION_DEFAULT) & (dfc['DATETIME'].notna())].copy()

        if df_viz.empty:
            st.warning("No posts were classified into primary themes OR no valid dates. Dashboard cannot be generated.")
        else:
            # 1) Volume by Theme
            st.markdown("### Post Volume by Theme")
            theme_metrics = df_viz.groupby('NARRATIVE_TAG').agg(Post_Volume=('POST_TEXT','size')).reset_index()
            theme_metrics = theme_metrics.sort_values('Post_Volume', ascending=False)
            fig_bar = px.bar(theme_metrics, x='NARRATIVE_TAG', y='Post_Volume',
                             labels={'Post_Volume':'Post Volume (Count)','NARRATIVE_TAG':'Narrative Theme'},
                             height=500, color='NARRATIVE_TAG', color_discrete_sequence=VIBRANT_QUAL)
            def wrap_labels_bar(text): return '<br>'.join(textwrap.wrap(text, 15))
            fig_bar.update_traces(marker_line_width=0)
            fig_bar.update_layout(
                xaxis={'categoryorder':'total descending','tickangle':0,'automargin':True,
                       'tickvals': theme_metrics['NARRATIVE_TAG'].tolist(),
                       'ticktext':[wrap_labels_bar(t) for t in theme_metrics['NARRATIVE_TAG']]},
                showlegend=False
            )
            fig_bar = finalize_figure(fig_bar, "Post volume by narrative theme", "Sorted by total posts", 500)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # 2) Trend over time (adaptive ticks)
            st.markdown("### Narrative Volume Trend Over Time")
            df_trends = (df_viz.groupby([df_viz['DATETIME'].dt.date,'NARRATIVE_TAG'])
                         .size().reset_index(name='Post_Volume'))
            df_trends['DATETIME'] = pd.to_datetime(df_trends['DATETIME'])
            if not df_trends.empty and df_trends['DATETIME'].dt.date.nunique() > 1:
                fig_line = px.line(df_trends, x='DATETIME', y='Post_Volume', color='NARRATIVE_TAG',
                                   labels={'Post_Volume':'Daily Post Volume','DATETIME':'Date','NARRATIVE_TAG':'Theme'},
                                   height=550, color_discrete_sequence=VIBRANT_QUAL)
                fig_line.update_traces(mode="lines", line={"width":3})
                fig_line.update_layout(xaxis_title="Date", yaxis_title="Daily posts",
                                       legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5))
                fig_line = configure_time_axis(fig_line, df_trends['DATETIME'])
                fig_line = finalize_figure(fig_line, "Narrative volume over time", "Daily volume, by theme", 520)
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Trend chart requires posts across at least two unique days.")

            # 3) Toxicity density
            st.markdown("### Toxicity Density by Narrative Theme")
            df_viz, theme_tox = compute_toxicity_scores(df_viz)
            fig_tox = plot_toxicity_by_theme(theme_tox)
            st.plotly_chart(fig_tox, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Based on {len(TOXIC_KEYWORDS)} keywords; density = toxic matches / total words per post.")

            # 4) Theme-specific top authors
            st.markdown("### Influencer Share of Engagement (Top Authors Only)")
            for theme in df_viz['NARRATIVE_TAG'].unique():
                fig_theme = plot_theme_influencer_share(df_viz, theme, AUTHOR_COLUMN, ENGAGEMENT_COLUMN, top_n=5)
                if fig_theme:
                    st.plotly_chart(fig_theme, use_container_width=True, config={"displayModeBar": False})

            # 5) Overall top authors
            st.markdown("### Top 10 Overall Authors by Total Likes")
            fig_overall = plot_overall_author_ranking(dfc, AUTHOR_COLUMN, ENGAGEMENT_COLUMN, top_n=10)
            st.plotly_chart(fig_overall, use_container_width=True, config={"displayModeBar": False})

        # Insights
        st.markdown("---")
        st.header("Insights from the Data")
        if st.session_state.takeaways_list is None:
            summary = perform_data_crunching_and_summary(st.session_state.classified_df)
            st.session_state.takeaways_list = generate_takeaways(summary, st.session_state.api_key)
            st.rerun()

        if st.session_state.takeaways_list:
            st.subheader("Executive Summary: 5 Key Takeaways")
            for i, t in enumerate(st.session_state.takeaways_list):
                st.markdown(f"**{i+1}.** {t}")

# Footer credit
st.markdown("---")
st.caption("_Developed by Trish Bailey_")

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

def generate_theme_explanations(corpus_sample: str, theme_titles: list[str], api_key: str):
    if not theme_titles: return []
    titles_json = json.dumps(theme_titles, ensure_ascii=False)
    system_prompt = (
        "You are an experienced OSINT analyst. Based ONLY on the provided corpus sample, "
        "write concise 2–3 sentence factual explanations for each theme title. "
        "Return a JSON array of objects with keys narrative_title and explanation. "
        "narrative_title must EXACTLY match each input title. No speculation."
    )
    user_query = f"Corpus sample:\n{corpus_sample}\n\nTheme titles (JSON): {titles_json}"
    payload = {
        "model": GROK_MODEL,
        "messages": [{"role":"system","content": system_prompt},
                     {"role":"user","content": user_query}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type":"ARRAY","items":{"type":"OBJECT","properties":{
                "narrative_title":{"type":"STRING"},"explanation":{"type":"STRING"}},
                "required":["narrative_title","explanation"]}}
        },
        "temperature": 0.3
    }
    with st.spinner("Drafting brief explanations for each theme..."):
        resp = call_grok_with_backoff(payload, api_key)
    if not resp: return []
    try:
        data = json.loads(resp)
    except json.JSONDecodeError:
        st.warning("Could not parse theme explanations JSON; skipping.")
        return []
    exp_map = {}
    if isinstance(data, list):
        for d in data:
            if isinstance(d, dict):
                t = str(d.get("narrative_title","")).strip()
                e = str(d.get("explanation","")).strip()
                if t and e: exp_map[t] = e
    return [{"narrative_title": t, "explanation": exp_map.get(t,"")} for t in theme_titles]

# ---------------------------
# Theme merging / dedup
# ---------------------------
_STOPWORDS = set("a an the of for to and or on in with by about over under through from into against across around".split())

def _normalize_title(s: str) -> set:
    s = s.lower()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    toks = [t for t in s.split() if t and t not in _STOPWORDS]
    return set(toks)

def _jaccard(a: set, b: set) -> float:
    if not a or not b: return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def merge_theme_lists(theme_lists: list[list[dict]], max_out: int = 12) -> list[dict]:
    """Merge and deduplicate theme dicts across batches via token Jaccard & simple similarity."""
    all_themes = []
    for L in theme_lists:
        if isinstance(L, list): all_themes.extend(L)

    merged = []
    sigs = []
    for t in all_themes:
        title = t.get('narrative_title','').strip()
        summ = t.get('summary','').strip()
        if not title: continue
        sig = _normalize_title(title)
        matched = False
        for i, s in enumerate(sigs):
            # consider similar if Jaccard >= 0.6 or SequenceMatcher >= 0.8
            if _jaccard(sig, s) >= 0.6 or SequenceMatcher(None, title.lower(), merged[i]['narrative_title'].lower()).ratio() >= 0.8:
                # merge summaries conservatively
                if summ and summ not in merged[i]['summary']:
                    merged[i]['summary'] = (merged[i]['summary'] + " " + summ).strip()
                matched = True
                break
        if not matched:
            merged.append({'narrative_title': title, 'summary': summ})
            sigs.append(sig)

    # limit to top N (keep order observed)
    return merged[:max_out]

# ---------------------------
# Hybrid classification
# ---------------------------
def classify_post_for_seed(post_text, themes_list, api_key):
    theme_options = ", ".join([f'"{t}"' for t in themes_list])
    system_prompt = (
        "You are an expert text classifier. Categorize the post into one of the provided themes, "
        f"or '{CLASSIFICATION_DEFAULT}' if none apply. Respond with ONLY the exact theme text."
    )
    user_query = f"Post: '{post_text}'\n\nThemes: [{theme_options}, '{CLASSIFICATION_DEFAULT}']"
    payload = {
        "model": CLASSIFICATION_MODEL,
        "messages": [{"role":"system","content": system_prompt},
                     {"role":"user","content": user_query}],
        "temperature": 0.1, "max_tokens": 50
    }
    r = call_grok_with_backoff(payload, api_key)
    if r:
        label = r.strip()
        if label in themes_list or label == CLASSIFICATION_DEFAULT:
            return label
    return CLASSIFICATION_DEFAULT

def train_and_classify_hybrid(df_full, theme_titles, api_key):
    st.info(f"Phase 1: Labeling {AI_SEED_SAMPLE_SIZE} examples with {CLASSIFICATION_MODEL}...")
    actual_seed = min(AI_SEED_SAMPLE_SIZE, len(df_full))
    df_sample = df_full.sample(actual_seed, random_state=42).copy()
    df_rest = df_full.drop(df_sample.index).copy()
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

    st.success("Phase 2: Training local TF-IDF/LinearSVC...")
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LinearSVC(C=1.0, random_state=42, max_iter=5000))
    ])
    model.fit(df_sample['POST_TEXT'], df_sample['NARRATIVE_TAG'])

    st.info(f"Phase 3: Classifying remaining {len(df_rest):,} posts...")
    df_rest['NARRATIVE_TAG'] = model.predict(df_rest['POST_TEXT'])
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

# ---------------------------
# Map-Reduce theme extraction
# ---------------------------
def extract_themes_map_reduce(df: pd.DataFrame, indices: list[int], batch_size:int, api_key:str) -> list[dict]:
    batches = []
    # Stable order for reproducibility
    idx = list(indices)
    for i in range(0, len(idx), batch_size):
        sub_idx = idx[i:i+batch_size]
        corpus = ' | '.join(df.loc[sub_idx, 'POST_TEXT'].astype(str).tolist())
        res = analyze_narratives(corpus, api_key)
        if res: batches.append(res)
    if not batches:
        return []
    merged = merge_theme_lists(batches, max_out=12)
    return merged

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
            st.session_state.narrative_data = themes_merged
            st.session_state.theme_titles = [t['narrative_title'] for t in themes_merged if t.get('narrative_title')]
        st.rerun()

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
        for i, narrative in enumerate(st.session_state.narrative_data):
            title = narrative.get('narrative_title', f"Theme {i+1}")
            short_summary = narrative.get('summary', '').strip()
            st.markdown(f"**{i+1}. {title}**")
            long_expl = exp_map.get(title, "")
            if long_expl:
                st.markdown(long_expl)
            elif short_summary:
                st.markdown(short_summary)
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

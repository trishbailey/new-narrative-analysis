import streamlit as st
import pandas as pd
import requests
import os
import json
import time
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
import concurrent.futures
import threading
import textwrap
import numpy as np

# ML Imports for Hybrid Classification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# --- Configuration ---
XAI_API_KEY_ENV_VAR = 'XAI_API_KEY'
GROK_MODEL = 'grok-4-fast-reasoning'
CLASSIFICATION_MODEL = 'grok-3-mini'  # Used only for the small AI seed
API_BASE_URL = "https://api.x.ai/v1/chat/completions"
MAX_POSTS_FOR_ANALYSIS = 100  # Fixed sample size for Theme Generation
AI_SEED_SAMPLE_SIZE = 50      # Fixed sample size for ML Training
CLASSIFICATION_DEFAULT = "Other/Unrelated"

# Confirmed by user: Columns start on the second row (index 1)
HEADER_ROWS_TO_SKIP = 1

# --- Meltwater Data Column Mapping (CORRECTED AND FINALIZED) ---
TEXT_COLUMNS = ['Opening Text', 'Headline', 'Hit Sentence']
ENGAGEMENT_COLUMN = 'Likes'
AUTHOR_COLUMN = 'Influencer'
DATE_COLUMN = 'Date'
TIME_COLUMN = 'Time'

# FIX: Explicit format confirmed to work for Meltwater date/time strings
DATE_TIME_FORMAT = '%d-%b-%Y %I:%M%p'

# --- Streamlit Theme Configuration (Light Mode) ---
PRIMARY_COLOR = "#1E88E5"   # Blue for primary buttons
BG_COLOR = "#ffffff"        # Main background
SIDEBAR_BG = "#f7f7f7"      # Sidebar background
TEXT_COLOR = "#333333"      # Main text
HEADER_COLOR = "#111111"    # Headers

# --- Page Config ---
st.set_page_config(
    page_title="Grok Narrative Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Grok-Powered Narrative Analyzer for Meltwater Data"}
)

# --- Global CSS (Light Mode UI + Inter font + extra spacing helpers) ---
st.markdown(
f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
/* Sidebar */
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    color: {TEXT_COLOR};
}}

/* App background + text */
.stApp {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}

/* Headers */
h1, h2, h3, h4, h5, h6 {{
    color: {HEADER_COLOR} !important;
}}

/* General text */
p, span, label, div {{
    color: {TEXT_COLOR} !important;
}}

/* Buttons (primary) */
.stButton > button.primary {{
    background-color: {PRIMARY_COLOR} !important;
    border-color: {PRIMARY_COLOR} !important;
    color: white !important;
}}
.stButton > button.primary:hover {{
    background-color: #1565C0 !important;
    border-color: #1565C0 !important;
}}

/* Also style any non-primary buttons for readability */
.stButton > button:not(.primary) {{
    color: {HEADER_COLOR} !important;
    border-color: #d0d0d0 !important;
    background-color: #fafafa !important;
}}

/* Alerts */
div[data-testid="stAlert"] * p, div[data-testid="stAlert"] * h5 {{
    color: {TEXT_COLOR} !important;
}}

/* Code snippets */
code {{
    background-color: #f2f2f2;
    color: #d63384;
    padding: 2px 4px;
    border-radius: 4px;
}}

/* Plotly background anchor */
.js-plotly-plot {{
    background-color: {BG_COLOR} !important;
}}

/* Chart captions (optional) */
.chart-subtitle {{
  font-size: 0.95rem; 
  color: #6b7280;
  margin-top: -10px;
  margin-bottom: 12px;
}}
.chart-source {{
  font-size: 0.85rem; 
  color: #9ca3af; 
  margin-top: 8px;
}}
/* Spacer below charts to avoid collisions with following text */
.chart-spacer {{ height: 18px; }}
</style>
""",
    unsafe_allow_html=True
)

# --- Vibrant Plotly Template (rich colorway + editorial polish + bigger bottom margin) ---
VIBRANT_QUAL = [
    "#1E88E5",  # vivid blue
    "#E53935",  # crimson
    "#8E24AA",  # royal purple
    "#43A047",  # emerald
    "#FB8C00",  # amber
    "#00ACC1",  # cyan
    "#F4511E",  # burnt orange
    "#3949AB",  # indigo
    "#6D4C41",  # cocoa
    "#7CB342",  # olive
    "#D81B60",  # magenta
    "#00897B",  # teal
]

# Base template
vibrant_layout = pio.templates["plotly_white"].layout.to_plotly_json()
vibrant_layout.update({
    "font": {"family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "size": 14, "color": "#111111"},
    "colorway": VIBRANT_QUAL,
    "title": {"x": 0.0, "xanchor": "left", "y": 0.95, "yanchor": "top", "font": {"size": 20}},
    "margin": {"l": 80, "r": 40, "t": 60, "b": 110},  # larger bottom margin to protect legend/source
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "xaxis": {
        "showline": False, "showgrid": True, "gridcolor": "#e9eaec", "gridwidth": 1,
        "tickformat": ",", "title": {"standoff": 10, "font": {"size": 14, "color": "#374151"}},
        "ticks": "", "tickfont": {"size": 12, "color": "#111111"}
    },
    "yaxis": {
        "showline": False, "showgrid": True, "gridcolor": "#e9eaec", "gridwidth": 1,
        "tickformat": ",", "title": {"standoff": 10, "font": {"size": 14, "color": "#374151"}},
        "ticks": "", "tickfont": {"size": 12, "color": "#111111"}, "automargin": True
    },
    # Legend at TOP to prevent collision with text below charts
    "legend": {
        "orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0,
        "title": {"text": "", "font": {"size": 12, "color": "#374151"}},
        "font": {"size": 12, "color": "#111111"},
        "itemclick": "toggleothers"
    },
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": "white", "bordercolor": "#d1d5db", "font": {"family": "Inter", "size": 12, "color": "#111111"}}
})
pio.templates["vibrant"] = pio.templates["plotly_white"]
pio.templates["vibrant"].layout.update(vibrant_layout)

# --- Helpers for presentation (spacing-safe) ---
def finalize_figure(fig, title:str, subtitle:str|None=None, source:str|None=None, height:int|None=None):
    fig.update_layout(template="vibrant", title={"text": title})
    if height:
        fig.update_layout(height=height)
    # subtitle above figure
    if subtitle:
        fig.add_annotation(
            text=f"<span style='color:#6b7280'>{subtitle}</span>",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False, align="left"
        )
    # source placed safely within the larger bottom margin
    if source:
        fig.add_annotation(
            text=f"<span style='font-size:12px;color:#9ca3af'>Source: {source}</span>",
            xref="paper", yref="paper", x=0, y=-0.18, showarrow=False, align="left"
        )
    return fig

def wrap_text(s: str, width: int = 16) -> str:
    return "<br>".join(textwrap.wrap(s, width))

# --- Utility Functions ---
def call_grok_with_backoff(payload, api_key, max_retries=5):
    """Handles POST request to Grok API with error handling and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=300)
            response.raise_for_status()
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'):
                return result['choices'][0]['message']['content']
            else:
                st.error("Error: Unexpected API response structure.")
                return None
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
            return None
    st.error("Max retries reached. API call failed.")
    return None

def classify_post_for_seed(post_text, themes_list, api_key):
    """Classifies a single post against a list of themes for training data."""
    theme_options = ", ".join([f'"{t}"' for t in themes_list])
    system_prompt = (
        "You are an expert text classifier. Your task is to categorize a social media post into one "
        "of the following narrative themes. Only respond with the EXACT text of the matching theme "
        f"or with '{CLASSIFICATION_DEFAULT}' if no theme is relevant. DO NOT add any other text, explanation, or punctuation."
    )
    user_query = (
        f"Post to classify: '{post_text}'\n\n"
        f"Available Themes: [{theme_options}, '{CLASSIFICATION_DEFAULT}']"
    )
    payload = {
        "model": CLASSIFICATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "temperature": 0.1,
        "max_tokens": 50
    }
    response_text = call_grok_with_backoff(payload, api_key)
    if response_text and response_text.strip() in themes_list or response_text.strip() == CLASSIFICATION_DEFAULT:
        return response_text.strip()
    return CLASSIFICATION_DEFAULT

def analyze_narratives(corpus, api_key):
    """Calls Grok to generate narrative themes and summaries (Step 1)"""
    system_prompt = (
        "You are a world-class social media narrative analyst. Your task is to analyze the provided corpus of "
        "social media posts (from Meltwater) and identify the 3 to 5 most significant, high-level, emerging "
        "discussion themes or sub-narratives. Output a single JSON array of objects."
    )
    user_query = f"Analyze the following combined text corpus for emerging narratives: {corpus}"
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "narrative_title": {"type": "STRING"},
                        "summary": {"type": "STRING"},
                    },
                    "propertyOrdering": ["narrative_title", "summary"]
                }
            }
        },
        "temperature": 0.5,
    }
    with st.spinner("Hold on, we are generating your narratives. They should be ready in about 60 seconds."):
        json_response = call_grok_with_backoff(payload, api_key)
    if json_response:
        try:
            return json.loads(json_response)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response from Grok. Check the API output.")
            st.code(json_response)
            return None
    return None

def train_and_classify_hybrid(df_full, theme_titles, api_key):
    """Hybrid Classification: Grok labels seed, then ML model classifies the rest."""
    st.info(f"Phase 1: Generating {AI_SEED_SAMPLE_SIZE} labeled training examples using {CLASSIFICATION_MODEL}...")
    actual_seed_size = min(AI_SEED_SAMPLE_SIZE, len(df_full))
    df_sample = df_full.sample(actual_seed_size, random_state=42).copy()
    df_remaining = df_full.drop(df_sample.index).copy()
    seed_tags = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_post = {
            executor.submit(classify_post_for_seed, post_text, theme_titles, api_key): post_text
            for post_text in df_sample['POST_TEXT']
        }
        progress_bar = st.progress(0)
        for i, future in enumerate(concurrent.futures.as_completed(future_to_post)):
            seed_tags.append(future.result())
            progress_bar.progress((i + 1) / actual_seed_size)
    df_sample['NARRATIVE_TAG'] = seed_tags
    progress_bar.empty()

    if df_sample['NARRATIVE_TAG'].nunique() < 2:
        st.error("AI failed to generate diverse enough labels for training. Check generated themes or try increasing AI Seed Sample Size.")
        return None

    st.success(f"Phase 1 Complete: {len(df_sample)} examples labeled by Grok.")

    st.info("Phase 2: Training local TF-IDF/LinearSVC model on Grok's labels...")
    X_train = df_sample['POST_TEXT']
    y_train = df_sample['NARRATIVE_TAG']
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LinearSVC(C=1.0, random_state=42, max_iter=5000))
    ])
    model.fit(X_train, y_train)
    st.success("Phase 2 Complete: Local ML Model Trained.")

    st.info(f"Phase 3: Classifying remaining {len(df_remaining):,} posts with local model...")
    X_test = df_remaining['POST_TEXT']
    df_remaining['NARRATIVE_TAG'] = model.predict(X_test)
    st.success("Phase 3 Complete: Full dataset classification finished.")

    df_classified = pd.concat([df_sample, df_remaining])
    return df_classified

def perform_data_crunching_and_summary(df_classified: pd.DataFrame) -> str:
    """Aggregations for the executive takeaways prompt."""
    theme_metrics = df_classified.groupby('NARRATIVE_TAG').agg(
        Volume=('POST_TEXT', 'size'),
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum')
    ).reset_index()
    theme_metrics['Avg_Likes_Per_Post'] = theme_metrics['Total_Likes'] / theme_metrics['Volume']
    theme_metrics = theme_metrics.sort_values(by='Volume', ascending=False)

    narrative_summary = "Narrative Metrics (Volume, Total Likes, Avg Likes Per Post):\n"
    narrative_summary += theme_metrics.to_string(index=False, float_format="%.2f") + "\n\n"

    overall_top_authors = df_classified.groupby(AUTHOR_COLUMN).agg(
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum'),
        Post_Count=('POST_TEXT', 'size')
    ).nlargest(3, 'Total_Likes').reset_index()
    author_summary = "Overall Top 3 Influencers (by Total Likes):\n"
    author_summary += overall_top_authors.to_string(index=False) + "\n\n"

    valid_dates = df_classified['DATETIME'].dropna()
    if not valid_dates.empty:
        start_date = valid_dates.min().strftime('%Y-%m-%d')
        end_date = valid_dates.max().strftime('%Y-%m-%d')
    else:
        start_date = "N/A (No valid dates found)"
        end_date = "N/A (No valid dates found)"

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
    """Calls Grok to generate 5 key data-driven takeaways (Step 3)"""
    system_prompt = (
        "You are a senior social media analyst reporting to C-suite executives. Provide five concise, "
        "data-driven takeaways based only on the provided quantitative summary. Output: JSON array of 5 strings."
    )
    user_query = f"Generate five key takeaways based on this quantitative data summary:\n\n{summary_data}"
    payload = {
        "model": GROK_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "ARRAY", "items": {"type": "STRING"}}
        },
        "temperature": 0.4,
    }
    with st.spinner("Analyzing the quantitative data to generate executive takeaways..."):
        json_response = call_grok_with_backoff(payload, api_key)
    if json_response:
        try:
            return json.loads(json_response)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response for takeaways.")
            return None
    return None

# --- Custom Visualization Functions (no 'Other/Long Tail') ---
def plot_stacked_author_share(df_classified, author_col, theme_col, engagement_col, top_n=5):
    """
    Horizontal stacked bar: total likes per theme, segments = top N authors (no 'Other/Long Tail').
    """
    # Total likes per theme (for % tooltip)
    theme_total_likes = df_classified.groupby(theme_col)[engagement_col].sum().rename('Theme_Total').reset_index()

    # Likes by Theme and Author
    df_grouped = (df_classified
                  .groupby([theme_col, author_col])[engagement_col]
                  .sum()
                  .reset_index(name='Author_Likes'))

    # Keep only top N authors per theme (no 'Other')
    df_top = (df_grouped
              .sort_values([theme_col, 'Author_Likes'], ascending=[True, False])
              .groupby(theme_col, as_index=False)
              .head(top_n))

    # Merge totals for percent
    df_top = df_top.merge(theme_total_likes, on=theme_col, how='left')
    df_top['Percentage'] = np.where(
        df_top['Theme_Total'] > 0, (df_top['Author_Likes'] / df_top['Theme_Total']) * 100, 0.0
    )

    fig = px.bar(
        df_top,
        x='Author_Likes',
        y=theme_col,
        color=author_col,
        orientation='h',
        title=f'Total Likes per Theme with Top {top_n} Author Share',
        labels={'Author_Likes': 'Total Likes', theme_col: 'Narrative Theme', author_col: 'Influencer'},
        height=550,
        category_orders={theme_col: df_top.groupby(theme_col)['Author_Likes'].sum().sort_values(ascending=True).index.tolist()},
        color_discrete_sequence=VIBRANT_QUAL
    )

    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Author: %{customdata[0]}<br>Likes: %{x:,}<br>Share: %{customdata[1]:.2f}%<extra></extra>',
        customdata=df_top[[author_col, 'Percentage']].values,
        marker_line_width=0
    )

    wrapped = {t: wrap_text(t, 15) for t in df_top[theme_col].unique()}
    fig.update_layout(
        yaxis={'tickmode': 'array', 'tickvals': list(wrapped.keys()), 'ticktext': list(wrapped.values()), 'automargin': True},
        showlegend=True,
        legend_title_text="Top Influencers"
    )

    fig = finalize_figure(
        fig,
        title=f"Engagement share by author within each theme (Top {top_n})",
        subtitle="Bars show total likes per theme; segments show author share of the top authors only",
        source="Meltwater; analysis by app",
        height=550
    )
    return fig

def plot_overall_author_ranking(df_classified, author_col, engagement_col, top_n=10):
    """
    Horizontal bar: top N authors by total likes, colored by most frequent theme.
    """
    df_classified['Theme_Rank'] = df_classified.groupby([author_col, 'NARRATIVE_TAG'])['POST_TEXT'].transform('count')
    df_classified = df_classified.sort_values(by=['Theme_Rank'], ascending=False)
    df_primary_theme = df_classified.drop_duplicates(subset=[author_col], keep='first')[[author_col, 'NARRATIVE_TAG']].rename(columns={'NARRATIVE_TAG': 'Primary_Theme'})

    overall_metrics = df_classified.groupby(author_col).agg(
        Total_Likes=(engagement_col, 'sum'),
        Total_Posts=('POST_TEXT', 'size')
    ).reset_index()

    overall_metrics = overall_metrics.merge(df_primary_theme, on=author_col, how='left')
    overall_metrics = overall_metrics.sort_values(by='Total_Likes', ascending=False).head(top_n)

    fig = px.bar(
        overall_metrics,
        x='Total_Likes',
        y=author_col,
        color='Primary_Theme',
        orientation='h',
        title=f'Top {top_n} Influencers by Total Likes (Colored by Primary Theme)',
        labels={'Total_Likes': 'Total Likes (Engagement)', author_col: 'Author/Influencer', 'Primary_Theme': 'Primary Narrative Theme'},
        height=550,
        category_orders={author_col: overall_metrics[author_col].tolist()},
        color_discrete_sequence=VIBRANT_QUAL
    )

    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Likes: %{x:,}<br>Posts: %{customdata[0]}<br>Primary Theme: %{customdata[1]}<extra></extra>',
        customdata=overall_metrics[['Total_Posts', 'Primary_Theme']].values,
        marker_line_width=0
    )

    fig.update_layout(yaxis={'title': None, 'automargin': True}, xaxis={'tickformat': ',', 'title': 'Total Likes'})

    fig = finalize_figure(
        fig,
        title=f"Top {top_n} influencers by total likes",
        subtitle="Colored by each author’s primary narrative theme",
        source="Meltwater; analysis by app",
        height=550
    )
    return fig

def plot_theme_influencer_share(df_viz, theme, author_col, engagement_col, top_n=5):
    """
    Single theme horizontal stacked bar: top N authors only (no 'Other/Long Tail').
    """
    df_theme = df_viz[df_viz['NARRATIVE_TAG'] == theme].copy()
    if df_theme.empty:
        return None

    df_grouped = df_theme.groupby(author_col)[engagement_col].sum().reset_index(name='Author_Likes')
    df_top = df_grouped.nlargest(top_n, 'Author_Likes')  # no 'Other'

    theme_total = df_theme[engagement_col].sum()
    df_top['Percentage'] = np.where(theme_total > 0, (df_top['Author_Likes'] / theme_total) * 100, 0.0)

    fig = px.bar(
        df_top,
        x='Author_Likes',
        y=author_col,
        orientation='h',
        title=f"{theme}: Top {top_n} Influencers by Likes",
        labels={'Author_Likes': 'Likes', author_col: 'Influencer'},
        height=400,
        color=author_col,
        color_discrete_sequence=VIBRANT_QUAL
    )
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Likes: %{x:,}<br>Share: %{customdata[0]:.1f}%<extra></extra>',
        customdata=df_top[['Percentage']],
        marker_line_width=0
    )
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title='Total Likes')

    fig = finalize_figure(
        fig,
        title=f"{theme}: top {top_n} influencers by likes",
        subtitle="Share computed against total theme likes; only top authors shown",
        source="Meltwater; analysis by app",
        height=380
    )
    return fig

# --- Streamlit UI and Workflow ---
# Initialize Session State
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'narrative_data' not in st.session_state:
    st.session_state.narrative_data = None
if 'classified_df' not in st.session_state:
    st.session_state.classified_df = None
if 'data_summary_text' not in st.session_state:
    st.session_state.data_summary_text = None

# Check for API key
XAI_KEY = os.getenv(XAI_API_KEY_ENV_VAR)
st.session_state.api_key = XAI_KEY

# --- Application Title ---
st.title("Grok-Powered Narrative Analysis Dashboard")
st.markdown("Automated thematic extraction and quantitative analysis of Meltwater data.")

# --- SIDEBAR (Configuration and Upload) ---
with st.sidebar:
    if not st.session_state.api_key:
        st.error(f"FATAL ERROR: Grok API Key not found. Please set the '{XAI_API_KEY_ENV_VAR}' environment variable.")

    st.markdown("#### File Upload")
    uploaded_file = st.file_uploader(
        "Upload Meltwater Data (.xlsx)",
        type=['xlsx'],
        help="Upload your Meltwater file in Excel format (.xlsx)."
    )

    if st.session_state.classified_df is not None and not st.session_state.classified_df.empty:
        st.markdown("---")
        st.markdown("#### Download Tagged Data")
        csv = st.session_state.classified_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Tagged Data (CSV)",
            data=csv,
            file_name='meltwater_narrative_analysis.csv',
            mime='text/csv',
            type="primary"
        )

# --- STOP APP IF NO KEY OR NO FILE ---
if not st.session_state.api_key or uploaded_file is None:
    if uploaded_file is None:
        st.info("Upload your Meltwater Data (.xlsx) in the sidebar to begin the analysis.")
    st.stop()

# --- Main App Logic ---
with st.container():
    if st.session_state.df_full is None:
        try:
            uploaded_file.seek(0)
            st.info("Reading Excel file (.xlsx)...")

            df = pd.read_excel(
                uploaded_file,
                skiprows=HEADER_ROWS_TO_SKIP,
                engine='openpyxl'
            )

            df.columns = df.columns.str.strip()
            required_cols = TEXT_COLUMNS + [ENGAGEMENT_COLUMN, AUTHOR_COLUMN, DATE_COLUMN, TIME_COLUMN]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"File is missing essential columns. Required: {', '.join(required_cols)}. Missing: {', '.join(missing_cols)}")

            # Parse full datetime in Date column
            df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).str.strip()
            df['DATETIME'] = pd.to_datetime(
                df[DATE_COLUMN],
                format=DATE_TIME_FORMAT,
                errors='coerce'
            )

            parse_success_rate = (df['DATETIME'].notna()).mean() * 100
            if parse_success_rate < 90:
                st.warning(f"Date parsing succeeded for only {parse_success_rate:.1f}% of rows. Check sample data for anomalies.")
            else:
                st.success(f"Date parsing complete: {parse_success_rate:.1f}% success rate ({df['DATETIME'].notna().sum():,} valid datetimes).")

            df['POST_TEXT'] = df.apply(
                lambda row: ' | '.join(str(row[col]) for col in TEXT_COLUMNS if col in df.columns),
                axis=1
            )

            df[ENGAGEMENT_COLUMN] = pd.to_numeric(df[ENGAGEMENT_COLUMN], errors='coerce').fillna(0).astype(int)

            df = df[df['POST_TEXT'].str.strip().lower() != 'nan | nan | nan']

            st.session_state.df_full = df.copy()
            st.success("File uploaded successfully!")

            data_rows = df.shape[0]
            valid_dates = df['DATETIME'].dropna()
            date_min = valid_dates.min().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"
            date_max = valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"
            st.markdown("#### File Data Summary")
            st.markdown(f"""
            - **Total Rows Processed:** {data_rows:,}
            - **Date Span:** {date_min} to {date_max}
            """)
            st.markdown("---")

            st.session_state.narrative_data = None
            st.session_state.classified_df = None
            st.session_state.data_summary_text = None
            st.rerun()

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.df_full = None
            st.stop()

    # Execution steps follow only if df_full is loaded
    if st.session_state.df_full is not None:
        # --- Narratives Extraction (Step 1) ---
        st.header("Narratives Extraction")
        if not st.session_state.narrative_data:
            if st.button(f"Click here to start narrative extraction using {GROK_MODEL}", type="primary"):
                df_sample = st.session_state.df_full.sample(min(MAX_POSTS_FOR_ANALYSIS, len(st.session_state.df_full)), random_state=42)
                corpus = ' | '.join(df_sample['POST_TEXT'].tolist())
                narrative_list = analyze_narratives(corpus, st.session_state.api_key)
                if narrative_list:
                    st.session_state.narrative_data = narrative_list
                    st.success("Grok identified narrative themes from a sample set of 100 posts. Based on those themes, it will now tag the entire dataset to enable the Python libraries to do the data analytics. (Python can handle much more volume than the LLMs.)")
                    st.rerun()
            else:
                st.info(f"Click the button to sample {min(MAX_POSTS_FOR_ANALYSIS, len(st.session_state.df_full))} posts and generate 3-5 key narratives.")

        if st.session_state.narrative_data:
            st.subheader("Identified Narrative Themes")
            for i, narrative in enumerate(st.session_state.narrative_data):

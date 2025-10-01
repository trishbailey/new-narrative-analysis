import streamlit as st
import pandas as pd
import requests
import os
import json
import time
import plotly.express as px
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

# --- Streamlit Theme Configuration (UPDATED for Light Mode) ---
PRIMARY_COLOR = "#1E88E5"   # Blue for primary buttons
BG_COLOR = "#ffffff"        # Main background
SIDEBAR_BG = "#f7f7f7"      # Sidebar background
TEXT_COLOR = "#333333"      # Main text
HEADER_COLOR = "#111111"    # Headers

# --- Inject Configuration into Streamlit (Light Mode CSS) ---
st.set_page_config(
    page_title="Grok Narrative Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Grok-Powered Narrative Analyzer for Meltwater Data"}
)

# Use st.markdown to inject the styling needed for Light Mode and custom colors
st.markdown(
f"""
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
        background-color: #1565C0 !important; /* darker blue */
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
        color: #d63384; /* magenta-like highlight */
        padding: 2px 4px;
        border-radius: 4px;
    }}

    /* Fix Plotly chart backgrounds in light mode */
    .js-plotly-plot {{
        background-color: {BG_COLOR} !important;
    }}
    </style>
""",
    unsafe_allow_html=True
)

# --- Utility Functions ---
# Exponential Backoff for API calls
def call_grok_with_backoff(payload, api_key, max_retries=5):
    """Handles POST request to Grok API with error handling and exponential backoff."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    for attempt in range(max_retries):
        try:
            response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=300)  # 5 min timeout
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            if result.get('choices') and result['choices'][0].get('message'):
                return result['choices'][0]['message']['content']
            else:
                st.error("Error: Unexpected API response structure.")
                return None
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
            return None
    st.error("Max retries reached. API call failed.")
    return None

# Single post classification for the AI Seed (Phase 1 of Step 2)
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
        "temperature": 0.1,  # Low temperature for deterministic output
        "max_tokens": 50     # Keep output short
    }
    response_text = call_grok_with_backoff(payload, api_key)
    # Clean and validate the response
    if response_text and response_text.strip() in themes_list or response_text.strip() == CLASSIFICATION_DEFAULT:
        return response_text.strip()
    return CLASSIFICATION_DEFAULT  # Fallback if Grok returns garbage

# --- Analysis Logic ---
def analyze_narratives(corpus, api_key):
    """Calls Grok to generate narrative themes and summaries (Step 1)"""
    system_prompt = (
        "You are a world-class social media narrative analyst. Your task is to analyze the provided corpus of "
        "social media posts (from Meltwater) and identify the 3 to 5 most significant, high-level, emerging "
        "discussion themes or sub-narratives. "
        "Output your findings as a single JSON array of objects. "
        "CRITICAL: Ensure the `narrative_title` for each theme is highly specific, descriptive, and suitable "
        "for use as a label on a bar chart (e.g., 'Escalator Sabotage & UN Conspiracy' instead of 'Technical Problems'). "
        "The language of the titles and summaries should match the dominant language of the corpus."
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
                        "narrative_title": {"type": "STRING", "description": "A specific, chart-ready title for the theme."},
                        "summary": {"type": "STRING", "description": "A brief summary of the theme and why it is significant."},
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
    # 1. AI Seed Generation (Grok Labels a small sample)
    st.info(f"Phase 1: Generating {AI_SEED_SAMPLE_SIZE} labeled training examples using {CLASSIFICATION_MODEL}...")
    # Ensure sample size doesn't exceed available data
    actual_seed_size = min(AI_SEED_SAMPLE_SIZE, len(df_full))
    df_sample = df_full.sample(actual_seed_size, random_state=42).copy()
    df_remaining = df_full.drop(df_sample.index).copy()
    seed_tags = []

    # Use concurrent execution for the seed generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all classification tasks for the seed
        future_to_post = {
            executor.submit(classify_post_for_seed, post_text, theme_titles, api_key): post_text
            for post_text in df_sample['POST_TEXT']
        }
        # Collect results with a progress bar
        progress_bar = st.progress(0)
        for i, future in enumerate(concurrent.futures.as_completed(future_to_post)):
            seed_tags.append(future.result())
            progress_bar.progress((i + 1) / actual_seed_size)

    df_sample['NARRATIVE_TAG'] = seed_tags
    progress_bar.empty()

    # Check if we have enough diverse training data
    if df_sample['NARRATIVE_TAG'].nunique() < 2:
        st.error("AI failed to generate diverse enough labels for training. Check generated themes or try increasing AI Seed Sample Size.")
        return None

    st.success(f"Phase 1 Complete: {len(df_sample)} examples labeled by Grok.")

    # 2. ML Model Training
    st.info("Phase 2: Training local TF-IDF/LinearSVC model on Grok's labels...")
    X_train = df_sample['POST_TEXT']
    y_train = df_sample['NARRATIVE_TAG']
    model = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('clf', LinearSVC(C=1.0, random_state=42, max_iter=5000))
    ])
    model.fit(X_train, y_train)
    st.success("Phase 2 Complete: Local ML Model Trained.")

    # 3. ML Model Prediction
    st.info(f"Phase 3: Classifying remaining {len(df_remaining):,} posts with local model...")
    X_test = df_remaining['POST_TEXT']
    df_remaining['NARRATIVE_TAG'] = model.predict(X_test)
    st.success("Phase 3 Complete: Full dataset classification finished.")

    # Combine the labeled seed data and the predicted remaining data
    df_classified = pd.concat([df_sample, df_remaining])
    return df_classified

# --- Data Crunching and Summary Generation (Step 3 Helper) ---
def perform_data_crunching_and_summary(df_classified: pd.DataFrame) -> str:
    """Performs required data aggregation and formats it into a text summary for Grok."""
    # 1. Theme Metrics (Volume, Likes, Average Likes)
    theme_metrics = df_classified.groupby('NARRATIVE_TAG').agg(
        Volume=('POST_TEXT', 'size'),
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum')
    ).reset_index()
    theme_metrics['Avg_Likes_Per_Post'] = theme_metrics['Total_Likes'] / theme_metrics['Volume']
    theme_metrics = theme_metrics.sort_values(by='Volume', ascending=False)

    narrative_summary = "Narrative Metrics (Volume, Total Likes, Avg Likes Per Post):\n"
    narrative_summary += theme_metrics.to_string(index=False, float_format="%.2f") + "\n\n"

    # 2. Overall Top Authors by Likes
    overall_top_authors = df_classified.groupby(AUTHOR_COLUMN).agg(
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum'),
        Post_Count=('POST_TEXT', 'size')
    ).nlargest(3, 'Total_Likes').reset_index()
    author_summary = "Overall Top 3 Influencers (by Total Likes):\n"
    author_summary += overall_top_authors.to_string(index=False) + "\n\n"

    # 3. Overall Time Context
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
        "You are a senior social media analyst reporting to C-suite executives. Your task is to provide five "
        "concise, compelling, and data-driven key takeaways based *only* on the provided quantitative data summary. "
        "Focus on unexpected trends, dominance, or high-engagement narratives. Output should be a JSON array of 5 strings."
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
            "responseSchema": {
                "type": "ARRAY",
                "items": {"type": "STRING", "description": "A concise, data-driven takeaway."},
            }
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

# --- Custom Visualization Functions ---
def plot_stacked_author_share(df_classified, author_col, theme_col, engagement_col, top_n=5):
    """
    Generates a Plotly horizontal stacked bar chart showing total likes per theme,
    with stacked segments for the top N authors + 'Other'.
    """
    # 1. Aggregate total likes per theme (required for percentage calculation)
    theme_total_likes = df_classified.groupby(theme_col)[engagement_col].sum().rename('Theme_Total')

    # 2. Aggregate likes by Theme and Author
    df_grouped = df_classified.groupby([theme_col, author_col])[engagement_col].sum().reset_index(name='Author_Likes')

    # 3. Identify Top N Authors for EACH theme and group the rest into 'Other'
    def get_top_n_authors(group):
        top_authors = group.nlargest(top_n, 'Author_Likes')
        other_likes = group['Author_Likes'].sum() - top_authors['Author_Likes'].sum()
        if other_likes > 0 and group['Author_Likes'].sum() > 0:
            other_row = pd.DataFrame({
                theme_col: [group.name],
                author_col: ['Other/Long Tail'],
                'Author_Likes': [other_likes]
            })
            return pd.concat([top_authors, other_row], ignore_index=True)
        return top_authors

    df_top_authors = df_grouped.groupby(theme_col).apply(get_top_n_authors).reset_index(drop=True)
    df_top_authors = df_top_authors[df_top_authors['Author_Likes'] > 0].copy()

    # 4. Merge theme totals for calculating percentages (used in tooltips)
    df_top_authors = df_top_authors.merge(theme_total_likes, on=theme_col)
    df_top_authors['Percentage'] = (df_top_authors['Author_Likes'] / df_top_authors['Theme_Total']) * 100

    # 5. Create Plotly Stacked Bar Chart
    fig = px.bar(
        df_top_authors,
        x='Author_Likes',
        y=theme_col,
        color=author_col,
        orientation='h',
        title=f'Total Likes per Theme with Top {top_n} Author Share',
        labels={
            'Author_Likes': 'Total Likes',
            theme_col: 'Narrative Theme',
            author_col: 'Influencer Share'
        },
        height=550,
        category_orders={
            theme_col: df_top_authors.groupby(theme_col)['Author_Likes'].sum().sort_values(ascending=True).index.tolist()
        },
        color_discrete_sequence=px.colors.qualitative.Alphabet
    )

    # 6. Customize Tooltips and Layout
    fig.update_traces(
        hovertemplate=(
            f'<b>%{{y}}</b><br>'
            f'Author: %{{customdata[0]}}<br>'
            f'Likes: %{{x:,}}<br>'
            f'Share: %{{customdata[1]:.2f}}%<extra></extra>'
        ),
        customdata=df_top_authors[[author_col, 'Percentage']].values
    )

    # Wrap long theme labels for y-axis
    TICK_WRAP_WIDTH = 15
    def wrap_labels(text):
        return '<br>'.join(textwrap.wrap(text, TICK_WRAP_WIDTH))
    wrapped_labels = {theme: wrap_labels(theme) for theme in df_top_authors[theme_col].unique()}

    fig.update_layout(
        yaxis={
            'tickmode': 'array',
            'tickvals': list(wrapped_labels.keys()),
            'ticktext': list(wrapped_labels.values()),
            'automargin': True
        },
        showlegend=True,
        legend_title_text="Top Influencers",
        template='plotly_white'
    )
    return fig

# --- New Overall Author Chart Function ---
def plot_overall_author_ranking(df_classified, author_col, engagement_col, top_n=10):
    """
    Generates a Plotly horizontal bar chart showing top N authors by Total Likes,
    colored by their most frequent theme.
    """
    # 1. Calculate Primary Theme for each author
    df_classified['Theme_Rank'] = df_classified.groupby([author_col, 'NARRATIVE_TAG'])['POST_TEXT'].transform('count')
    df_classified = df_classified.sort_values(by=['Theme_Rank'], ascending=False)
    df_primary_theme = df_classified.drop_duplicates(subset=[author_col], keep='first')[[author_col, 'NARRATIVE_TAG']]
    df_primary_theme = df_primary_theme.rename(columns={'NARRATIVE_TAG': 'Primary_Theme'})

    # 2. Calculate Overall Total Likes and Posts
    overall_metrics = df_classified.groupby(author_col).agg(
        Total_Likes=(engagement_col, 'sum'),
        Total_Posts=('POST_TEXT', 'size')
    ).reset_index()

    # 3. Merge primary theme and sort to get Top N
    overall_metrics = overall_metrics.merge(df_primary_theme, on=author_col, how='left')
    overall_metrics = overall_metrics.sort_values(by='Total_Likes', ascending=False).head(top_n)

    # 4. Create Plotly Horizontal Bar Chart
    fig = px.bar(
        overall_metrics,
        x='Total_Likes',
        y=author_col,
        color='Primary_Theme',
        orientation='h',
        title=f'Top {top_n} Influencers by Total Likes (Colored by Primary Theme)',
        labels={
            'Total_Likes': 'Total Likes (Engagement)',
            author_col: 'Author/Influencer',
            'Primary_Theme': 'Primary Narrative Theme'
        },
        height=550,
        category_orders={
            author_col: overall_metrics[author_col].tolist()
        },
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_traces(
        hovertemplate=(
            f'<b>%{{y}}</b><br>'
            f'Likes: %{{x:,}}<br>'
            f'Posts: %{{customdata[0]}}<br>'
            f'Primary Theme: %{{customdata[1]}}<extra></extra>'
        ),
        customdata=overall_metrics[['Total_Posts', 'Primary_Theme']].values
    )

    fig.update_layout(
        yaxis={'title': None, 'automargin': True},
        xaxis={'tickformat': ',', 'title': 'Total Likes'},
        template='plotly_white'
    )
    return fig

def plot_theme_influencer_share(df_viz, theme, author_col, engagement_col, top_n=5):
    """
    Generates a single horizontal stacked bar chart for one theme's top authors.
    """
    df_theme = df_viz[df_viz['NARRATIVE_TAG'] == theme].copy()
    if df_theme.empty:
        return None

    df_grouped = df_theme.groupby(author_col)[engagement_col].sum().reset_index(name='Author_Likes')
    top_authors = df_grouped.nlargest(top_n, 'Author_Likes')
    other_likes = df_grouped['Author_Likes'].sum() - top_authors['Author_Likes'].sum()
    if other_likes > 0:
        other_row = pd.DataFrame({author_col: ['Other/Long Tail'], 'Author_Likes': [other_likes]})
        df_top = pd.concat([top_authors, other_row], ignore_index=True)
    else:
        df_top = top_authors

    theme_total = df_theme[engagement_col].sum()
    df_top['Percentage'] = (df_top['Author_Likes'] / theme_total) * 100

    fig = px.bar(
        df_top,
        x='Author_Likes',
        y=author_col,
        orientation='h',
        title=f"{theme}: Top {top_n} Influencers by Likes",
        labels={'Author_Likes': 'Likes', author_col: 'Influencer'},
        height=400,
        color=author_col,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(
        hovertemplate=f'<b>%{{y}}</b><br>Likes: %{{x:,}}<br>Share: %{{customdata[0]:.1f}}%<extra></extra>',
        customdata=df_top[['Percentage']]
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title=None,
        xaxis_title='Total Likes',
        template='plotly_white'
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

# FIX: Check for API key securely outside of the sidebar
XAI_KEY = os.getenv(XAI_API_KEY_ENV_VAR)
st.session_state.api_key = XAI_KEY

# --- Application Title (FIX: Restored the main title here) ---
st.title("Grok-Powered Narrative Analysis Dashboard")
st.markdown("Automated thematic extraction and quantitative analysis of Meltwater data.")

# --- SIDEBAR (Configuration and Upload) ---
with st.sidebar:
    # --- API Key Check (Discreet) ---
    if not st.session_state.api_key:
        st.error(f"FATAL ERROR: Grok API Key not found. Please set the '{XAI_API_KEY_ENV_VAR}' environment variable.")

    # --- File Upload ---
    st.markdown("#### File Upload")
    uploaded_file = st.file_uploader(
        "Upload Meltwater Data (.xlsx)",
        type=['xlsx'],
        help="Upload your Meltwater file in Excel format (.xlsx)."
    )

    # --- Download Button ---
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

# --- Main App Logic (Container for upload feedback) ---
with st.container():
    if st.session_state.df_full is None:
        try:
            uploaded_file.seek(0)
            st.info("Reading Excel file (.xlsx)...")

            # --- Excel Reading ---
            df = pd.read_excel(
                uploaded_file,
                skiprows=HEADER_ROWS_TO_SKIP,
                engine='openpyxl'
            )

            # --- Final Validation and Preprocessing ---
            df.columns = df.columns.str.strip()
            required_cols = TEXT_COLUMNS + [ENGAGEMENT_COLUMN, AUTHOR_COLUMN, DATE_COLUMN, TIME_COLUMN]
            if not all(col in df.columns for col in required_cols):
                missing_cols = [col for col in required_cols if col not in df.columns]
                raise ValueError(f"File is missing essential columns. Required: {', '.join(required_cols)}. Missing: {', '.join(missing_cols)}")

            # --- Date Parsing (parse full datetime in Date column) ---
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

            # Combined text column
            df['POST_TEXT'] = df.apply(
                lambda row: ' | '.join(str(row[col]) for col in TEXT_COLUMNS if col in df.columns),
                axis=1
            )

            # Likes numeric
            df[ENGAGEMENT_COLUMN] = pd.to_numeric(df[ENGAGEMENT_COLUMN], errors='coerce').fillna(0).astype(int)

            # Remove rows where all text columns were NaN or empty strings
            df = df[df['POST_TEXT'].str.strip().str.lower() != 'nan | nan | nan']

            st.session_state.df_full = df.copy()
            st.success("File uploaded successfully!")

            # --- DATA SUMMARY ---
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

            # Clear previous results when a new file is uploaded
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
                st.markdown(f"**{i+1}. {narrative['narrative_title']}**: {narrative['summary']}")
            st.session_state.theme_titles = [item['narrative_title'] for item in st.session_state.narrative_data]
            st.success("Grok identified narrative themes from a sample set of 100 posts. Based on those themes, it will now tag the entire dataset to enable the Python libraries to do the data analytics. (Python can handle much more volume than the LLMs.)")

        # --- Data Analysis by Narrative (Step 2) ---
        st.markdown("---")
        st.header("Data Analysis by Narrative")

        if st.session_state.narrative_data and (st.session_state.classified_df is None or st.session_state.classified_df.empty):
            if st.button(f"Click here to classify {len(st.session_state.df_full):,} posts by narrative", type="primary"):
                df_classified = train_and_classify_hybrid(st.session_state.df_full, st.session_state.theme_titles, st.session_state.api_key)
                if df_classified is not None:
                    st.session_state.classified_df = df_classified
                    st.success("Hybrid classification complete. Dashboard generated.")
                    st.rerun()

        if st.session_state.classified_df is not None and not st.session_state.classified_df.empty:
            df_classified = st.session_state.classified_df

            # Filter for visualization (exclude Other/Unrelated and require valid dates)
            df_viz = df_classified[
                (df_classified['NARRATIVE_TAG'] != CLASSIFICATION_DEFAULT) &
                (df_classified['DATETIME'].notna())
            ].copy()

            if df_viz.empty:
                st.warning("No posts were classified into the primary narrative themes OR no posts had valid date information. The dashboard cannot be generated.")
            else:
                st.subheader("Narrative Analysis Dashboard")

                # 1. Bar Chart: Volume only
                st.markdown("### Post Volume by Theme")
                theme_metrics = df_viz.groupby('NARRATIVE_TAG').agg(
                    Post_Volume=('POST_TEXT', 'size'),
                ).reset_index()
                theme_metrics = theme_metrics.sort_values(by='Post_Volume', ascending=False)
                fig_bar = px.bar(
                    theme_metrics,
                    x='NARRATIVE_TAG',
                    y='Post_Volume',
                    title='Post Volume by Theme',
                    labels={'Post_Volume': 'Post Volume (Count)', 'NARRATIVE_TAG': 'Narrative Theme'},
                    height=500,
                    color='NARRATIVE_TAG',
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )

                TICK_WRAP_WIDTH_BAR = 15
                def wrap_labels_bar(text):
                    return '<br>'.join(textwrap.wrap(text, TICK_WRAP_WIDTH_BAR))
                fig_bar.update_layout(
                    xaxis={
                        'categoryorder':'total descending',
                        'tickangle': 0,
                        'automargin': True,
                        'tickfont': {'size': 12},
                        'tickvals': theme_metrics['NARRATIVE_TAG'].tolist(),
                        'ticktext': [wrap_labels_bar(t) for t in theme_metrics['NARRATIVE_TAG']],
                    },
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # 2. Line Graph: Trend Over Time (Volume)
                st.markdown("### Narrative Volume Trend Over Time (7-Day Rolling Average)")
                df_trends_theme = df_viz.groupby([df_viz['DATETIME'].dt.date, 'NARRATIVE_TAG']).size().reset_index(name='Post_Volume')
                df_trends_theme['DATETIME'] = pd.to_datetime(df_trends_theme['DATETIME'])
                df_trends_theme['Volume_Roll_Avg'] = df_trends_theme.groupby('NARRATIVE_TAG')['Post_Volume'].transform(
                    lambda x: x.rolling(window=7, min_periods=1).mean()
                )

                if not df_trends_theme.empty and len(df_trends_theme['DATETIME'].dt.date.unique()) > 1:
                    fig_line = px.line(
                        df_trends_theme,
                        x='DATETIME',
                        y='Volume_Roll_Avg',
                        color='NARRATIVE_TAG',
                        title='Volume Trend by Narrative Theme (7-Day Rolling Average)',
                        labels={'Volume_Roll_Avg': 'Rolling Avg. Post Volume', 'DATETIME': 'Date', 'NARRATIVE_TAG': 'Theme'},
                        height=550,
                        color_discrete_sequence=px.colors.qualitative.Plotly
                    )
                    fig_line.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Rolling Avg. Post Volume",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5
                        ),
                        template='plotly_white'
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.warning("Trend chart requires posts spanning at least two unique days. Chart cannot be generated with current data.")

                # 3. Stacked Bar Charts: Influencer Share per Theme (Separate charts)
                st.markdown("### Influencer Share of Engagement")
                for theme in df_viz['NARRATIVE_TAG'].unique():
                    fig_theme = plot_theme_influencer_share(df_viz, theme, AUTHOR_COLUMN, ENGAGEMENT_COLUMN, top_n=5)
                    if fig_theme:
                        st.plotly_chart(fig_theme, use_container_width=True)

                # 4. Overall Top Authors by Likes
                st.markdown("### Top 10 Overall Authors by Total Likes")
                st.plotly_chart(
                    plot_overall_author_ranking(
                        df_classified,
                        AUTHOR_COLUMN,
                        ENGAGEMENT_COLUMN,
                        top_n=10
                    ),
                    use_container_width=True
                )

                # --- Insights from the Data (Step 3) ---
                st.markdown("---")
                st.header("Insights from the Data")
                if st.button(f"Click here to generate 5 key takeaways from the data", type="primary"):
                    data_summary_text = perform_data_crunching_and_summary(df_classified)
                    takeaways_list = generate_takeaways(data_summary_text, st.session_state.api_key)
                    if takeaways_list:
                        st.subheader("Executive Summary: 5 Key Takeaways")
                        for i, takeaway in enumerate(takeaways_list):
                            st.markdown(f"**{i+1}.** {takeaway}")

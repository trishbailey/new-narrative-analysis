import streamlit as st
import pandas as pd
import requests
import os
import json
import time
import plotly.express as px
import matplotlib.pyplot as plt

# --- Configuration ---
XAI_API_KEY_ENV_VAR = 'XAI_API_KEY'
GROK_MODEL = 'grok-4-fast-reasoning'
CLASSIFICATION_MODEL = 'grok-3-mini' # Use a faster, cheaper model for the simple classification task
API_BASE_URL = "https://api.x.ai/v1/chat/completions"
MAX_POSTS_FOR_ANALYSIS = 100
CLASSIFICATION_DEFAULT = "Other/Unrelated"

# --- Meltwater Data Column Mapping (Based on user-provided file structure) ---
TEXT_COLUMNS = ['Opening Text', 'Headline', 'Hit Sentence']
ENGAGEMENT_COLUMN = 'Likes'
AUTHOR_COLUMN = 'Screen Name'
DATE_COLUMN = 'Date'
TIME_COLUMN = 'Time'

# --- Streamlit Setup ---
st.set_page_config(
    page_title="Grok Narrative Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("Grok-Powered Narrative Analyzer")
st.markdown("Upload your Meltwater data to perform a 3-step thematic and quantitative analysis using Grok AI.")


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
            response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=300) # 5 min timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            result = response.json()
            # Grok API uses 'content' -> 'parts' -> 'text' for the JSON string
            if result.get('choices') and result['choices'][0].get('message'):
                return result['choices'][0]['message']['content']
            else:
                st.error("Error: Unexpected API response structure.")
                return None

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"Rate limit hit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                st.error(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                return None
        except requests.exceptions.RequestException as e:
            st.error(f"API Request Failed: {e}")
            return None
    
    st.error("Max retries reached. API call failed.")
    return None

# Classification utility using a separate, faster model
@st.cache_data(show_spinner=False)
def classify_post(post_text, themes_list, api_key):
    """Classifies a single post against a list of themes."""
    # Build a list of valid themes for Grok to choose from
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
        "max_tokens": 50  # Keep output short
    }
    
    response_text = call_grok_with_backoff(payload, api_key)
    
    # Clean and validate the response
    if response_text and response_text.strip() in themes_list or response_text.strip() == CLASSIFICATION_DEFAULT:
        return response_text.strip()
    return CLASSIFICATION_DEFAULT # Fallback if Grok returns garbage

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

# --- Data Crunching and Summary Generation ---

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
    start_date = df_classified['DATETIME'].min().strftime('%Y-%m-%d')
    end_date = df_classified['DATETIME'].max().strftime('%Y-%m-%d')
    total_posts = len(df_classified)
    total_likes_all = df_classified[ENGAGEMENT_COLUMN].sum()
    
    context_summary = (
        f"Dataset Context:\n"
        f"  Total Posts Analyzed: {total_posts:,}\n"
        f"  Total Likes Across All Posts: {total_likes_all:,}\n"
        f"  Timeframe: {start_date} to {end_date}\n"
    )
    
    return narrative_summary + author_summary + context_summary


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

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Configuration")
    
    # API Key Input - prioritizes env var
    XAI_KEY = os.getenv(XAI_API_KEY_ENV_VAR)
    key_placeholder = "API Key not found in environment." if not XAI_KEY else "Key loaded from environment."
    api_key = st.text_input(
        "XAI/Grok API Key", 
        type="password",
        value=XAI_KEY,
        placeholder=key_placeholder
    )
    st.session_state.api_key = api_key

    st.markdown("---")
    st.subheader("Models")
    st.write(f"Theme Generation (Step 1): `{GROK_MODEL}`")
    st.write(f"Classification (Step 2): `{CLASSIFICATION_MODEL}`")


# --- Step 0: File Upload & Setup ---
uploaded_file = st.file_uploader(
    "Upload your Meltwater CSV File", 
    type=['csv'],
    help="Requires columns: Headline, Opening Text, Hit Sentence, Likes, Date, Time, Screen Name"
)

# Container for upload feedback
with st.container(border=True):
    st.info("👈 Upload your Meltwater CSV file to begin the 3-step analysis.")
    if uploaded_file is not None and st.session_state.df_full is None:
        try:
            # --- Robust File Reading Logic (Attempting multiple delimiters) ---
            uploaded_file.seek(0)
            
            # List of delimiters to try, along with the most likely encoding fix
            delimiters = [',', ';', '\t', '|'] 
            df = None
            
            for sep in delimiters:
                try:
                    df = pd.read_csv(uploaded_file, low_memory=False, encoding='unicode_escape', sep=sep, on_bad_lines='skip')
                    
                    # A quick check to see if the main columns exist (if not, try next delimiter)
                    if all(col in df.columns for col in TEXT_COLUMNS):
                        break # Found the correct delimiter
                    
                    # Rewind the file pointer to the start for the next attempt
                    uploaded_file.seek(0) 

                except pd.errors.ParserError:
                    uploaded_file.seek(0) # Rewind for the next attempt
                    continue
            
            if df is None:
                 raise ValueError("Could not determine the correct CSV delimiter or file is corrupted.")
            
            # --- Data Preprocessing/Cleaning ---
            df.columns = df.columns.str.strip()
            
            # Combine Date and Time
            df['DATETIME'] = pd.to_datetime(df[DATE_COLUMN] + ' ' + df[TIME_COLUMN], errors='coerce')
            df.dropna(subset=['DATETIME'], inplace=True)
            
            # Create a combined text column
            df['POST_TEXT'] = df.apply(
                lambda row: ' | '.join(str(row[col]) for col in TEXT_COLUMNS if col in df.columns),
                axis=1
            )
            
            # Convert Likes to numeric, handling errors
            df[ENGAGEMENT_COLUMN] = pd.to_numeric(df[ENGAGEMENT_COLUMN], errors='coerce').fillna(0).astype(int)

            st.session_state.df_full = df.copy()
            
            data_rows = df.shape[0]
            date_min = df['DATETIME'].min().strftime('%Y-%m-%d')
            date_max = df['DATETIME'].max().strftime('%Y-%m-%d')

            st.success("File uploaded successfully! ")
            st.markdown(f"""
            - Data rows: **{data_rows:,}**
            - Date range: **{date_min}** to **{date_max}**
            """)
            
            # Clear previous results when a new file is uploaded
            st.session_state.narrative_data = None
            st.session_state.classified_df = None
            st.session_state.data_summary_text = None

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.df_full = None

if st.session_state.df_full is None:
    st.stop()


# --- Step 1: Generate Themes ---
st.header("Step 1: Generate Themes (Qualitative)")

if not st.session_state.narrative_data:
    if st.button(f"Start Theme Generation using {GROK_MODEL}"):
        if not st.session_state.api_key:
            st.error("Please enter your XAI/Grok API Key in the sidebar.")
            st.stop()
        
        # Take a sample of the data for narrative generation
        df_sample = st.session_state.df_full.sample(min(MAX_POSTS_FOR_ANALYSIS, len(st.session_state.df_full)), random_state=42)
        corpus = ' | '.join(df_sample['POST_TEXT'].tolist())
        
        narrative_list = analyze_narratives(corpus, st.session_state.api_key)
        
        if narrative_list:
            st.session_state.narrative_data = narrative_list
            st.success("Narratives successfully generated.")
            st.rerun()
    else:
        st.info(f"Click the button to sample {min(MAX_POSTS_FOR_ANALYSIS, len(st.session_state.df_full))} posts and generate 3-5 key narratives.")
        
if st.session_state.narrative_data:
    st.subheader("Generated Narrative Themes")
    st.dataframe(pd.DataFrame(st.session_state.narrative_data), use_container_width=True)
    
    # Store list of titles for Step 2
    st.session_state.theme_titles = [item['narrative_title'] for item in st.session_state.narrative_data]
    st.success("Themes are ready for the classification step.")


# --- Step 2: Classify & Dashboard (Quantitative) ---
st.header("Step 2: Classify Full Dataset & Generate Dashboard")

if st.session_state.narrative_data and not st.session_state.classified_df:
    if st.button(f"Classify {len(st.session_state.df_full):,} Posts using {CLASSIFICATION_MODEL} (Start Dashboard)"):
        if not st.session_state.api_key:
            st.error("Please enter your XAI/Grok API Key in the sidebar.")
            st.stop()
        
        df_to_classify = st.session_state.df_full.copy()
        theme_titles = st.session_state.theme_titles
        
        classified_tags = []
        progress_bar = st.progress(0)
        
        # Classification loop
        for i, post_text in enumerate(df_to_classify['POST_TEXT']):
            # Add a small sleep to prevent rapid-fire rate limit hits
            time.sleep(0.05) 
            
            tag = classify_post(post_text, theme_titles, st.session_state.api_key)
            classified_tags.append(tag)
            progress_bar.progress((i + 1) / len(df_to_classify))

        df_to_classify['NARRATIVE_TAG'] = classified_tags
        st.session_state.classified_df = df_to_classify
        st.success("Classification complete. Dashboard generated.")
        st.rerun()

if st.session_state.classified_df is not None:
    df_classified = st.session_state.classified_df
    st.subheader("Narrative Analysis Dashboard")

    # 1. Bar Chart: Volume and Engagement
    st.markdown("### 1. Post Volume vs. Normalized Likes per Theme")

    theme_metrics = df_classified.groupby('NARRATIVE_TAG').agg(
        Post_Volume=('POST_TEXT', 'size'),
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum')
    ).reset_index()

    max_volume = theme_metrics['Post_Volume'].max()
    max_likes = theme_metrics['Total_Likes'].max()
    scale_factor = max_volume / max_likes if max_likes > 0 else 1
    theme_metrics['Normalized_Likes'] = theme_metrics['Total_Likes'] * scale_factor
    theme_metrics = theme_metrics.sort_values(by='Post_Volume', ascending=False)
    
    fig_bar = px.bar(
        theme_metrics, 
        x='NARRATIVE_TAG', 
        y=['Post_Volume', 'Normalized_Likes'], 
        title='Post Volume vs. Engagement (Likes)',
        labels={'value': 'Count / Normalized Likes', 'NARRATIVE_TAG': 'Narrative Theme'},
        height=500,
        color_discrete_sequence=px.colors.qualitative.PlotlyExpress
    )
    fig_bar.update_layout(xaxis={'categoryorder':'total descending'}, legend_title_text='Metric')
    st.plotly_chart(fig_bar, use_container_width=True)

    # 2. Line Graph: Trend Over Time (Volume)
    st.markdown("### 2. Volume and Likes Trend Over Time (7-Day Rolling Average)")

    df_trends = df_classified.set_index('DATETIME').resample('D').agg(
        Post_Volume=('POST_TEXT', 'size'),
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum')
    ).reset_index()
    
    df_trends['Volume_Roll_Avg'] = df_trends['Post_Volume'].rolling(window=7).mean()
    df_trends['Likes_Roll_Avg'] = df_trends['Total_Likes'].rolling(window=7).mean()

    max_volume_roll = df_trends['Volume_Roll_Avg'].max()
    df_trends['Normalized_Likes_Roll_Avg'] = (df_trends['Likes_Roll_Avg'] / df_trends['Likes_Roll_Avg'].max()) * max_volume_roll
    
    # Melt data for Plotly to use a single color dimension
    df_melted = df_trends.melt(id_vars='DATETIME', 
                                value_vars=['Volume_Roll_Avg', 'Normalized_Likes_Roll_Avg'],
                                var_name='Metric', 
                                value_name='Value')

    fig_line = px.line(
        df_melted, 
        x='DATETIME', 
        y='Value', 
        color='Metric',
        title='Rolling 7-Day Average: Posts vs. Engagement',
        labels={'Value': 'Count / Normalized Likes', 'DATETIME': 'Date'},
        height=500,
        color_discrete_sequence=['#4B0082', '#DC143C'] # Deep Purple for Volume, Crimson for Likes
    )
    st.plotly_chart(fig_line, use_container_width=True)


    col1, col2 = st.columns(2)

    # 3 & 4. Top Authors by Volume and Likes per Theme
    with col1:
        st.markdown("### 3. Top Authors by Post Volume (Per Theme)")
        author_volume = df_classified.groupby(['NARRATIVE_TAG', AUTHOR_COLUMN]).size().reset_index(name='Post_Count')
        top_authors_vol = author_volume.sort_values(['NARRATIVE_TAG', 'Post_Count'], ascending=[True, False]).groupby('NARRATIVE_TAG').head(5).reset_index(drop=True)
        st.dataframe(top_authors_vol, use_container_width=True, height=500, hide_index=True)

    with col2:
        st.markdown("### 4. Top Authors by Likes (Per Theme)")
        author_likes = df_classified.groupby(['NARRATIVE_TAG', AUTHOR_COLUMN])[ENGAGEMENT_COLUMN].sum().reset_index(name='Total_Likes')
        top_authors_likes = author_likes.sort_values(['NARRATIVE_TAG', 'Total_Likes'], ascending=[True, False]).groupby('NARRATIVE_TAG').head(5).reset_index(drop=True)
        st.dataframe(top_authors_likes, use_container_width=True, height=500, hide_index=True)


    # 5. Overall Top 5 Posters by Likes with Theme Context
    st.markdown("---")
    st.markdown("### 5. Top 5 Overall Authors by Total Likes")
    
    overall_top_authors = df_classified.groupby(AUTHOR_COLUMN).agg(
        Total_Likes=(ENGAGEMENT_COLUMN, 'sum'),
        Total_Posts=('POST_TEXT', 'size'),
        Themes_Contributed=('NARRATIVE_TAG', lambda x: ', '.join(x.unique()))
    ).sort_values(by='Total_Likes', ascending=False).head(5).reset_index()

    st.dataframe(overall_top_authors, use_container_width=True, hide_index=True)


    # --- Step 3: Generate Key Takeaways ---
    st.header("Step 3: Generate Key Takeaways (Synthesis)")
    
    data_summary_text = perform_data_crunching_and_summary(df_classified)

    if st.button(f"Generate 5 Key Takeaways using {GROK_MODEL}"):
        takeaways_list = generate_takeaways(data_summary_text, st.session_state.api_key)
        
        if takeaways_list:
            st.subheader("Executive Summary: 5 Key Takeaways")
            for i, takeaway in enumerate(takeaways_list):
                st.markdown(f"**{i+1}.** {takeaway}")
            
    # --- Data Download ---
    st.markdown("---")
    st.markdown("### Download Classified Data")
    csv = st.session_state.classified_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Classified CSV",
        data=csv,
        file_name='meltwater_narrative_analysis.csv',
        mime='text/csv',
    )

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

MAX_POSTS_FOR_ANALYSIS = 150  # Fixed sample size for Theme Generation
AI_SEED_SAMPLE_SIZE = 50      # Fixed sample size for ML Training
CLASSIFICATION_DEFAULT = "Other/Unrelated"

# --- Meltwater Data Column Mapping (CONFIRMED) ---
TEXT_COLUMNS = ['Opening Text', 'Headline', 'Hit Sentence']
ENGAGEMENT_COLUMN = 'Likes'
AUTHOR_COLUMN = 'Influencer'
DATE_COLUMN = 'Date'
TIME_COLUMN = 'Time'

# Primary explicit format used by some Meltwater exports when Date already includes time
DATE_TIME_FORMAT = '%d-%b-%Y %I:%M%p'

# --- Toxicity Keywords (expanded) ---
TOXIC_KEYWORDS = [
    # Core hate/violence terms
    'hate', 'hateful', 'hatred', 'racist', 'sexist', 'homophobe', 'bigot', 'fascist',
    'kill', 'murder', 'assassinate', 'slaughter', 'genocide', 'exterminate',
    'violent', 'violence', 'rage', 'angry', 'fury', 'wrath', 'attack', 'beat',
    'bomb', 'explode', 'shoot', 'stab', 'fight', 'war', 'terror', 'terrorist',
    'die', 'dead', 'death', 'suffer', 'pain', 'hurt', 'bleed', 'burn',
    'bitch', 'cunt', 'pussy', 'dick', 'asshole', 'fuck', 'shit', 'damn',
    'nigger', 'chink', 'spic', 'kike', 'faggot', 'dyke', 'tranny',
    'rape', 'molest', 'abuse', 'torment',
    # Expanded racial/ethnic slurs (modern variants from common sources)
    'abeed', 'abid', 'abo', 'abbo', 'afro engineering', 'ali baba', 'alligator bait', 'ang mo', 'ann', 'ape', 'apple', 'arapis', 'arabush', 'argie', 'armo', 'asing', 'aseng', 'ashke-nazi', 'aunt jemima', 'baiano', 'balija', 'bamboula', 'bambus', 'banaan', 'banana', 'banderite', 'barbarian', 'beaner', 'bimbo', 'bing', 'binghi', 'black buck', 'blackie', 'blatte', 'bluegum', 'boche', 'boerehater', 'bog', 'bohunk', 'bolita', 'boong', 'boonga', 'bootlip', 'bougnoule', 'bounty bar', 'bozgor', 'brillo pad', 'brownie', 'buckwheat', 'buddhahead', 'bulbash',
    'camel jockey', 'camel-fucker', 'camelfucker', 'cap', 'checkerboard', 'cheddar man', 'cheeky monkey', 'chi-chi', 'chee-chee', 'chi chi', 'cheerio', 'cherry', 'chew', 'chief', 'chigga', 'chigger', 'chili eater', 'chink', 'chink eye', 'chino', 'choco', 'chocolate drop', 'cholo', 'chong', 'chonk', 'chonkasaurus', 'chonkosaurus', 'chubby bunny', 'chug', 'chugger', 'chumbo', 'chump', 'ciapaty', 'ciapak', 'cigar store indian', 'coconut', 'coon', 'coona', 'coonass', 'cooney', 'coonhound', 'coontown', 'copperhead', 'cotton picker', 'cracker', 'crackerjack', 'crackhead', 'crackpipe', 'crapaholic', 'craphead', 'cricket', 'crip', 'cro', 'cromag', 'crow', 'crumpet', 'crunky', 'curry muncher', 'curry nigger',
    'dago', 'dego', 'darkie', 'darky', 'dawg', 'deadbeat', 'dengue', 'dengue fever', 'desi', 'dickhead', 'diesel', 'digger', 'dildo', 'dink', 'dip', 'dirty jew', 'dirty mexican', 'dirty sanchez', 'dixie', 'dog', 'dogbreath', 'dogface', 'dogfucker', 'doggy doo', 'dogman', 'doh', 'doh je', 'doh pei', 'doje', 'dolla', 'dollface', 'dolly', 'douche', 'douchebag', 'douchecanoe', 'douchelord', 'douchey', 'dove', 'doven', 'dovl', 'dow', 'drongo', 'dropkick', 'drug mule', 'druggie', 'druid', 'drummer boy', 'drygulch', 'dude', 'dudebro', 'dumbass', 'dumb blonde', 'dumb cluck', 'dumb fuck', 'dumb jock', 'dumbshit', 'dummy', 'dune coon', 'dung eater', 'dung heap', 'dungaree', 'dunk', 'dunkie', 'dunny', 'dusk', 'duster', 'dutch oven', 'dutchie', 'dweeb', 'dyke',
    # Modern internet slang insults and toxicity
    'simp', 'incel', 'cuck', 'beta', 'soyboy', 'chad', 'normie', 'cringe', 'based', 'redpilled', 'woke', 'snowflake', 'karens', 'thot', 'yeet', 'sus', 'cap', 'no cap', 'rizz', 'sigma', 'alpha', 'beta male', 'white knight', 'orbiter', 'friendzoned', 'clown', 'cope', 'seethe', 'dilate', 'yikes', 'yawn', 'lmao', 'rofl', 'kek', 'pwned', 'noob', 'gg', 'ez', 'trash', 'scrub', 'retard', 'tard', 'tarded', 'autist', 'asperg', 'spaz', 'spastic', 'mong', 'mongoloid', 'downie', 'troll', 'flamer', 'glowie', 'fed', 'sheeple', 'normcuck', 'volcel', 'mogged', 'looksmax', 'blackpill', 'doomer', 'bugman', 'npc', 'midwit',
    # Gender/sexuality modern slurs
    'trap', 'shemale', 'ladyboy', 'he-she', 'it', 'xir', 'ze', 'they/them', 'pronoun', 'cishet', 'cis scum', 'terf', 'tim', 'tucute', 'transtrender', 'egg', 'clocked', 'passoid', 'hsts', 'gira', 'handmaiden', 'peak trans', 'cotton ceiling', 'die cis scum', 'kill all men', 'male tears', 'mansplain', 'mansplaining', 'gaslight', 'gatekeep', 'girlboss', 'girlmath', 'boymath',
    # Violence/threat modern slang
    'clout chase', 'beef', 'diss', 'roast', 'cancel', 'dox', 'swat', 'raid', 'grief', 'flame war', 'shitpost', 'ragequit', 'tilt', 'smd', 'stfu', 'gtfo', 'nuke', 'yeet off a cliff', 'touch grass', 'log off', 'ratio', 'embarrassing', 'cringe compilation', 'fail', 'epic fail', 'owned', 'btfo', 'destroyed', 'salty', 'mad', 'pressing', 'fuming', 'melting down', 'crying', 'bawling', 'sobbing', 'triggered', 'schooled', 'humbled', 'cooked', 'done for', 'fucked up', 'screwed', 'royally fucked', 'assblasted', 'mindbroken',
    # Political/extremist modern slang
    'degen', 'chud', 'frogposter', 'groyper', 'pepe', 'wojak', 'soyjak', 'doomer', 'bloomer', 'zoomer', 'millennial', 'boomer', 'ok boomer', 'npc wojak', 'red scare', 'horseshoe theory', 'dirtbag left', 'dirtbag right', 'maga', 'trump derangement', 'biden crime family', 'deep state', 'qanon', 'pizzagate', 'groomer', 'grooming gang', 'great replacement', 'white genocide', 'kalergi', 'jew world order', 'zog', 'globalist', 'nwo', 'new world order', 'antifa', 'blm', 'acorn', 'soros', 'rothschild', 'illuminati', 'flat earth', 'vaxxed', 'antivax', 'plandemic', 'hoax', 'psyop', 'false flag', 'staged', 'crisis actor', 'glow in the dark', 'fedsurrection', 'ray epps', 'uniparty', 'rino', 'dino', 'nevertrumper', 'lincoln project', 'nevernicker', 'blueanon',
    # Drug/addiction slang (hateful context)
    'junkie', 'crackhead', 'methhead', 'tweaker', 'fiend', 'addict', 'doper', 'pothead', 'stoner', 'cokehead', 'heroin chic', 'chasing the dragon', 'speedball', 'shooting up', 'mainlining', 'track marks', 'overdose', 'nodding off', 'withdrawal', 'dts', 'cold turkey', 'rehab', 'relapse',
    # General modern insults (additional)
    'loser', 'lame', 'basic', 'tryhard', 'edgelord', 'contrarian', 'devil\'s advocate', 'concern troll', 'sea lion', 'tone troll', 'whataboutist', 'both sides', 'centrist', 'moderate', 'libtard', 'repubtard', 'socdem', 'ancom', 'mlm', 'tankie', 'dunkie', 'strawman', 'nuance', 'context', 'not all', 'all lives matter', 'blue lives matter', 'defund', 'abolish', 'icop', 'acop', 'thin blue line', 'back the blue', 'porky', 'class traitor', 'scab', 'rat', 'fink', 'snitch', 'grass', 'bent copper', 'dirty cop', 'badge bunny', 'copaganda', 'prison industrial complex', 'pic', 'school to prison pipeline', 'war on drugs', 'war on poverty', 'welfare queen', 'food stamp president', 'obama phone', 'trickle down', 'voodoo economics', 'supply side', 'austerity', 'neoliberal', 'shock doctrine', 'disaster capitalism', 'crony capitalism', 'corporatism'
]
TOXICITY_THRESHOLD = 0.01  # Density > this flags a post as toxic

# --- Streamlit Theme Configuration (Light Mode) ---
PRIMARY_COLOR = "#1E88E5"  # Blue for primary buttons
BG_COLOR = "#ffffff"       # Main background
SIDEBAR_BG = "#f7f7f7"     # Sidebar background
TEXT_COLOR = "#333333"     # Main text
HEADER_COLOR = "#111111"   # Headers

# --- Page Config ---
st.set_page_config(
    page_title="Narrative Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Narrative Analyzer for Meltwater Data"}
)

# --- Global CSS (Light Mode UI + Inter font) ---
st.markdown(
    f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG};
    color: {TEXT_COLOR};
}}
.stApp {{
    background-color: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
}}
h1, h2, h3, h4, h5, h6 {{
    color: {HEADER_COLOR} !important;
}}
p, span, label, div {{
    color: {TEXT_COLOR} !important;
}}
.stButton > button.primary {{
    background-color: {PRIMARY_COLOR} !important;
    border-color: {PRIMARY_COLOR} !important;
    color: white !important;
}}
.stButton > button.primary:hover {{
    background-color: #1565C0 !important;
    border-color: #1565C0 !important;
}}
.stButton > button:not(.primary) {{
    color: {HEADER_COLOR} !important;
    border-color: #d0d0d0 !important;
    background-color: #fafafa !important;
}}
div[data-testid="stAlert"] * p, div[data-testid="stAlert"] * h5 {{
    color: {TEXT_COLOR} !important;
}}
code {{
    background-color: #f2f2f2;
    color: #d63384;
    padding: 2px 4px;
    border-radius: 4px;
}}
.js-plotly-plot {{
    background-color: {BG_COLOR} !important;
}}
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
</style>
""",
    unsafe_allow_html=True
)

# --- Vibrant Plotly Template ---
VIBRANT_QUAL = [
    "#1E88E5", "#E53935", "#8E24AA", "#43A047", "#FB8C00", "#00ACC1",
    "#F4511E", "#3949AB", "#6D4C41", "#7CB342", "#D81B60", "#00897B",
]
vibrant_layout = pio.templates["plotly_white"].layout.to_plotly_json()
vibrant_layout.update({
    "font": {"family": "Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "size": 14, "color": "#111111"},
    "colorway": VIBRANT_QUAL,
    "title": {"x": 0.0, "xanchor": "left", "y": 0.95, "yanchor": "top", "font": {"size": 20}},
    "margin": {"l": 80, "r": 40, "t": 60, "b": 60},
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
    "legend": {
        "orientation": "h", "yanchor": "bottom", "y": -0.25, "xanchor": "left", "x": 0.0,
        "title": {"text": "", "font": {"size": 12, "color": "#374151"}},
        "font": {"size": 12, "color": "#111111"},
        "itemclick": "toggleothers"
    },
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": "white", "bordercolor": "#d1d5db", "font": {"family": "Inter", "size": 12, "color": "#111111"}}
})
pio.templates["vibrant"] = pio.templates["plotly_white"]
pio.templates["vibrant"].layout.update(vibrant_layout)

# --- Helpers ---
def finalize_figure(fig, title: str, subtitle: str | None = None, source: str | None = None, height: int | None = None):
    fig.update_layout(template="vibrant", title={"text": title})
    if height:
        fig.update_layout(height=height)
    if subtitle:
        fig.add_annotation(
            text=f"<span style='color:#6b7280'>{subtitle}</span>",
            xref="paper", yref="paper", x=0, y=1.07, showarrow=False, align="left"
        )
    return fig

def wrap_text(s: str, width: int = 16) -> str:
    return "<br>".join(textwrap.wrap(s, width))

# --- Narrative Normalizer (prevents KeyError on missing/variant keys) ---
def normalize_narratives(raw):
    """
    Normalize Grok output into a list of dicts with keys:
      - 'narrative_title' (required)
      - 'summary' (optional -> default '')
    Accepts:
      - list[dict] with variant keys
      - list[str] (titles only)
    Returns: list[{'narrative_title': str, 'summary': str}]
    """
    if not isinstance(raw, list):
        return []

    title_keys = {'narrative_title', 'title', 'narrative', 'theme', 'name'}
    summary_keys = {'summary', 'desc', 'description', 'overview', 'rationale', 'explanation'}

    out = []
    for item in raw:
        if isinstance(item, dict):
            title = None
            summary = ''
            for k in title_keys:
                if k in item and item[k]:
                    title = str(item[k]).strip()
                    break
            for k in summary_keys:
                if k in item and item[k]:
                    summary = str(item[k]).strip()
                    break
            if title:
                out.append({'narrative_title': title, 'summary': summary})
        elif isinstance(item, str) and item.strip():
            out.append({'narrative_title': item.strip(), 'summary': ''})
    return out

# --- Toxicity Scoring ---
def compute_toxicity_scores(df_viz):
    def post_density(text):
        if pd.isna(text) or not text:
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0
        matches = sum(1 for kw in TOXIC_KEYWORDS if kw.lower() in text.lower())
        return matches / len(words)

    df_viz['toxicity_density'] = df_viz['POST_TEXT'].apply(post_density)
    df_viz['is_toxic'] = df_viz['toxicity_density'] > TOXICITY_THRESHOLD

    theme_toxicity = df_viz.groupby('NARRATIVE_TAG').agg(
        Avg_Density=('toxicity_density', 'mean'),
        Pct_Toxic_Posts=('is_toxic', 'mean')
    ).reset_index()
    theme_toxicity['Pct_Toxic_Posts'] *= 100
    theme_toxicity = theme_toxicity.sort_values('Avg_Density', ascending=False)
    return df_viz, theme_toxicity

# --- API Call with Backoff ---
def call_grok_with_backoff(payload, api_key, max_retries=5):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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

# --- Seed Classification (LLM) ---
def classify_post_for_seed(post_text, themes_list, api_key):
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
    if response_text and (response_text.strip() in themes_list or response_text.strip() == CLASSIFICATION_DEFAULT):
        return response_text.strip()
    return CLASSIFICATION_DEFAULT

# --- Narrative Extraction (LLM) ---
def analyze_narratives(corpus, api_key):
    system_prompt = (
        "You are a world-class social media narrative analyst. Your task is to analyze the provided corpus of "
        "social media posts (from Meltwater) and identify the 3 to 5 most significant, high-level, emerging "
        "discussion themes or sub-narratives. "
        "Output your findings as a single JSON array of objects. "
        "CRITICAL: Ensure the `narrative_title` for each theme is highly specific, descriptive, and suitable "
        "for use as a label on a bar chart."
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
    with st.spinner("Generating narrative themes from a sample of posts..."):
        json_response = call_grok_with_backoff(payload, api_key)
    if json_response:
        try:
            parsed = json.loads(json_response)
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response from Grok. Check the API output.")
            st.code(json_response)
            return None

        normalized = normalize_narratives(parsed)
        if not normalized:
            st.error("Narrative parse/normalization produced no usable themes. Showing raw model output for debugging.")
            st.code(json_response)
            return None
        return normalized
    return None

# --- Theme Explanations (LLM) ---
def generate_theme_explanations(corpus_sample: str, theme_titles: list[str], api_key: str):
    """
    Produce 2–3 sentence explanations for each theme title.
    Returns list of {'narrative_title': str, 'explanation': str}
    """
    if not theme_titles:
        return []

    titles_json = json.dumps(theme_titles, ensure_ascii=False)
    system_prompt = (
        "You are an experienced OSINT analyst. Based ONLY on the provided corpus sample, "
        "write 2–3 sentence, factual explanations for each theme title. "
        "Do not speculate. Do not add sources. Return a JSON array of objects with "
        "keys: narrative_title, explanation. The narrative_title must exactly match the input title."
    )
    user_query = (
        f"Corpus sample:\n{corpus_sample}\n\n"
        f"Theme titles (JSON array): {titles_json}"
    )
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
                        "explanation": {"type": "STRING"}
                    },
                    "required": ["narrative_title", "explanation"]
                }
            }
        },
        "temperature": 0.3
    }
    with st.spinner("Drafting brief explanations for each theme..."):
        json_response = call_grok_with_backoff(payload, api_key)
    if not json_response:
        return []

    try:
        data = json.loads(json_response)
    except json.JSONDecodeError:
        st.warning("Could not parse theme explanations JSON; skipping.")
        return []

    # Normalize to exact mapping and preserve order of theme_titles
    exp_map = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                t = str(item.get("narrative_title", "")).strip()
                e = str(item.get("explanation", "")).strip()
                if t and e:
                    exp_map[t] = e

    result = []
    for t in theme_titles:
        result.append({"narrative_title": t, "explanation": exp_map.get(t, "")})
    return result

# --- Hybrid Classification (LLM seed + LinearSVC) ---
def train_and_classify_hybrid(df_full, theme_titles, api_key):
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

# --- Data Crunching for Takeaways ---
def perform_data_crunching_and_summary(df_classified: pd.DataFrame) -> str:
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
                "items": {"type": "STRING"},
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

# --- Visualization Helpers ---
def plot_stacked_author_share(df_classified, author_col, theme_col, engagement_col, top_n=5):
    theme_total_likes = df_classified.groupby(theme_col)[engagement_col].sum().rename('Theme_Total').reset_index()
    df_grouped = (df_classified
                  .groupby([theme_col, author_col])[engagement_col]
                  .sum()
                  .reset_index(name='Author_Likes'))
    df_top = (df_grouped
              .sort_values([theme_col, 'Author_Likes'], ascending=[True, False])
              .groupby(theme_col, as_index=False)
              .head(top_n))
    df_top = df_top.merge(theme_total_likes, on=theme_col, how='left')
    df_top['Percentage'] = np.where(df_top['Theme_Total'] > 0,
                                    (df_top['Author_Likes'] / df_top['Theme_Total']) * 100,
                                    0.0)
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
        height=550
    )
    return fig

def plot_overall_author_ranking(df_classified, author_col, engagement_col, top_n=10):
    df_classified['Theme_Rank'] = df_classified.groupby([author_col, 'NARRATIVE_TAG'])['POST_TEXT'].transform('count')
    df_classified = df_classified.sort_values(by=['Theme_Rank'], ascending=False)
    df_primary_theme = df_classified.drop_duplicates(subset=[author_col], keep='first')[[author_col, 'NARRATIVE_TAG']].rename(columns={'NARRATIVE_TAG': 'Primary_Theme'})
    overall_metrics = df_classified.groupby(author_col).agg(Total_Likes=(engagement_col, 'sum'),
                                                            Total_Posts=('POST_TEXT', 'size')).reset_index()
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
    fig = finalize_figure(fig, title=f"Top {top_n} influencers by total likes", height=550)
    return fig

def plot_theme_influencer_share(df_viz, theme, author_col, engagement_col, top_n=5):
    df_theme = df_viz[df_viz['NARRATIVE_TAG'] == theme].copy()
    if df_theme.empty:
        return None
    df_grouped = df_theme.groupby(author_col)[engagement_col].sum().reset_index(name='Author_Likes')
    df_top = df_grouped.nlargest(top_n, 'Author_Likes')
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
        height=380
    )
    return fig

def plot_toxicity_by_theme(theme_toxicity):
    fig = px.bar(
        theme_toxicity,
        x='NARRATIVE_TAG',
        y='Avg_Density',
        title='Toxicity Density by Narrative Theme',
        labels={'Avg_Density': 'Average Toxic Keyword Density', 'NARRATIVE_TAG': 'Narrative Theme'},
        height=500,
        color='Avg_Density',
        color_continuous_scale='Reds'
    )
    def wrap_labels_bar(text):
        return '<br>'.join(textwrap.wrap(text, 15))
    fig.update_traces(marker_line_width=0)
    fig.update_layout(
        xaxis={
            'categoryorder': 'total descending',
            'tickangle': 0,
            'automargin': True,
            'tickfont': {'size': 12},
            'tickvals': theme_toxicity['NARRATIVE_TAG'].tolist(),
            'ticktext': [wrap_labels_bar(t) for t in theme_toxicity['NARRATIVE_TAG']],
        },
        showlegend=False,
        coloraxis_colorbar=dict(title="Density")
    )
    fig = finalize_figure(
        fig,
        title="Toxicity density by narrative theme",
        subtitle="Average proportion of angry/hateful/violent keywords per post; sorted descending",
        height=500
    )
    return fig

# --- Session State ---
if 'df_full' not in st.session_state:
    st.session_state.df_full = None
if 'narrative_data' not in st.session_state:
    st.session_state.narrative_data = None
if 'theme_titles' not in st.session_state:
    st.session_state.theme_titles = []
if 'theme_explanations' not in st.session_state:
    st.session_state.theme_explanations = None
if 'classified_df' not in st.session_state:
    st.session_state.classified_df = None
if 'data_summary_text' not in st.session_state:
    st.session_state.data_summary_text = None
if 'takeaways_list' not in st.session_state:
    st.session_state.takeaways_list = None
if 'corpus_sample' not in st.session_state:
    st.session_state.corpus_sample = None

# --- API Key ---
XAI_KEY = os.getenv(XAI_API_KEY_ENV_VAR)
st.session_state.api_key = XAI_KEY

# --- Title ---
st.title("X Narrative Analysis Dashboard")
st.markdown("Automated thematic extraction and quantitative analysis of Meltwater data. Please note that it takes about 90 seconds to fully load and display all data.")
st.caption("_Developed by Trish Bailey_")

# --- Sidebar: Upload + Download ---
with st.sidebar:
    if not st.session_state.api_key:
        st.error(f"FATAL ERROR: Grok API Key not found. Please set the '{XAI_API_KEY_ENV_VAR}' environment variable.")
    st.markdown("#### File Upload")
    uploaded_file = st.file_uploader(
        "Upload Meltwater Data (.xlsx or .csv)",
        type=['xlsx', 'csv'],
        help="Upload your Meltwater export as Excel (.xlsx) or CSV (.csv)."
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

# --- Early Exit if Missing ---
if not st.session_state.api_key or uploaded_file is None:
    if uploaded_file is None:
        st.info("Upload your Meltwater Data (.xlsx or .csv) in the sidebar to begin the analysis.")
    st.stop()

# --- Robust Loader for Meltwater exports ---
REQUIRED_COLS = TEXT_COLUMNS + [ENGAGEMENT_COLUMN, AUTHOR_COLUMN, DATE_COLUMN, TIME_COLUMN]

def _has_required(df: pd.DataFrame) -> bool:
    cols = set(df.columns.astype(str).str.strip())
    return all(c in cols for c in REQUIRED_COLS)

def load_meltwater(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    is_csv = name.endswith(".csv")

    # First pass: header=0, no row skipping
    if is_csv:
        uploaded.seek(0)
        df0 = pd.read_csv(uploaded, dtype=str)
    else:
        uploaded.seek(0)
        df0 = pd.read_excel(uploaded, engine='openpyxl')

    # If required columns not present, try a single-row skip fallback
    if not _has_required(df0):
        if is_csv:
            uploaded.seek(0)
            df1 = pd.read_csv(uploaded, header=None, dtype=str)
            # Promote first row to header
            df1.columns = df1.iloc[0].astype(str).str.strip()
            df1 = df1.iloc[1:].reset_index(drop=True)
        else:
            uploaded.seek(0)
            df1 = pd.read_excel(uploaded, engine='openpyxl', skiprows=1)

        if _has_required(df1):
            df = df1
        else:
            have = set(df0.columns.astype(str).str.strip())
            missing = [c for c in REQUIRED_COLS if c not in have]
            raise ValueError(
                "File is missing essential columns after header detection.\n"
                f"Required: {', '.join(REQUIRED_COLS)}\nMissing: {', '.join(missing)}"
            )
    else:
        df = df0

    # Clean header names
    df.columns = df.columns.astype(str).str.strip()

    # Quick guardrail: signal obviously skipped headers
    if any(str(c).startswith("Unnamed") for c in df.columns):
        st.warning("Detected 'Unnamed' columns. The header row may be malformed in this file.")

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

    # Robust DATETIME parsing
    df[DATE_COLUMN] = df[DATE_COLUMN].astype(str).str.strip()
    df[TIME_COLUMN] = df[TIME_COLUMN].astype(str).str.strip()

    # Case A: Date already contains full datetime
    dt_primary = pd.to_datetime(df[DATE_COLUMN], format=DATE_TIME_FORMAT, errors='coerce')

    # Case B: Fallback to Date + Time with inference
    needs_b = dt_primary.isna()
    if needs_b.any():
        combined = (df.loc[needs_b, DATE_COLUMN] + " " + df.loc[needs_b, TIME_COLUMN]).str.strip()
        dt_fallback = pd.to_datetime(combined, errors='coerce')  # handles "4:56 PM", "09:22:00", etc.
        dt_primary = dt_primary.where(~needs_b, dt_fallback)

    df["DATETIME"] = dt_primary

    # Diagnostics
    parse_success = df["DATETIME"].notna().mean() * 100
    if parse_success < 90:
        st.warning(f"Datetime parsing succeeded for only {parse_success:.1f}% of rows. Check sample values in Date/Time.")
    else:
        st.success(f"Datetime parsing complete: {parse_success:.1f}% success rate ({df['DATETIME'].notna().sum():,} valid datetimes).")

    # Drop rows with truly empty text
    df = df[df["POST_TEXT"].str.strip().astype(bool)]

    return df

# --- Main App Logic (Automatic step-chaining) ---
with st.container():
    # 0) Load file (once)
    if st.session_state.df_full is None:
        try:
            st.info("Loading file...")
            df = load_meltwater(uploaded_file)

            st.session_state.df_full = df.copy()
            st.success("File uploaded successfully!")

            data_rows = df.shape[0]
            valid_dates = df['DATETIME'].dropna()
            date_min = valid_dates.min().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"
            date_max = valid_dates.max().strftime('%Y-%m-%d') if not valid_dates.empty else "N/A"

            st.markdown("#### File Data Summary")
            st.markdown(f"- **Total Rows Processed:** {data_rows:,}\n- **Date Span:** {date_min} to {date_max}")
            st.markdown("---")

            # Prepare a fixed sample corpus for LLM steps (narratives + explanations)
            df_sample = st.session_state.df_full.sample(min(MAX_POSTS_FOR_ANALYSIS, len(st.session_state.df_full)), random_state=42)
            st.session_state.corpus_sample = ' | '.join(df_sample['POST_TEXT'].tolist())

            # Trigger rerun to continue automatically
            st.rerun()
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.session_state.df_full = None
            st.stop()

    # 1) Narratives Extraction (auto)
    if st.session_state.df_full is not None and st.session_state.narrative_data is None:
        narrative_list = analyze_narratives(st.session_state.corpus_sample, st.session_state.api_key)
        if narrative_list:
            st.session_state.narrative_data = narrative_list
            st.session_state.theme_titles = [
                item.get('narrative_title') for item in narrative_list if item.get('narrative_title')
            ]
        st.rerun()

    # 1a) Theme Explanations (auto, not used for charts)
    if st.session_state.narrative_data is not None and st.session_state.theme_explanations is None:
        st.session_state.theme_explanations = generate_theme_explanations(
            st.session_state.corpus_sample,
            st.session_state.theme_titles,
            st.session_state.api_key
        )
        st.rerun()

    # Present identified themes + explanations
    if st.session_state.narrative_data is not None:
        st.header("Narratives Extraction")
        st.subheader("Identified Narrative Themes")
        # Build a mapping for explanations for quick lookup
        exp_map = {}
        if st.session_state.theme_explanations:
            for item in st.session_state.theme_explanations:
                t = item.get('narrative_title', '')
                e = item.get('explanation', '')
                if t:
                    exp_map[t] = e

        for i, narrative in enumerate(st.session_state.narrative_data):
            title = narrative.get('narrative_title', f"Theme {i+1}")
            short_summary = narrative.get('summary', '').strip()
            st.markdown(f"**{i+1}. {title}**")
            # Show either the LLM-provided summary or the generated explanation (prefer explanation if present)
            long_expl = exp_map.get(title, "")
            if long_expl:
                st.markdown(long_expl)
            elif short_summary:
                st.markdown(short_summary)
        st.success("Themes identified. Proceeding to full-dataset tagging and analytics...")
        st.markdown("---")

    # 2) Data Analysis by Narrative (auto: classification)
    if (st.session_state.narrative_data is not None and
        st.session_state.classified_df is None and
        st.session_state.theme_titles):
        df_classified = train_and_classify_hybrid(
            st.session_state.df_full,
            st.session_state.theme_titles,
            st.session_state.api_key
        )
        if df_classified is not None:
            st.session_state.classified_df = df_classified
        st.rerun()

    # 3) Visualization + Insights (auto)
    if st.session_state.classified_df is not None and not st.session_state.classified_df.empty:
        st.header("Data Analysis by Narrative")

        df_classified = st.session_state.classified_df

        # Filter for visualization (exclude Other/Unrelated and require valid dates)
        df_viz = df_classified[
            (df_classified['NARRATIVE_TAG'] != CLASSIFICATION_DEFAULT) &
            (df_classified['DATETIME'].notna())
        ].copy()

        if df_viz.empty:
            st.warning("No posts were classified into the primary narrative themes OR no posts had valid date information. The dashboard cannot be generated.")
        else:
            # Compute toxicity scores
            df_viz, theme_toxicity = compute_toxicity_scores(df_viz)

            st.subheader("Narrative Analysis Dashboard")

            # 1) Post Volume by Theme (Bar)
            st.markdown("### Post Volume by Theme")
            theme_metrics = df_viz.groupby('NARRATIVE_TAG').agg(Post_Volume=('POST_TEXT', 'size')).reset_index()
            theme_metrics = theme_metrics.sort_values(by='Post_Volume', ascending=False)
            fig_bar = px.bar(
                theme_metrics,
                x='NARRATIVE_TAG',
                y='Post_Volume',
                title='Post Volume by Theme',
                labels={'Post_Volume': 'Post Volume (Count)', 'NARRATIVE_TAG': 'Narrative Theme'},
                height=500,
                color='NARRATIVE_TAG',
                color_discrete_sequence=VIBRANT_QUAL
            )
            def wrap_labels_bar(text):
                return '<br>'.join(textwrap.wrap(text, 15))
            fig_bar.update_traces(marker_line_width=0)
            fig_bar.update_layout(
                xaxis={
                    'categoryorder': 'total descending',
                    'tickangle': 0,
                    'automargin': True,
                    'tickfont': {'size': 12},
                    'tickvals': theme_metrics['NARRATIVE_TAG'].tolist(),
                    'ticktext': [wrap_labels_bar(t) for t in theme_metrics['NARRATIVE_TAG']],
                },
                showlegend=False
            )
            fig_bar = finalize_figure(fig_bar, title="Post volume by narrative theme", subtitle="Sorted by total posts", height=500)
            st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})

            # 2) Narrative Volume Trend Over Time (Line)
            st.markdown("### Narrative Volume Trend Over Time")
            df_trends_theme = df_viz.groupby([df_viz['DATETIME'].dt.date, 'NARRATIVE_TAG']).size().reset_index(name='Post_Volume')
            df_trends_theme['DATETIME'] = pd.to_datetime(df_trends_theme['DATETIME'])
            if not df_trends_theme.empty and len(df_trends_theme['DATETIME'].dt.date.unique()) > 1:
                fig_line = px.line(
                    df_trends_theme,
                    x='DATETIME',
                    y='Post_Volume',
                    color='NARRATIVE_TAG',
                    title='Volume Trend by Narrative Theme',
                    labels={'Post_Volume': 'Daily Post Volume', 'DATETIME': 'Date', 'NARRATIVE_TAG': 'Theme'},
                    height=550,
                    color_discrete_sequence=VIBRANT_QUAL
                )
                fig_line.update_traces(mode="lines", line={"width": 3})
                fig_line.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Daily posts",
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
                )
                fig_line = finalize_figure(fig_line, title="Narrative volume over time", subtitle="Daily volume, by theme", height=520)
                st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})
            else:
                st.warning("Trend chart requires posts spanning at least two unique days. Chart cannot be generated with current data.")

            # 3) Toxicity Density by Theme
            st.markdown("### Toxicity Density by Narrative Theme")
            fig_tox = plot_toxicity_by_theme(theme_toxicity)
            st.plotly_chart(fig_tox, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Based on {len(TOXIC_KEYWORDS)} keywords; density = toxic matches / total words per post.")

            # 4) Influencer Share per Theme (Top authors only)
            st.markdown("### Influencer Share of Engagement (Top Authors Only)")
            for theme in df_viz['NARRATIVE_TAG'].unique():
                fig_theme = plot_theme_influencer_share(df_viz, theme, AUTHOR_COLUMN, ENGAGEMENT_COLUMN, top_n=5)
                if fig_theme:
                    st.plotly_chart(fig_theme, use_container_width=True, config={"displayModeBar": False})

            # 5) Overall Top Authors by Likes
            st.markdown("### Top 10 Overall Authors by Total Likes")
            fig_overall = plot_overall_author_ranking(df_classified, AUTHOR_COLUMN, ENGAGEMENT_COLUMN, top_n=10)
            st.plotly_chart(fig_overall, use_container_width=True, config={"displayModeBar": False})

        # Auto-generate Insights (Takeaways)
        st.markdown("---")
        st.header("Insights from the Data")
        if st.session_state.takeaways_list is None:
            st.session_state.data_summary_text = perform_data_crunching_and_summary(st.session_state.classified_df)
            st.session_state.takeaways_list = generate_takeaways(st.session_state.data_summary_text, st.session_state.api_key)
            st.rerun()

        if st.session_state.takeaways_list:
            st.subheader("Executive Summary: 5 Key Takeaways")
            for i, takeaway in enumerate(st.session_state.takeaways_list):
                st.markdown(f"**{i+1}.** {takeaway}")

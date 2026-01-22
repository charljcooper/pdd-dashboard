import streamlit as st
import pandas as pd
import math
from pathlib import Path
import pathlib

app_code = r'''
# Run this once in Colab:
!pip install streamlit vaderSentiment scikit-learn wordcloud pandas numpy matplotlib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title='Reddit Topics & Sentiment (PD Discourse)', layout='wide')

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'created_datetime' in df.columns:
        df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce')
    return df

@st.cache_resource(show_spinner=False)
def get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()

@st.cache_resource(show_spinner=False)
def fit_lda_cached(
    texts_tuple,
    n_topics: int,
    max_features: int,
    min_df: int,
    max_df: float,
    random_state: int
):
    # texts_tuple used to make caching stable
    texts = list(texts_tuple)

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r'(?u)\\b[a-zA-Z]{2,}\\b'
    )
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method='batch'
    )
    doc_topic = lda.fit_transform(X)
    vocab = np.array(vectorizer.get_feature_names_out())
    return lda, doc_topic, vocab

def top_words_for_topic(lda, vocab, topic_idx: int, n_top_words: int = 15):
    weights = lda.components_[topic_idx]
    top_idx = np.argsort(weights)[::-1][:n_top_words]
    return vocab[top_idx], weights[top_idx]

def render_wordcloud(words, weights=None):
    if weights is not None:
        freqs = {w: float(s) for w, s in zip(words, weights)}
    else:
        freqs = {w: 1.0 for w in words}

    wc = WordCloud(width=1000, height=500, background_color='white')
    wc.generate_from_frequencies(freqs)

    fig = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    return fig

@st.cache_data(show_spinner=False)
def compute_vader_scores_cached(texts_tuple):
    vader = get_vader()
    return [vader.polarity_scores(t)['compound'] for t in texts_tuple]


st.title('Reddit Topic & Sentiment Dashboard (Computed in-app)')
st.caption('Dataset: cln_reddit_comments.csv | Text field: extra_cleaned_text | Topics: LDA | Sentiment: VADER')

df = load_data('cln_reddit_comments.csv')

required_col = 'extra_cleaned_text'
if required_col not in df.columns:
    st.error(f"Missing required column: '{required_col}'. Available columns: {list(df.columns)}")
    st.stop()

# --- Sidebar controls ---
with st.sidebar:
    st.header('Controls')

    # Language filter
    if 'lang' in df.columns:
        available_langs = sorted([x for x in df['lang'].dropna().unique()])
        default_lang = ['en'] if 'en' in available_langs else (available_langs[:1] if available_langs else [])
        lang_filter = st.multiselect('Language(s)', available_langs, default=default_lang)
    else:
        lang_filter = []

    # Text length filter
    min_len = st.slider('Minimum text length', min_value=0, max_value=500, value=10, step=5)

    st.divider()
    st.subheader('Topic Modelling (LDA)')

    # ✅ Faster defaults for Colab
    n_topics = st.slider('Number of topics (k)', min_value=2, max_value=20, value=6, step=1)
    topn_words = st.slider('Top words per topic', min_value=5, max_value=30, value=12, step=1)

    max_features = st.slider('Max vocabulary size', min_value=2000, max_value=50000, value=8000, step=1000)
    min_df = st.slider('min_df', min_value=1, max_value=20, value=10, step=1)
    max_df = st.slider('max_df', min_value=0.2, max_value=1.0, value=0.75, step=0.05)

    random_state = st.number_input('Random seed', value=42, step=1)

    st.divider()
    st.subheader('Performance')
    # We'll set sample slider after filtering so it is valid
    show_sentiment = st.checkbox('Compute and show sentiment (VADER)', value=True)


# --- Filter rows ---
df_f = df.copy()
df_f[required_col] = df_f[required_col].fillna('').astype(str)
df_f = df_f[df_f[required_col].str.len() >= min_len]

if lang_filter and 'lang' in df_f.columns:
    df_f = df_f[df_f['lang'].isin(lang_filter)]

# Provide transparency about dataset size (important for non-technical users)
st.markdown(
    f"**Rows loaded:** {len(df):,}  |  **Rows after filters:** {len(df_f):,}"
)

if len(df_f) < 50:
    st.warning('Not enough text after filtering to fit LDA. Reduce min length or language filters.')
    st.stop()

# Sample size slider based on filtered size
max_sample = int(min(30000, len(df_f)))
default_sample = int(min(2000, len(df_f)))

sample_n = st.sidebar.slider(
    'Sample size (used for LDA & sentiment)',
    min_value=200,
    max_value=max_sample,
    value=default_sample,
    step=200
)

# --- Sample for modelling ---
df_m = df_f.sample(n=min(sample_n, len(df_f)), random_state=int(random_state)).copy()
texts_tuple = tuple(df_m[required_col].tolist())  # tuple for caching stability

st.markdown(f"**Rows used for modelling (sample):** {len(df_m):,}")

# --- Fit LDA (cached) ---
with st.spinner('Fitting LDA topics...'):
    lda, doc_topic, vocab = fit_lda_cached(
        texts_tuple,
        int(n_topics),
        int(max_features),
        int(min_df),
        float(max_df),
        int(random_state)
    )

# Topic assignments
topic_assign = doc_topic.argmax(axis=1)
df_m = df_m.reset_index(drop=True)
df_m['topic'] = topic_assign
df_m['topic_prob'] = doc_topic.max(axis=1)

topic_counts = pd.Series(topic_assign).value_counts().sort_index()
topic_share = (topic_counts / topic_counts.sum()).rename('share').reset_index().rename(columns={'index': 'topic'})

# --- Layout ---
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader('Topic prevalence (sample)')
    fig = plt.figure(figsize=(8, 4))
    plt.bar(topic_share['topic'].astype(str), topic_share['share'])
    plt.xlabel('Topic')
    plt.ylabel('Share of comments')
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader('Top words per topic')
    rows = []
    for t in range(int(n_topics)):
        words, _ = top_words_for_topic(lda, vocab, t, int(topn_words))
        rows.append({'topic': t, 'top_words': ', '.join(words)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.subheader('Explore a topic')

sel_topic = st.selectbox('Select topic', list(range(int(n_topics))), index=0)
words, weights = top_words_for_topic(lda, vocab, int(sel_topic), int(topn_words))

wc_col, ex_col = st.columns([1.0, 1.2])

with wc_col:
    st.markdown('**Topic word cloud (top words weighted by LDA)**')
    st.pyplot(render_wordcloud(words, weights), clear_figure=True)

with ex_col:
    st.markdown('**Example comments (highest topic probability)**')
    ex = df_m[df_m['topic'] == int(sel_topic)].sort_values('topic_prob', ascending=False)
    show_cols = [c for c in ['post_id', 'comment_id', 'score', 'created_datetime', required_col, 'topic_prob'] if c in ex.columns]
    st.dataframe(ex[show_cols].head(20), use_container_width=True)

# --- Sentiment ---
if show_sentiment:
    st.divider()
    st.subheader('Sentiment (VADER)')

    with st.spinner('Computing VADER sentiment for sampled comments...'):
        df_m['vader_compound'] = compute_vader_scores_cached(texts_tuple)

    s1, s2 = st.columns([1.0, 1.0])

    with s1:
        st.markdown('**Distribution of VADER compound scores (sample)**')
        fig2 = plt.figure(figsize=(8, 4))
        plt.hist(df_m['vader_compound'], bins=30)
        plt.xlabel('VADER compound (-1 to 1)')
        plt.ylabel('Frequency')
        st.pyplot(fig2, clear_figure=True)

    with s2:
        st.markdown('**Average sentiment by topic (sample)**')
        topic_sent = df_m.groupby('topic')['vader_compound'].mean().reset_index()
        fig3 = plt.figure(figsize=(8, 4))
        plt.bar(topic_sent['topic'].astype(str), topic_sent['vader_compound'])
        plt.xlabel('Topic')
        plt.ylabel('Mean VADER compound')
        st.pyplot(fig3, clear_figure=True)

    # Sentiment over time
    if 'created_datetime' in df_m.columns and df_m['created_datetime'].notna().any():
        st.markdown('**Sentiment over time (monthly, sample)**')
        tmp = df_m.dropna(subset=['created_datetime']).copy()
        tmp['month'] = tmp['created_datetime'].dt.to_period('M').dt.to_timestamp()
        monthly = tmp.groupby('month')['vader_compound'].mean().reset_index()

        fig4 = plt.figure(figsize=(10, 4))
        plt.plot(monthly['month'], monthly['vader_compound'])
        plt.xlabel('Month')
        plt.ylabel('Mean VADER compound')
        st.pyplot(fig4, clear_figure=True)
    else:
        st.info('created_datetime not available/parsable; skipping sentiment-over-time panel.')

st.caption('Notes: LDA + VADER are computed on a filtered sample for responsiveness. Increase sample size for stability once the dashboard loads quickly.')
'''

pathlib.Path('app.py').write_text(app_code)
print('Wrote improved app.py')


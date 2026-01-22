import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Reddit Topics & Sentiment (PD Discourse)", layout="wide")

TEXT_COL = "extra_cleaned_text"


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_resource(show_spinner=False)
def get_vader() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@st.cache_resource(show_spinner=False)
def fit_lda_cached(
    texts_tuple: tuple[str, ...],
    n_topics: int,
    max_features: int,
    min_df: int,
    max_df: float,
    random_state: int,
):
    texts = list(texts_tuple)

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
    )
    X = vectorizer.fit_transform(texts)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method="batch",
    )
    doc_topic = lda.fit_transform(X)
    vocab = np.array(vectorizer.get_feature_names_out())
    return lda, doc_topic, vocab


def top_words_for_topic(lda, vocab, topic_idx: int, n_top_words: int):
    weights = lda.components_[topic_idx]
    top_idx = np.argsort(weights)[::-1][:n_top_words]
    return vocab[top_idx], weights[top_idx]


def render_wordcloud(words, weights):
    freqs = {w: float(s) for w, s in zip(words, weights)}
    wc = WordCloud(width=1000, height=500, background_color="white")
    wc.generate_from_frequencies(freqs)

    fig = plt.figure(figsize=(12, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    return fig


@st.cache_data(show_spinner=False)
def compute_vader_scores_cached(texts_tuple: tuple[str, ...]):
    vader = get_vader()
    return [vader.polarity_scores(t)["compound"] for t in texts_tuple]


# ---------------- UI ----------------
st.title("Reddit Topic & Sentiment Dashboard")
st.caption(
    "Upload a CSV. The app computes LDA topics + VADER sentiment in-app (no precomputed topic/sentiment columns required)."
)

uploaded = st.file_uploader("Upload your Reddit CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

df = load_csv(uploaded)

if TEXT_COL not in df.columns:
    st.error(f"Missing required text column: '{TEXT_COL}'")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Parse datetime if present
if "created_datetime" in df.columns:
    df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")

with st.sidebar:
    st.header("Controls")

    # Filters
    min_len = st.slider("Minimum text length", 0, 500, 10, 5)

    if "lang" in df.columns:
        langs = sorted([x for x in df["lang"].dropna().unique()])
        default_lang = ["en"] if "en" in langs else (langs[:1] if langs else [])
        lang_filter = st.multiselect("Language(s)", langs, default=default_lang)
    else:
        lang_filter = []

    st.divider()
    st.subheader("Topic Modelling (LDA)")

    n_topics = st.slider("Number of topics (k)", 2, 20, 6, 1)
    topn_words = st.slider("Top words per topic", 5, 30, 12, 1)

    max_features = st.slider("Max vocabulary size", 2000, 50000, 8000, 1000)
    min_df = st.slider("min_df", 1, 20, 10, 1)
    max_df = st.slider("max_df", 0.2, 1.0, 0.75, 0.05)

    random_state = st.number_input("Random seed", value=42, step=1)

    st.divider()
    show_sentiment = st.checkbox("Compute and show sentiment (VADER)", value=True)

# -------- Filtering --------
df_f = df.copy()
df_f[TEXT_COL] = df_f[TEXT_COL].fillna("").astype(str)
df_f = df_f[df_f[TEXT_COL].str.len() >= min_len]

if lang_filter and "lang" in df_f.columns:
    df_f = df_f[df_f["lang"].isin(lang_filter)]

st.markdown(f"**Rows loaded:** {len(df):,}  |  **Rows after filters:** {len(df_f):,}")

if len(df_f) < 50:
    st.warning("Not enough text after filtering to fit LDA. Reduce filters.")
    st.stop()

# Sample size based on filtered size
max_sample = int(min(30000, len(df_f)))
default_sample = int(min(2000, len(df_f)))

sample_n = st.sidebar.slider(
    "Sample size (used for LDA & sentiment)",
    min_value=200,
    max_value=max_sample,
    value=default_sample,
    step=200,
)

df_m = df_f.sample(n=min(sample_n, len(df_f)), random_state=int(random_state)).copy()
texts_tuple = tuple(df_m[TEXT_COL].tolist())
st.markdown(f"**Rows used for modelling (sample):** {len(df_m):,}")

# -------- LDA --------
with st.spinner("Fitting LDA topics..."):
    lda, doc_topic, vocab = fit_lda_cached(
        texts_tuple, int(n_topics), int(max_features), int(min_df), float(max_df), int(random_state)
    )

topic_assign = doc_topic.argmax(axis=1)
df_m = df_m.reset_index(drop=True)
df_m["topic"] = topic_assign
df_m["topic_prob"] = doc_topic.max(axis=1)

topic_counts = pd.Series(topic_assign).value_counts().sort_index()
topic_share = (topic_counts / topic_counts.sum()).rename("share").reset_index().rename(columns={"index": "topic"})

col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Topic prevalence (sample)")
    fig = plt.figure(figsize=(8, 4))
    plt.bar(topic_share["topic"].astype(str), topic_share["share"])
    plt.xlabel("Topic")
    plt.ylabel("Share of comments")
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Top words per topic")
    rows = []
    for t in range(int(n_topics)):
        words, _ = top_words_for_topic(lda, vocab, t, int(topn_words))
        rows.append({"topic": t, "top_words": ", ".join(words)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

st.divider()
st.subheader("Explore a topic")

sel_topic = st.selectbox("Select topic", list(range(int(n_topics))), index=0)
words, weights = top_words_for_topic(lda, vocab, int(sel_topic), int(topn_words))

wc_col, ex_col = st.columns([1.0, 1.2])

with wc_col:
    st.markdown("**Topic word cloud (weighted by LDA term importance)**")
    st.pyplot(render_wordcloud(words, weights), clear_figure=True)

with ex_col:
    st.markdown("**Example comments (highest topic probability)**")
    ex = df_m[df_m["topic"] == int(sel_topic)].sort_values("topic_prob", ascending=False)

    show_cols = [c for c in ["post_id", "comment_id", "score", "created_datetime", TEXT_COL, "topic_prob"] if c in ex.columns]
    st.dataframe(ex[show_cols].head(20), use_container_width=True)

# -------- Sentiment --------
if show_sentiment:
    st.divider()
    st.subheader("Sentiment (VADER)")

    with st.spinner("Computing VADER sentiment for sampled comments..."):
        df_m["vader_compound"] = compute_vader_scores_cached(texts_tuple)

    s1, s2 = st.columns([1.0, 1.0])

    with s1:
        st.markdown("**Distribution of VADER compound scores (sample)**")
        fig2 = plt.figure(figsize=(8, 4))
        plt.hist(df_m["vader_compound"], bins=30)
        plt.xlabel("VADER compound (-1 to 1)")
        plt.ylabel("Frequency")
        st.pyplot(fig2, clear_figure=True)

    with s2:
        st.markdown("**Average sentiment by topic (sample)**")
        topic_sent = df_m.groupby("topic")["vader_compound"].mean().reset_index()

        fig3 = plt.figure(figsize=(8, 4))
        plt.bar(topic_sent["topic"].astype(str), topic_sent["vader_compound"])
        plt.xlabel("Topic")
        plt.ylabel("Mean VADER compound")
        st.pyplot(fig3, clear_figure=True)

    if "created_datetime" in df_m.columns and df_m["created_datetime"].notna().any():
        st.markdown("**Sentiment over time (monthly, sample)**")
        tmp = df_m.dropna(subset=["created_datetime"]).copy()
        tmp["month"] = tmp["created_datetime"].dt.to_period("M").dt.to_timestamp()
        monthly = tmp.groupby("month")["vader_compound"].mean().reset_index()

        fig4 = plt.figure(figsize=(10, 4))
        plt.plot(monthly["month"], monthly["vader_compound"])
        plt.xlabel("Month")
        plt.ylabel("Mean VADER compound")
        st.pyplot(fig4, clear_figure=True)

st.caption(
    "Notes: LDA + VADER are computed on a filtered sample for responsiveness. Increase sample size for stability once deployment is stable."
)

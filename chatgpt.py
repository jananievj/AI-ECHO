import os
import pickle
import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="AI Echo — Sentiment Dashboard", layout="wide")
#  LOAD MODEL
model_path = r"C:\Users\ADMIN\Desktop\AI ECHO\NLP_model.pkl"
try:
    nlp_model = pickle.load(open(model_path, "rb"))
    st.sidebar.success("✅ Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Model loading failed: {e}")
    nlp_model = None

# ──────────────────────────────────────────────
# 🧭 SIDEBAR MENU
# ──────────────────────────────────────────────
with st.sidebar:
    selected = option_menu(
        "AI Echo: Your Smartest Conversational Partner",
        ["💬 AI Echo Sentiment Analysis", "📈 Sentiment Analysis"]
    )

# ──────────────────────────────────────────────
# 💬 AI ECHO SENTIMENT ANALYSIS PAGE
# ──────────────────────────────────────────────
if selected == "💬 AI Echo Sentiment Analysis":
    # 🎨 Custom background and text design
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #F0F4F8, #D9EAFD, #C2E9FB);
        color: black;
    }
    .title-style {
        font-weight: 900;
        color: #000000;
        text-transform: uppercase;
        font-size: 28px;
        text-align: center;
        margin-bottom: 15px;
    }
    .sub-style {
        font-weight: 700;
        color: #000000;
        text-transform: uppercase;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Title
    st.markdown('<div class="title-style">AI ECHO — YOUR SMARTEST CONVERSATIONAL PARTNER</div>', unsafe_allow_html=True)
    st.markdown('<p class="sub-style">Fill out the form below to analyze review sentiment.</p>', unsafe_allow_html=True)

    # Input Form
    with st.form("📋form"):
        title = st.text_input("TITLE")
        rating = st.selectbox("RATING", [1, 2, 3, 4, 5])
        username = st.text_input("USERNAME")
        platform = st.text_input("PLATFORM")
        language = st.text_input("LANGUAGE")
        location = st.text_input("LOCATION")
        verified_purchase = st.selectbox("VERIFIED PURCHASE", ["YES", "NO"])
        clean_review = st.text_input("CLEAN REVIEW")
        submitted = st.form_submit_button("PREDICT SENTIMENT")

    # Prediction
    if submitted and nlp_model:
        verified_bool = 1 if verified_purchase == "YES" else 0

        input_df = pd.DataFrame([{
            'title': title,
            'rating': rating,
            'username': username,
            'platform': platform,
            'language': language,
            'location': location,
            'verified_purchase': verified_bool,
            'clean_review': clean_review
        }])

        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                input_df[col] = input_df[col].astype('category').cat.codes

        prediction = nlp_model.predict(input_df)

        st.markdown("---")
        st.markdown('<div class="sub-style">SENTIMENT PREDICTION</div>', unsafe_allow_html=True)

        if prediction[0] == 2 or str(prediction[0]).lower() == "positive":
            st.markdown("<h3 style='color:#008000;'>POSITIVE SENTIMENT DETECTED</h3>", unsafe_allow_html=True)
        elif prediction[0] == 1 or str(prediction[0]).lower() == "neutral":
            st.markdown("<h3 style='color:#DAA520;'>NEUTRAL SENTIMENT DETECTED</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:#B22222;'>NEGATIVE SENTIMENT DETECTED</h3>", unsafe_allow_html=True)
# ──────────────────────────────────────────────
# 📊 SENTIMENT ANALYSIS DASHBOARD PAGE
# ──────────────────────────────────────────────
elif selected == "📈 Sentiment Analysis":
    st.title("📊 AI Echo Sentiment Dashboard")

    # Load dataset safely
    try:
        df = pd.read_csv("C:/Users/ADMIN/Desktop/AI ECHO/cleaned chatgpt.csv")
    except FileNotFoundError:
        st.error("🚫 File not found. Please check the path and try again.")
        st.stop()

    # Styling
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1, h2, h3, h4 {
        color: #F6F6F6 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    sns.set_style("darkgrid")
    sns.set_palette("bright")
    neon_colors = ["#FF66B3", "#FFD700", "#40E0D0", "#A569BD", "#FF6F61", "#58D68D", "#F5B041"]

    # ==================== Q1 ====================
    st.header("1️⃣ Overall Sentiment of User Reviews")
    sentiment_counts = df['sentiment'].value_counts()
    sentiment_percent = (sentiment_counts / len(df)) * 100
    st.dataframe(sentiment_percent.round(2).astype(str) + "%")

    fig1, ax1 = plt.subplots(figsize=(6,6), facecolor="#0E1117")
    wedges, texts, autotexts = ax1.pie(
        sentiment_counts,
        labels=sentiment_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=neon_colors[:len(sentiment_counts)],
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    for t in texts + autotexts:
        t.set_color('white')
    ax1.set_title("Overall Sentiment", color='white', fontsize=14)
    st.pyplot(fig1)
   # ==================== Q2 ====================
    st.header("2️⃣ Sentiment vs Rating — Are Ratings Consistent?")
    fig2, ax2 = plt.subplots(figsize=(10,6), facecolor="#0E1117")
    sns.violinplot(data=df, x='rating', y='review_length', hue='sentiment',
               palette=['#FF69B4', '#FFD700', '#00FFFF'], ax=ax2, split=True, inner='quartile')
    ax2.set_title("Sentiment Spread Across Ratings", color='white')
    ax2.set_xlabel("Rating", color='white')
    ax2.set_ylabel("Review Length", color='white')
    ax2.tick_params(colors='white')
    st.pyplot(fig2)
   # ==================== Q3 ====================
    st.header("3️⃣ Keyword Cloud for Each Sentiment")
    sentiment_classes = df['sentiment'].unique()
    selected_sentiment = st.selectbox("Select Sentiment Class", sentiment_classes)
    filtered_df = df[df['sentiment'] == selected_sentiment]
    all_words = " ".join(filtered_df['clean_review'].dropna().astype(str))
    wordcloud = WordCloud(width=900, height=400, background_color='#0E1117',
                      colormap='spring', contour_color='white', contour_width=1).generate(all_words)
    st.image(wordcloud.to_array(), caption=f"🌈 Word Cloud — {selected_sentiment.title()} Reviews")
   # ==================== Q4 ====================
    st.header("4️⃣ Verified vs Non-Verified User Sentiments")
    fig4, ax4 = plt.subplots(figsize=(8,5), facecolor="#0E1117")
    sns.scatterplot(data=df, x='sentiment', y='rating', hue='verified_purchase',
                palette=['#FF69B4', '#00FFFF'], s=120, alpha=0.8, ax=ax4, edgecolor='white')
    ax4.set_title("Verified vs Non-Verified Sentiment Pattern", color='white')
    ax4.tick_params(colors='white')
    st.pyplot(fig4)
   # ==================== Q5 ====================
    st.header("5️⃣ Review Length vs Sentiment Distribution")
    df['review_length'] = df['clean_review'].astype(str).apply(len)
    fig5, ax5 = plt.subplots(figsize=(8,5), facecolor="#0E1117")
    sns.boxplot(data=df, x='sentiment', y='review_length',
            palette=['#FF69B4', '#FFD700', '#00FFFF'], ax=ax5, linewidth=1.5)
    ax5.set_title("Review Length Variation by Sentiment", color='white')
    ax5.tick_params(colors='white')
    st.pyplot(fig5)
   # ==================== Q6 ====================
    st.header("6️⃣ Top 10 Locations with Sentiment Mix")
    top_locations = df['location'].value_counts().head(10).index
    location_data = df[df['location'].isin(top_locations)]
    fig6, ax6 = plt.subplots(figsize=(10,6), facecolor="#0E1117")
    sns.stripplot(data=location_data, x='location', y='rating', hue='sentiment',
              jitter=True, palette=['#FF69B4', '#FFD700', '#00FFFF'], dodge=True, ax=ax6)
    plt.xticks(rotation=45, color='white')
    ax6.set_title("Sentiment Scatter by Top 10 Locations", color='white')
    ax6.tick_params(colors='white')
    st.pyplot(fig6)
   # ==================== Q7 ====================
    st.header("7️⃣ Sentiment by Platform")
    fig7, ax7 = plt.subplots(figsize=(8,5), facecolor="#0E1117")

    sns.barplot(data=df, x='platform', y='rating', hue='sentiment',
            palette='husl', ax=ax7)

    ax7.set_title("Sentiment Comparison Across Platforms", color='white')
    plt.xticks(rotation=45, color='white')
    ax7.tick_params(colors='white')
    st.pyplot(fig7)
   # ==================== Q8 ====================
    st.header("8️⃣ ChatGPT Versions & Sentiment Patterns") 
    fig8, ax8 = plt.subplots(figsize=(10,5), facecolor="#0E1117") 
    sns.stripplot(data=df, x='version', y='rating', hue='sentiment', palette='coolwarm', jitter=True, dodge=True, ax=ax8) 
    plt.xticks(rotation=45, color='white') 
    ax8.set_title("Version-wise Sentiment Spread", color='white') 
    ax8.tick_params(colors='white') 
    st.pyplot(fig8)
    # ==================== Q9 ====================
    st.header("9️⃣ Common Negative Feedback Themes")
    neg_reviews = df[df['sentiment'].str.lower() == 'negative']['clean_review'].dropna().astype(str)
    if len(neg_reviews) > 0:
     vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(neg_reviews)
    word_counts = X.sum(axis=0).A1
    keywords = vectorizer.get_feature_names_out()
    word_freq = dict(zip(keywords, word_counts))
    top_keywords = Counter(word_freq).most_common(15)
    top_df = pd.DataFrame(top_keywords, columns=['Keyword', 'Frequency'])
    fig9, ax9 = plt.subplots(figsize=(8,5), facecolor="#0E1117")
    sns.barplot(data=top_df, x='Frequency', y='Keyword', palette='magma', ax=ax9)
    ax9.set_title("🔥 Top Negative Feedback Keywords", color='white')
    ax9.tick_params(colors='white')
    st.pyplot(fig9)
else:
    st.warning("No negative reviews found for keyword extraction.")
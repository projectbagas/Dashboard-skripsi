import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
import string
import nltk
from nltk.corpus import stopwords

# ---------------------------
# Setup NLTK
# ---------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# ---------------------------
# Halaman Streamlit
# ---------------------------
st.set_page_config(page_title="Dashboard Analisis Kepuasan Maxim", layout="wide")
st.sidebar.title("Menu")
menu = ["Dashboard", "Dataset", "Model Klasifikasi", "Implementasi Algoritma"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

# ---------------------------
# Load Data dan rename kolom otomatis
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("maxim_reviews.csv")  # pastikan CSV ada di repo
    df.rename(columns={df.columns[0]: 'review_text', df.columns[1]: 'sentiment'}, inplace=True)
    return df

data = load_data()

# ---------------------------
# Preprocessing Teks
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

data['cleaned_review'] = data['review_text'].apply(clean_text)

# ---------------------------
# Encode Label
# ---------------------------
le = LabelEncoder()
data['sentiment_encoded'] = le.fit_transform(data['sentiment'])

# ---------------------------
# Menu 1: Dashboard
# ---------------------------
if choice == "Dashboard":
    st.title("Dashboard Analisis Kepuasan Pengguna Maxim")
    
    # Jumlah total dataset
    st.subheader("Jumlah Total Dataset")
    st.metric("Jumlah Ulasan", len(data))
    
    # Distribusi Sentimen
    st.subheader("Distribusi Sentimen")
    sentiment_counts = data['sentiment'].value_counts()

    # Bar Chart
    fig, ax = plt.subplots()
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="pastel", ax=ax)
    ax.set_xlabel("Sentimen")
    ax.set_ylabel("Jumlah Ulasan")
    st.pyplot(fig)

    # Pie Chart
    fig2, ax2 = plt.subplots()
    ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
            colors=['#8BC34A','#FFC107','#F44336'])
    ax2.set_title("Distribusi Sentimen")
    st.pyplot(fig2)

    # Word Cloud per Sentimen di Tab
    st.subheader("Word Cloud Per Sentimen")
    def generate_wordcloud(text):
        return WordCloud(width=800, height=400, background_color='white').generate(' '.join(text))
    
    tab1, tab2, tab3 = st.tabs(["Puas", "Netral", "Tidak Puas"])
    tabs = [tab1, tab2, tab3]
    categories = ["Puas", "Netral", "Tidak Puas"]
    
    for tab, sentiment in zip(tabs, categories):
        with tab:
            subset = data[data['sentiment'] == sentiment]
            if not subset.empty:
                wc = generate_wordcloud(subset['cleaned_review'])
                fig_wc, ax_wc = plt.subplots(figsize=(10,5))
                ax_wc.imshow(wc, interpolation='bilinear')
                ax_wc.axis('off')
                st.pyplot(fig_wc)
            else:
                st.write(f"Tidak ada data untuk kategori {sentiment}")

# ---------------------------
# Menu 2: Dataset
# ---------------------------
elif choice == "Dataset":
    st.title("Dataset Mentah & Preprocessed")
    st.dataframe(data[['review_text','sentiment','cleaned_review','sentiment_encoded']])

# ---------------------------
# Menu 3: Model Klasifikasi
# ---------------------------
elif choice == "Model Klasifikasi":
    st.title("Penjelasan Model Klasifikasi")
    st.markdown("""
    **XGBoost**  
    - Algoritma boosting berbasis pohon keputusan.  
    - Optimalkan prediksi melalui iterasi dan pengurangan error.  

    **Random Forest**  
    - Algoritma ensemble learning berbasis banyak pohon keputusan.  
    - Voting mayoritas dari semua pohon keputusan.  

Kedua algoritma digunakan untuk **mengklasifikasikan tingkat kepuasan pengguna** menjadi `Puas`, `Netral`, dan `Tidak Puas`.
    """)

# ---------------------------
# Menu 4: Implementasi Algoritma
# ---------------------------
elif choice == "Implementasi Algoritma":
    st.title("Implementasi dan Evaluasi Model")
    
    # Pilihan vectorizer
    vectorizer_type = st.selectbox("Pilih Vectorizer", ["CountVectorizer", "TfidfVectorizer"])
    if vectorizer_type == "CountVectorizer":
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()
    
    X_vect = vectorizer.fit_transform(data['cleaned_review'])
    y = data['sentiment_encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
    
    # XGBoost
    model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_xgb.fit(X_train, y_train)
    pred_xgb = model_xgb.predict(X_test)
    
    # Random Forest
    model_rf = RandomForestClassifier()
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)
    
    # Evaluasi
    def evaluate(y_true, y_pred):
        return {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, average='weighted'),
            "Recall": recall_score(y_true, y_pred, average='weighted'),
            "F1-Score": f1_score(y_true, y_pred, average='weighted')
        }
    
    eval_xgb = evaluate(y_test, pred_xgb)
    eval_rf = evaluate(y_test, pred_rf)
    
    st.subheader("Perbandingan Metrik Evaluasi")
    df_eval = pd.DataFrame({
        "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "XGBoost": [eval_xgb["Accuracy"], eval_xgb["Precision"], eval_xgb["Recall"], eval_xgb["F1-Score"]],
        "Random Forest": [eval_rf["Accuracy"], eval_rf["Precision"], eval_rf["Recall"], eval_rf["F1-Score"]]
    })
    st.dataframe(df_eval)
    
    # Confusion Matrix
    labels = le.classes_  # nama label asli
    st.subheader("Confusion Matrix XGBoost")
    cm_xgb = confusion_matrix(y_test, pred_xgb)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    
    st.subheader("Confusion Matrix Random Forest")
    cm_rf = confusion_matrix(y_test, pred_rf)
    fig_cm2, ax_cm2 = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax_cm2)
    ax_cm2.set_xlabel("Predicted")
    ax_cm2.set_ylabel("Actual")
    st.pyplot(fig_cm2)

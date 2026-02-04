import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
# Streamlit Config
# ---------------------------
st.set_page_config(page_title="Dashboard Analisis Kepuasan Maxim", layout="wide", page_icon="📊")
st.markdown("""
<style>
body {background-color: #1e1e2f; color: white;}
.sidebar .sidebar-content {background-color: #252535;}
h1, h2, h3, h4 {color: white;}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Sidebar Menu
# ---------------------------
st.sidebar.title("Menu")
menu = ["Dashboard", "Dataset", "Klasifikasi", "Metode Klasifikasi", "Implementasi Algoritma"]
choice = st.sidebar.selectbox("Pilih Menu", menu)

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("maxim_reviews.csv")
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
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

data['cleaned_review'] = data['review_text'].apply(clean_text)

# ---------------------------
# Label Encoding
# ---------------------------
le = LabelEncoder()
data['sentiment_encoded'] = le.fit_transform(data['sentiment'])

# ---------------------------
# Menu: Dashboard
# ---------------------------
if choice == "Dashboard":
    st.title("Dashboard Analisis Kepuasan Pengguna Maxim")
    
    # Metrics
    total = len(data)
    positif = data[data['sentiment'] == 'Puas'].shape[0]
    negatif = data[data['sentiment'] == 'Tidak Puas'].shape[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Ulasan", total)
    col2.metric("Ulasan Positif", positif)
    col3.metric("Ulasan Negatif", negatif)
    
    # Sentimen Polarity Pie Chart
    st.subheader("Sentimen Polarity")
    fig1, ax1 = plt.subplots()
    ax1.pie(data['sentiment'].value_counts(), labels=data['sentiment'].value_counts().index,
            autopct='%1.1f%%', colors=['#8BC34A','#FFC107','#F44336'])
    st.pyplot(fig1)
    
    # Word Cloud Positif & Negatif
    st.subheader("Word Cloud Ulasan Positif & Negatif")
    col_wc1, col_wc2 = st.columns(2)
    
    # Positif
    wc_pos = WordCloud(width=400, height=300, background_color='black',
                       colormap='spring').generate(' '.join(data[data['sentiment']=='Puas']['cleaned_review']))
    fig_wc1, ax_wc1 = plt.subplots()
    ax_wc1.imshow(wc_pos, interpolation='bilinear')
    ax_wc1.axis('off')
    col_wc1.pyplot(fig_wc1)
    
    # Negatif
    wc_neg = WordCloud(width=400, height=300, background_color='black',
                       colormap='autumn').generate(' '.join(data[data['sentiment']=='Tidak Puas']['cleaned_review']))
    fig_wc2, ax_wc2 = plt.subplots()
    ax_wc2.imshow(wc_neg, interpolation='bilinear')
    ax_wc2.axis('off')
    col_wc2.pyplot(fig_wc2)

# ---------------------------
# Menu: Dataset
# ---------------------------
elif choice == "Dataset":
    st.title("Dataset Mentah & Preprocessed")
    st.dataframe(data[['review_text','sentiment','cleaned_review','sentiment_encoded']])

# ---------------------------
# Menu: Klasifikasi
# ---------------------------
elif choice == "Klasifikasi":
    st.title("Penjelasan Model Klasifikasi")
    st.markdown("""
    **XGBoost**  
    - Boosting berbasis pohon keputusan.  
    - Optimalkan prediksi melalui iterasi.  

    **Random Forest**  
    - Ensemble dari banyak pohon keputusan.  
    - Voting mayoritas untuk prediksi akhir.  

Digunakan untuk **mengklasifikasikan tingkat kepuasan** (`Puas`, `Netral`, `Tidak Puas`) berdasarkan ulasan.
    """)

# ---------------------------
# Menu: Metode Klasifikasi
# ---------------------------
elif choice == "Metode Klasifikasi":
    st.title("Metode Klasifikasi")
    st.markdown("Menjelaskan proses preprocessing, vectorization, dan training model XGBoost & Random Forest.")

# ---------------------------
# Menu: Implementasi Algoritma
# ---------------------------
elif choice == "Implementasi Algoritma":
    st.title("Implementasi & Evaluasi Model")
    
    # Vectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data['cleaned_review'])
    y = data['sentiment_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    # Confusion Matrix XGBoost
    st.subheader("Confusion Matrix XGBoost")
    labels = le.classes_
    cm_xgb = confusion_matrix(y_test, pred_xgb)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)
    
    # Confusion Matrix RF
    st.subheader("Confusion Matrix Random Forest")
    cm_rf = confusion_matrix(y_test, pred_rf)
    fig_cm2, ax_cm2 = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, ax=ax_cm2)
    ax_cm2.set_xlabel("Predicted")
    ax_cm2.set_ylabel("Actual")
    st.pyplot(fig_cm2)

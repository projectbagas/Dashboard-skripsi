# app.py - DASHBOARD SKRIPSI MAXIM (LOAD CSV & PICKLE MODEL)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="XGBoost vs Random Forest - Maxim Skripsi",
    page_icon="📱",
    layout="wide"
)

# CSS styling seperti gambar
st.markdown("""
<style>
    .main-header {font-size: 2.5rem !important; font-weight: bold; color: #10B981; text-align: center;}
    .sub-header {font-size: 1.5rem !important; color: #1F2937; text-align: center;}
    .metric-card {background-color: #f0fdf4; padding: 1rem; border-radius: 10px; border-left: 5px solid #10B981;}
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load data CSV dan model pickle
@st.cache_data
def load_data_and_models():
    data = {}
    
    # 1. LOAD CSV RAW DATASET
    if os.path.exists('maxim_raw.csv'):
        data['df_raw'] = pd.read_csv('maxim_raw.csv')
    else:
        st.warning("📁 Upload maxim_raw.csv terlebih dahulu!")
        data['df_raw'] = pd.DataFrame()
    
    # 2. LOAD TEST DATASET (fitur + label)
    if os.path.exists('maxim_test_features.csv'):
        data['df_test'] = pd.read_csv('maxim_test_features.csv')
    else:
        data['df_test'] = pd.DataFrame()
    
    # 3. LOAD XGBoost MODEL
    if os.path.exists('xgboost_model.pkl'):
        with open('xgboost_model.pkl', 'rb') as f:
            data['xgb_model'] = pickle.load(f)
    else:
        data['xgb_model'] = None
    
    # 4. LOAD Random Forest MODEL
    if os.path.exists('randomforest_model.pkl'):
        with open('randomforest_model.pkl', 'rb') as f:
            data['rf_model'] = pickle.load(f)
    else:
        data['rf_model'] = None
    
    return data

# Load semua data dan model
data = load_data_and_models()
df_raw = data['df_raw']
df_test = data['df_test']
xgb_model = data['xgb_model']
rf_model = data['rf_model']

# Status file
st.sidebar.markdown("### 📁 **Status File**")
st.sidebar.metric("Raw Dataset", "✅" if len(df_raw)>0 else "❌")
st.sidebar.metric("Test Dataset", "✅" if len(df_test)>0 else "❌")
st.sidebar.metric("XGBoost Model", "✅" if xgb_model else "❌")
st.sidebar.metric("RF Model", "✅" if rf_model else "❌")

# UPLOAD FILE (jika belum ada)
if len(df_raw) == 0:
    st.warning("📤 **Upload file CSV dan model pickle Anda:**")
    uploaded_raw = st.file_uploader("maxim_raw.csv (review_text, label, rating)", type='csv')
    uploaded_test = st.file_uploader("maxim_test_features.csv (fitur + label)", type='csv')
    
    xgb_uploaded = st.file_uploader("xgboost_model.pkl", type=['pkl','pickle'])
    rf_uploaded = st.file_uploader("randomforest_model.pkl", type=['pkl','pickle'])

# SIDEBAR NAVIGATION
with st.sidebar:
    st.markdown("🏛️ **Universitas Nasional**")
    st.markdown("---")
    page = st.radio("Pilih Halaman:", [
        "📊 Dashboard", "📋 Dataset", "🔍 Model", "⚙️ Implementasi"
    ])

# === DASHBOARD ===
if page == "📊 Dashboard":
    st.markdown('<h1 class="main-header">Perbandingan XGBoost vs Random Forest</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Klasifikasi Tingkat Kepuasan Pengguna Aplikasi Maxim di Google Play Store</h2>', unsafe_allow_html=True)
    
    if len(df_raw) > 0:
        # METRIK 4 KOLOM
        col1, col2, col3, col4 = st.columns(4)
        total = len(df_raw)
        puas = len(df_raw[df_raw['label']=='puas'])
        netral = len(df_raw[df_raw['label']=='netral'])
        tidak_puas = len(df_raw[df_raw['label']=='tidak puas'])
        
        with col1: st.metric("Total Ulasan", f"{total:,}", "📱 Play Store")
        with col2: st.metric("Puas 👍", f"{puas:,}", f"{puas/total*100:.1f}%")
        with col3: st.metric("Netral ➡️", f"{netral:,}", f"{netral/total*100:.1f}%")
        with col4: st.metric("Tidak Puas 👎", f"{tidak_puas:,}", f"{tidak_puas/total*100:.1f}%")
        
        # PIE CHART
        col_pie, _ = st.columns([3,1])
        with col_pie:
            fig_pie = px.pie(values=[puas, netral, tidak_puas], 
                           names=['Puas 👍', 'Netral ➡️', 'Tidak Puas 👎'],
                           color_discrete_sequence=['#10B981', '#F59E0B', '#EF4444'], hole=0.4)
            fig_pie.update_traces(textposition='inside')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # PERBANDINGAN MODEL (dari test set)
        if len(df_test) > 0 and xgb_model and rf_model:
            st.subheader("⚔️ Perbandingan Prediksi")
            col_xgb, col_rf = st.columns(2)
            
            # Prediksi real-time
            X_test = df_test.drop('label', axis=1)
            y_test = df_test['label']
            
            xgb_pred = xgb_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            
            with col_xgb:
                fig_xgb = px.bar(x=['Puas','Netral','Tidak Puas'], 
                               y=pd.Series(xgb_pred).value_counts().reindex([0,1,2], fill_value=0).values,
                               title="XGBoost", color_continuous_scale='teal', text_auto=True)
                st.plotly_chart(fig_xgb, height=350)
            
            with col_rf:
                fig_rf = px.bar(x=['Puas','Netral','Tidak Puas'], 
                              y=pd.Series(rf_pred).value_counts().reindex([0,1,2], fill_value=0).values,
                              title="Random Forest", color_continuous_scale='blues', text_auto=True)
                st.plotly_chart(fig_rf, height=350)
        
        # WORDCLOUD
        if 'review_text' in df_raw.columns:
            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                text_pos = ' '.join(df_raw[df_raw['label']=='puas']['review_text'].dropna())
                wc_pos = WordCloud(width=500,height=400,colormap='Greens').generate(text_pos)
                fig, ax = plt.subplots(figsize=(6,5))
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Kata Positif', color='#10B981', fontsize=16)
                st.pyplot(fig)
            
            with col_wc2:
                text_neg = ' '.join(df_raw[df_raw['label']=='tidak puas']['review_text'].dropna())
                wc_neg = WordCloud(width=500,height=400,colormap='Reds').generate(text_neg)
                fig, ax = plt.subplots(figsize=(6,5))
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Kata Negatif', color='#EF4444', fontsize=16)
                st.pyplot(fig)

# === DATASET ===
elif page == "📋 Dataset":
    st.header("📋 Dataset Ulasan Maxim")
    if len(df_raw) > 0:
        st.dataframe(df_raw, use_container_width=True)
    else:
        st.info("📤 Upload maxim_raw.csv terlebih dahulu")

# === MODEL ===
elif page == "🔍 Model":
    st.header("🔍 Model Klasifikasi")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### **XGBoost**")
        st.markdown("- Gradient Boosting berbasis tree\n- Sequential learning\n- Regularisasi L1/L2\n- GPU support")
    with col2:
        st.markdown("### **Random Forest**")
        st.markdown("- Ensemble Bagging\n- Multiple decision trees\n- Robust terhadap noise\n- Feature importance")

# === IMPLEMENTASI ===
elif page == "⚙️ Implementasi":
    st.header("⚙️ Evaluasi Model")
    if len(df_test) > 0 and xgb_model and rf_model:
        X_test = df_test.drop('label', axis=1)
        y_test = df_test['label']
        
        # Label Encoder
        le = LabelEncoder()
        y_test_enc = le.fit_transform(y_test)
        
        # Prediksi & Metrik
        xgb_pred = xgb_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("XGBoost Akurasi", f"{accuracy_score(y_test_enc, xgb_model.predict(X_test)):.2%}")
        with col2:
            st.metric("RF Akurasi", f"{accuracy_score(y_test_enc, rf_model.predict(X_test)):.2%}")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm_xgb = confusion_matrix(y_test_enc, xgb_pred)
        cm_rf = confusion_matrix(y_test_enc, rf_pred)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        sns.heatmap(cm_xgb, annot=True, fmt='d', ax=ax1, cmap='Blues')
        ax1.set_title('XGBoost')
        sns.heatmap(cm_rf, annot=True, fmt='d', ax=ax2, cmap='Greens')
        ax2.set_title('Random Forest')
        st.pyplot(fig)
    else:
        st.info("📤 Upload model pickle & test dataset terlebih dahulu")

st.markdown("---")
st.markdown("*Dashboard Skripsi Maxim - Universitas Nasional*")

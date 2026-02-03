import streamlit as st
import pandas as pd
import joblib  # Mengganti pickle agar tidak error EOF/Unpickling
import re
import os

# Set Page Config
st.set_page_config(page_title="Skripsi Bagas - Maxim Analisis", layout="wide")

# Load model dan tfidf yang sudah disimpan
@st.cache_resource # Agar loading lebih cepat
def load_assets():
    # Cek apakah file ada sebelum di-load untuk menghindari crash
    required = ['model_xgb.pkl', 'model_rf.pkl', 'tfidf_vectorizer.pkl']
    for f in required:
        if not os.path.exists(f):
            st.error(f"File {f} tidak ditemukan di GitHub!")
            st.stop()
            
    # Menggunakan joblib karena lebih stabil di server Cloud daripada pickle
    model_xgb = joblib.load('model_xgb.pkl')
    model_rf = joblib.load('model_rf.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model_xgb, model_rf, tfidf

# Menjalankan fungsi load
try:
    xgb_model, rf_model, tfidf = load_assets()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()

# --- HEADER ---
st.title("📊 Dashboard Analisis Kepuasan Pengguna Maxim")
st.markdown("Oleh: **Bagas Dwi Ardianto** (217006516109)")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("📂 Data Source")
uploaded_file = st.sidebar.file_uploader("Upload maxim_reviews.csv", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Pre-processing sederhana untuk visualisasi
    def simple_label(score):
        if score >= 4: return 'Puas'
        elif score == 3: return 'Netral'
        else: return 'Tidak Puas'
    df['label'] = df['score'].apply(simple_label)

    # --- LAYOUT UTAMA ---
    tab1, tab2, tab3 = st.tabs(["📈 Statistik Data", "⚖️ Perbandingan Model", "🔍 Uji Sentimen"])

    with tab1:
        st.subheader("Distribusi Sentimen Pengguna")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.write("10 Data Pertama:")
            st.dataframe(df[['content', 'label']].head(10))
        with col_b:
            st.bar_chart(df['label'].value_counts())

    with tab2:
        st.subheader("Hasil Evaluasi (Sesuai Bab 4)")
        c1, c2 = st.columns(2)
        # Angka disesuaikan dengan isi skripsi Anda
        c1.metric("Akurasi XGBoost", "93%", "Lebih Unggul")
        c2.metric("Akurasi Random Forest", "80%", "-13%", delta_color="inverse")
        
        st.table(pd.DataFrame({
            'Metrik': ['Presisi', 'Recall', 'F1-Score'],
            'XGBoost': [0.88, 0.93, 0.90],
            'Random Forest': [0.73, 0.80, 0.77]
        }))

    with tab3:
        st.subheader("Live Sentiment Testing")
        raw_text = st.text_area("Masukkan ulasan baru untuk diprediksi:")
        method = st.radio("Pilih Algoritma:", ("XGBoost", "Random Forest"))

        if st.button("Analisis Sekarang"):
            if raw_text:
                # 1. Transform teks input
                clean_input = re.sub(r'[^a-z\s]', '', raw_text.lower())
                vec_input = tfidf.transform([clean_input])
                
                # 2. Predict
                if method == "XGBoost":
                    res = xgb_model.predict(vec_input)[0]
                else:
                    res = rf_model.predict(vec_input)[0]
                
                # 3. Map result (Sesuaikan dengan urutan LabelEncoder Anda)
                # Umumnya: 0: Tidak Puas, 1: Netral, 2: Puas
                labels = {0: "Tidak Puas ❌", 1: "Netral 😐", 2: "Puas ✅"}
                st.success(f"Hasil Prediksi ({method}): {labels[res]}")
            else:
                st.warning("Silahkan masukkan teks terlebih dahulu.")

else:
    st.info("Silahkan upload file `maxim_reviews.csv` pada sidebar untuk memulai.")
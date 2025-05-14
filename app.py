from pywebio.input import input, select
from pywebio.output import put_text, put_markdown, put_error, put_html, put_scope, use_scope, clear
from pywebio import start_server
import joblib
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Uygulamayı çalıştırmak için terminalde aşağıdaki komutu kullanın:
# python app.py
# Uygulama çalıştıktan sonra tarayıcınızda aşağıdaki bağlantıyı açabilirsiniz:
# http://192.168.2.212:8086/
# Not: PyWebIO ile çalıştığı için tarayıcıda "güvenli değil" uyarısı görünebilir devam et butonuna tıklayarak uygulamayı açabilirsiniz.
# (Not: Bu bağlantı terminalde çalıştırıldığında bir kez gösterildiği için buraya not olarak ekledim.)
# Siteye girmek istemezseniz sitenin bazı görüntülerini yandaki images kısmında bulabilirsiniz.

# --- CEFR MODEL YÜKLE ---
def load_cefr_model():
    model = joblib.load('CEFR_model/my_cefr_model.pkl') #DİKKAT :sitenin çalışması için kendı model yolunuza göre değiştirmelisiniz
    tfidf = joblib.load('CEFR_model/my_tfidf_vectorizer.pkl') #DİKKAT :sitenin çalışması için kendı model yolunuza göre değiştirmelisiniz
    le = joblib.load('CEFR_model/my_label_encoder.pkl') #DİKKAT :sitenin çalışması için kendı model yolunuza göre değiştirmelisiniz
    return model, tfidf, le

# CEFR ön işleme
def preprocess_cefr(text):
    return text.lower().strip()

# CEFR tahmin
def predict_cefr(sentence, model, tfidf, le):
    processed = preprocess_cefr(sentence)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)[0]
    return le.inverse_transform(prediction)[0], {level: prob for level, prob in zip(le.classes_, proba)}

# --- TENSE MODEL YÜKLE ---
model_tense = tf.keras.models.load_model("tense_model/tense_classifier.h5") #DİKKAT :sitenin çalışması için kendı model yolunuza göre değiştirmelisiniz
with open("tense_model/tokenizer.pickle", "rb") as handle: #DİKKAT :sitenin çalışması için kendı model yolunuza göre değiştirmelisiniz
    tokenizer = pickle.load(handle)
max_length = 50

# Tense ön işleme
def preprocess_tense(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# Tense tahmin
def predict_tense(sentence):
    processed = preprocess_tense(sentence)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model_tense.predict(pad)[0]
    predicted_class = np.argmax(pred)
    return predicted_class + 1, pred

# --- CSS & HTML ---
def render_css():
    put_html("""
    <style>
        body {
            background: linear-gradient(to right, #FFDEE9, #B5FFFC);
            font-family: 'Segoe UI', sans-serif;
            padding: 20px;
            color: #333;
        }
        .result-box {
            background-color: rgba(255,255,255,0.9);
            border-radius: 12px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .result-box h3 {
            color: #FF6347;
        }
        .result-box p {
            font-size: 16px;
        }
    </style>
    """)

# --- ANA UYGULAMA ---
def main():
    render_css()
    try:
        cefr_model, tfidf, le = load_cefr_model()
    except Exception as e:
        put_error(f"CEFR model dosyaları yüklenemedi: {str(e)}")
        return

    put_markdown("## 🧠 İngilizce Cümle Analiz Sistemi")
    put_scope("results")

    while True:
        choice = select("🔍 Hangi analiz türünü yapmak istersiniz?", ["CEFR Tahmini", "Zaman (Tense) Tahmini"])
        
        sentence = input("✏️ İngilizce bir cümle girin (çıkmak için boş bırakın):")
        if not sentence:
            break

        if choice == "CEFR Tahmini":
            try:
                level, probabilities = predict_cefr(sentence, cefr_model, tfidf, le)
                with use_scope("results", clear=False):
                    put_html(f"""
                        <div class='result-box'>
                            <h3>✏️ Girilen Cümle:</h3>
                            <p>{sentence}</p>
                            <h3>🎯 CEFR Seviyesi:</h3>
                            <p><strong>{level}</strong></p>
                            <h3>📊 Olasılıklar:</h3>
                            {"".join(f"<p>{lvl}: %{prob*100:.2f}</p>" for lvl, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True))}
                        </div>
                    """)
            except Exception as e:
                put_error(f"Hata oluştu: {str(e)}")

        elif choice == "Zaman (Tense) Tahmini":
            try:
                tense_index, confidence = predict_tense(sentence)
                tense_map = {1: "Present Tense", 2: "Past Tense", 3: "Future Tense"}
                with use_scope("results", clear=False):
                    put_html(f"""
                        <div class='result-box'>
                            <h3>✏️ Girilen Cümle:</h3>
                            <p>{sentence}</p>
                            <h3>🎯 Zaman Tahmini:</h3>
                            <p><strong>{tense_map[tense_index]}</strong></p>
                            <h3>📊 Güven Düzeyleri:</h3>
                            <p>Present: %{confidence[0]*100:.1f}</p>
                            <p>Past: %{confidence[1]*100:.1f}</p>
                            <p>Future: %{confidence[2]*100:.1f}</p>
                        </div>
                    """)
            except Exception as e:
                put_error(f"Hata oluştu: {str(e)}")

# Başlat
if __name__ == '__main__':
    start_server(main, port=8086, debug=True)

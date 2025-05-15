# Uygulamayı çalıştırmak için terminalde aşağıdaki komutu kullanın:
# python app.py
# Uygulama çalıştıktan sonra tarayıcınızda aşağıdaki bağlantıyı açabilirsiniz:
# http://172.20.10.3:8086/
# Not: PyWebIO ile çalıştığı için tarayıcıda "güvenli değil" uyarısı görünebilir devam et butonuna tıklayarak uygulamayı açabilirsiniz.
# (Not: Bu bağlantı terminalde çalıştırıldığında bir kez gösterildiği için buraya not olarak ekledim.)
# Siteye girmek istemezseniz sitenin bazı görüntülerini yandaki images kısmında bulabilirsiniz.
from pywebio.input import input, select
from pywebio.output import put_text, put_markdown, put_error, put_html, put_scope, use_scope, clear
from pywebio import start_server
import joblib
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import base64

def get_background_image_base64(path='Ingiltere.jpg'):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

b64_string = get_background_image_base64()

# --- CEFR MODEL YÜKLE ---
def load_cefr_model():
    model = joblib.load('CEFR_model/my_cefr_model.pkl') 
    tfidf = joblib.load('CEFR_model/my_tfidf_vectorizer.pkl') 
    le = joblib.load('CEFR_model/my_label_encoder.pkl') 
    return model, tfidf, le

def preprocess_cefr(text):
    return text.lower().strip()

def predict_cefr(sentence, model, tfidf, le):
    processed = preprocess_cefr(sentence)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)[0]
    return le.inverse_transform(prediction)[0], {level: prob for level, prob in zip(le.classes_, proba)}

# --- TENSE MODEL YÜKLE ---
model_tense = tf.keras.models.load_model("tense_model/tense_classifier.h5") 
with open("tense_model/tokenizer.pickle", "rb") as handle: 
    tokenizer = pickle.load(handle)
max_length = 50

def preprocess_tense(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

def predict_tense(sentence):
    processed = preprocess_tense(sentence)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model_tense.predict(pad)[0]
    predicted_class = np.argmax(pred)
    return predicted_class + 1, pred

# --- CSS & HTML ---
def render_css():
    put_html(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

        body {{
            background: linear-gradient(rgba(80, 40, 100, 0.6), rgba(80, 40, 100, 0.6)), 
                        url("data:image/jpeg;base64,{b64_string}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            padding: 30px;
            color: #2c3e50;
        }}

        h1, h2, h3 {{
            font-weight: 600;
        }}

        .result-box {{
            background-color: #b1c4d3;  /* Pastel Mavi-Mor Tonu */
            border-left: 6px solid #9b59b6;
            border-radius: 10px;
            padding: 25px 30px;
            margin-top: 25px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease;
        }}

        .result-box:hover {{
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.12);
        }}

        .result-box h3 {{
            color: #6c3483;
            font-size: 20px;
            margin-bottom: 10px;
        }}

        .result-box p {{
            font-size: 16px;
            line-height: 1.6;
            color: #555;
            margin: 5px 0;
        }}

        .highlight {{
            font-weight: bold;
            color: #8e44ad;
        }}

        .proba-bar {{
            background-color: #f3e5f5;
            border-radius: 6px;
            overflow: hidden;
            margin-bottom: 10px;
        }}

        .proba-fill {{
            height: 16px;
            background-color: #9b59b6;
            text-align: right;
            padding-right: 5px;
            color: white;
            font-size: 12px;
            line-height: 16px;
        }}
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

    put_html("""
        <h2 style='text-align:center; color:#2c3e50;'>
            🧠 <span style="color:#3498db;">İngilizce Cümle Analiz Sistemi</span>
        </h2>
    """)
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
                            <p class='highlight'>{level}</p>
                            <h3>📊 Olasılıklar:</h3>
                            {"".join(f"<p>{lvl}</p><div class='proba-bar'><div class='proba-fill' style='width:{prob*100:.2f}%;'>{prob*100:.1f}%</div></div>" for lvl, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True))}
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
                            <p class='highlight'>{tense_map[tense_index]}</p>
                            <h3>📊 Güven Düzeyleri:</h3>
                            {"".join(f"<p>{label}</p><div class='proba-bar'><div class='proba-fill' style='width:{confidence[i]*100:.1f}%;'>{confidence[i]*100:.1f}%</div></div>" for i, label in enumerate(["Present", "Past", "Future"]))}
                        </div>
                    """)
            except Exception as e:
                put_error(f"Hata oluştu: {str(e)}")

# --- SUNUCU BAŞLAT ---
if __name__ == '__main__':
    start_server(main, port=8086, debug=True)
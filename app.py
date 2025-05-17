# Uygulamayı çalıştırmak için terminalde aşağıdaki komutu kullanın:
# python app.py
# Uygulama çalıştıktan sonra tarayıcınızda aşağıdaki bağlantıyı açabilirsiniz:
# http://172.20.10.3:8086/
# Not: PyWebIO ile çalıştığı için tarayıcıda "güvenli değil" uyarısı görünebilir devam et butonuna tıklayarak uygulamayı açabilirsiniz.
# (Not: Bu bağlantı terminalde çalıştırıldığında bir kez gösterildiği için buraya not olarak ekledim.)
# Siteye girmek istemezseniz sitenin bazı görüntülerini yandaki images kısmında bulabilirsiniz.
# --- Ortak ve Gerekli Kütüphaneler ---
import os
import re
import pickle
import random
import warnings
import base64
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pywebio.input import input, select, actions
from pywebio.output import put_text, put_markdown, put_error, put_html, put_scope, use_scope, clear
from pywebio import start_server

# --- Ortam Ayarları ---
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Arka Plan Görseli (Base64) ---
def get_background_image_base64(path='Ingiltere.jpg'):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

b64_string = get_background_image_base64()

# --- CSS ---
def render_css():
    put_html(f"""
    <style>
        body {{
            background: linear-gradient(rgba(80, 40, 100, 0.6), rgba(80, 40, 100, 0.6)), 
                        url("data:image/jpeg;base64,{b64_string}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Poppins', sans-serif;
            padding: 30px;
            color: #2c3e50;
        }}
        .result-box {{
            background-color: #b1c4d3;
            border-left: 6px solid #9b59b6;
            border-radius: 10px;
            padding: 25px 30px;
            margin-top: 25px;
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.08);
        }}
        .result-box h3 {{ color: #6c3483; }}
        .highlight {{ font-weight: bold; color: #8e44ad; }}
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

# --- CEFR MODEL ---
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

# --- TENSE MODEL ---
model_tense = tf.keras.models.load_model("tense_model/tense_classifier.h5") 
with open("tense_model/tokenizer.pickle", "rb") as handle: 
    tokenizer = pickle.load(handle)
max_length = 50

def preprocess_tense(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return re.sub(r'\d+', '', text).strip()

def predict_tense(sentence):
    processed = preprocess_tense(sentence)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    pred = model_tense.predict(pad)[0]
    return np.argmax(pred) + 1, pred

# --- TEST UYGULAMASI (DistilBERT ile) ---
class EnglishQuizWeb:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.questions = None
        self.load_resources()

    def load_resources(self):
        model_path = os.path.abspath(r"C:\Users\PC\OneDrive\Masaüstü\tense_cefr -analyzerr\english_exam_model")
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = TFDistilBertForSequenceClassification.from_pretrained(model_path)

        data_path = os.path.abspath(r"C:\Users\PC\OneDrive\Masaüstü\tense_cefr -analyzerr\english_exam_model\clean_questions.csv")
        self.questions = pd.read_csv(data_path)

        self.questions = self.questions.dropna(subset=['question'])
        # Sadece gerekli sütunları tut

    def get_random_question(self):
        row = self.questions.sample(1).iloc[0]
        return {
            'question': row['question'],
            'choices': {
                1: row['choice_1'],
                2: row['choice_2'],
                3: row['choice_3'],
                4: row['choice_4']
            },
            'answer': int(row['answer'])
        }

    def predict(self, question, choices):
        text = f"Soru: {question}\nSeçenekler:\n1) {choices[1]}\n2) {choices[2]}\n3) {choices[3]}\n4) {choices[4]}"
        inputs = self.tokenizer(text, return_tensors='tf', truncation=True, max_length=128)
        outputs = self.model(inputs)
        return tf.argmax(outputs.logits, axis=1).numpy()[0] + 1

    def run(self):
        clear()
        put_markdown("## 📘 İngilizce Test Uygulaması")
        while True:
            q = self.get_random_question()
            put_markdown(f"### ❓ {q['question']}")
            options = [f"{i}) {c}" for i, c in q['choices'].items()]
            selected = select("Cevabınızı seçin:", options=options)
            user_answer = int(selected[0])
            predicted = self.predict(q['question'], q['choices'])

            if user_answer == q['answer']:
                put_markdown("✅ **Doğru!**")
            else:
                put_markdown(f"❌ **Yanlış!** Doğru cevap: {q['answer']}) {q['choices'][q['answer']]}")

            put_markdown(f"🤖 **Model tahmini:** {predicted}) {q['choices'].get(predicted, 'N/A')}")
            next_step = actions("Devam?", ["Sonraki Soru", "Testi Bitir"])
            if next_step == "Testi Bitir":
                put_text("Test sona erdi. Başarılar!")
                break

# --- ANALİZ UYGULAMASI ---
def run_analysis_app():
    try:
        cefr_model, tfidf, le = load_cefr_model()
    except Exception as e:
        put_error(f"CEFR model yüklenemedi: {str(e)}")
        return

    put_html("<h2 style='text-align:center;'>🧠 İngilizce Cümle Analiz Sistemi</h2>")
    put_scope("results")
    while True:
        choice = select("🔍 Hangi analiz türünü yapmak istersiniz?", ["CEFR Tahmini", "Zaman (Tense) Tahmini"])
        sentence = input("✏️ İngilizce bir cümle girin (çıkmak için boş bırakın):")
        if not sentence:
            break

        if choice == "CEFR Tahmini":
            level, probs = predict_cefr(sentence, cefr_model, tfidf, le)
            with use_scope("results", clear=False):
                put_html(f"""
                <div class='result-box'>
                    <h3>✏️ Girilen Cümle:</h3><p>{sentence}</p>
                    <h3>🎯 CEFR Seviyesi:</h3><p class='highlight'>{level}</p>
                    <h3>📊 Olasılıklar:</h3>
                    {"".join(f"<p>{lvl}</p><div class='proba-bar'><div class='proba-fill' style='width:{p*100:.2f}%;'>{p*100:.1f}%</div></div>" for lvl, p in sorted(probs.items(), key=lambda x: x[1], reverse=True))}
                </div>
                """)

        elif choice == "Zaman (Tense) Tahmini":
            index, conf = predict_tense(sentence)
            tense_map = {1: "Present", 2: "Past", 3: "Future"}
            with use_scope("results", clear=False):
                put_html(f"""
                <div class='result-box'>
                    <h3>✏️ Girilen Cümle:</h3><p>{sentence}</p>
                    <h3>🎯 Zaman Tahmini:</h3><p class='highlight'>{tense_map[index]}</p>
                    <h3>📊 Güven Skorları:</h3>
                    {"".join(f"<p>{label}</p><div class='proba-bar'><div class='proba-fill' style='width:{conf[i]*100:.1f}%;'>{conf[i]*100:.1f}%</div></div>" for i, label in enumerate(['Present', 'Past', 'Future']))}
                </div>
                """)

# --- ANA MENÜ ---
def main():
    render_css()
    put_html("<h2 style='text-align:center;'>📚 <span style='color:#3498db;'>İngilizce Dil Uygulamaları</span></h2>")
    option = select("Lütfen bir uygulama seçin:", ["Cümle Analizi", "İngilizce Testi", "Çık"])
    
    if option == "Cümle Analizi":
        run_analysis_app()
    elif option == "İngilizce Testi":
        quiz = EnglishQuizWeb()
        quiz.run()
    else:
        put_text("Görüşmek üzere!")

# --- SUNUCU BAŞLAT ---
if __name__ == '__main__':
    start_server(main, port=8086, debug=True)

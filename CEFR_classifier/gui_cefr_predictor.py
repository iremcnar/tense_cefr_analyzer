#===================== WEB SİTESİ DISINDA KISACA GORMEK ICIN GUİ KODU =======================
import joblib
import tkinter as tk
from tkinter import messagebox, scrolledtext
# Model yükleme
def load_my_model():
    model = joblib.load(r'C:\Users\PC\OneDrive\Masaüstü\tense_cefr\CEFR_model\my_cefr_model.pkl') #DİKKAT kendı model yolunuza göre değiştirmelisiniz
    tfidf = joblib.load(r'C:\Users\PC\OneDrive\Masaüstü\tense_cefr\CEFR_model\my_tfidf_vectorizer.pkl') #DİKKAT kendı model yolunuza göre değiştirmelisiniz
    le = joblib.load(r'C:\Users\PC\OneDrive\Masaüstü\tense_cefr\CEFR_model\my_label_encoder.pkl') #DİKKAT kendı model yolunuza göre değiştirmelisiniz
    return model, tfidf, le

# Ön işleme
def preprocess_text(text):
    return text.lower().strip()

# Tahmin
def predict_with_my_model(sentence, model, tfidf, le):
    processed = preprocess_text(sentence)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)
    proba = model.predict_proba(vector)[0]
    return le.inverse_transform(prediction)[0], {level: prob for level, prob in zip(le.classes_, proba)}

# Tahmin sonucu gösterme
def on_predict():
    sentence = input_text.get("1.0", tk.END).strip()
    
    if not sentence:
        messagebox.showwarning("Uyarı", "Lütfen bir cümle girin!")
        return

    try:
        level, probabilities = predict_with_my_model(sentence, model, tfidf, le)
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Tahmin Edilen Seviye: {level}\n\nOlasılıklar:\n")
        
        for lvl, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            result_text.insert(tk.END, f"{lvl}: %{prob * 100:.2f}\n")
    
    except Exception as e:
        messagebox.showerror("Hata", f"Tahmin sırasında hata oluştu:\n{str(e)}")

# Ana pencere oluştur
root = tk.Tk()
root.title("CEFR Seviye Tahmin Sistemi")

# Giriş metni
tk.Label(root, text="İngilizce bir cümle girin:", font=('Arial', 12)).pack(pady=5)
input_text = scrolledtext.ScrolledText(root, width=60, height=4, font=('Arial', 11))
input_text.pack(padx=10, pady=5)

# Tahmin butonu
tk.Button(root, text="Tahmin Et", command=on_predict, font=('Arial', 12), bg="lightblue").pack(pady=10)

# Sonuç gösterimi
result_text = scrolledtext.ScrolledText(root, width=60, height=10, font=('Arial', 11))
result_text.pack(padx=10, pady=5)

# Model ve yardımcıları yükle
try:
    model, tfidf, le = load_my_model()
except Exception as e:
    messagebox.showerror("Model Hatası", f"Model veya vektörizer dosyaları yüklenemedi:\n{str(e)}")
    root.destroy()

# Uygulamayı çalıştır
root.mainloop()

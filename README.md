# İngilizce Cümle Analiz Sistemi 📚🧠

Bu proje, kullanıcıların İngilizce cümleleri girerek CEFR (Common European Framework of Reference) seviyesi ve zaman dilimi (tense) tahminleri yapabilmesini sağlar. Ayrıca, çoktan seçmeli İngilizce sorulara cevap veren bir Quiz Sistemi içerir. Tüm bunlar, Python tabanlı ve PyWebIO ile web arayüzü üzerinden çalışan interaktif bir uygulamada sunulmaktadır.

---
## Quiz Model Dosyası
"tf_model.h5" model dosyam büyük olduğu için GitHub'a eklenmemiştir. Yandaki linkten indirebilirsiniz:
[Modeli İndir](https://drive.google.com/file/d/1BYZ1QZatTGixWXNMrfijB9YHJY2ObbMP/view?usp=sharing)
Projeyi çalıştırabilmek için modeli "english_exam_model" klasörüne ekleyin.

## Özellikler 🌟

1. **CEFR Seviyesi Tahmini:** Kullanıcıların girdiği İngilizce cümleye göre dil seviyesi (A1, A2, B1, B2, C1, C2) tahmin edilir.
2. **Zaman (Tense) Tahmini:** Cümlede kullanılan zaman dilimi (Present, Past, Future) tahmin edilir.
3. **Quiz Sistemi:**  Çoktan seçmeli sorular ile İngilizce dilbilgisi ve kelime bilgisi ölçümü,BERT tabanlı derin öğrenme modeli ile doğru şıkkın tahmini
Kullanıcının cevaplarına göre anlık geri bildirim ve puanlama
4. **Web Arayüzü:** PyWebIO kullanılarak geliştirilmiş bir web arayüzü üzerinden kullanıcılar, uygulamaya cümle girer ve sonuçları görsel olarak alırlar.
5. **Hızlı ve Kullanıcı Dostu:** Sonuçlar, şık bir görsel ile birlikte kullanıcıya sunulur.

---

## Kullanılan Teknolojiler 💻

- **Python**
- **PyWebIO**: Web uygulaması için.
- **TensorFlow & Keras**: Derin öğrenme modellerinin eğitimi ve kullanımı (özellikle zaman tahmini ve quiz sınıflandırması için)
- **Scikit-learn**: CEFR seviyesi tahmini için.
- **HuggingFace Transformers**: BERT ve DistilBERT modellerinin kullanımı ve eğitimi (Quiz Sistemi için)
- **Joblib**: Modelleri ve vektörizer dosyalarını yüklemek için.
- **Pickle**: Tokenizer dosyasını yüklemek için.
- **HTML & CSS**: Web sayfası tasarımı için.




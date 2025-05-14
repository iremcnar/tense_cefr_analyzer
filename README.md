# İngilizce Cümle Analiz Sistemi 📚🧠

Bu proje, kullanıcıların İngilizce cümleleri girerek CEFR (Common European Framework of Reference) seviyesi ve zaman dilimi (tense) tahminleri yapabilmesini sağlar. PyWebIO ile geliştirilmiş olan bu uygulama, Python tabanlıdır ve web üzerinden çalışmaktadır. Kullanıcıların girdikleri cümleye göre İngilizce seviyeleri ve zaman dilimleri tahmin edilmektedir.

---

## Özellikler 🌟

1. **CEFR Seviyesi Tahmini:** Kullanıcıların girdiği İngilizce cümleye göre dil seviyesi (A1, A2, B1, B2, C1, C2) tahmin edilir.
2. **Zaman (Tense) Tahmini:** Cümlede kullanılan zaman dilimi (Present, Past, Future) tahmin edilir.
3. **Web Arayüzü:** PyWebIO kullanılarak geliştirilmiş bir web arayüzü üzerinden kullanıcılar, uygulamaya cümle girer ve sonuçları görsel olarak alırlar.
4. **Hızlı ve Kullanıcı Dostu:** Sonuçlar, şık bir görsel ile birlikte kullanıcıya sunulur.

---

## Kullanılan Teknolojiler 💻

- **Python**
- **PyWebIO**: Web uygulaması için.
- **TensorFlow**: Zaman sınıflandırma (Tense) için derin öğrenme modeli.
- **Scikit-learn**: CEFR seviyesi tahmini için.
- **Keras**: LSTM tabanlı zaman sınıflandırma modeli için.
- **Joblib**: Modelleri ve vektörizer dosyalarını yüklemek için.
- **Pickle**: Tokenizer dosyasını yüklemek için.
- **HTML & CSS**: Web sayfası tasarımı için.



- Python 3.x yüklü olmalıdır.
- Aşağıdaki komutla gerekli kütüphaneleri yükleyebilirsiniz:

```bash
pip install -r requirements.txt

# FRC 9029 Algae Algılama Sistemi

Merhaba! Bu repo, FRC 2025 oyunundaki algae'leri tespit etmek için geliştirdiğimiz görüntü işleme sistemini içerir. Docker üzerinde çalışan, hem PC'de geliştirme hem de Raspberry Pi'de yarışma sırasında kullanılabilen kompakt bir sistem kurduk.

## Özellikler

- 🎯 Algae algılamak için TensorFlow Lite tabanlı nesne tespiti
- 🔄 Nesne takibi (Kalman, IoU, SORT ve OpenCV algoritmaları)
- 📏 PnP algoritması ile 3B konum tespiti (mesafe ve açı hesaplama)
- 🌐 NetworkTables ile FRC robotu ile haberleşme
- 🖥️ Modern, şık web arayüzü (PC ve RPI'da çalışır)
- 🐋 Docker desteği ile kolay kurulum ve deployment
- 🔄 RoboRIO ile senkronizasyon

## Başlangıç

Sistemi iki modda çalıştırabilirsin: Test (PC) modu ve Yarışma (Raspberry Pi) modu.

### Test Modu (PC)

Geliştirme sırasında veya robotta test etmeden önce PC'de çalıştırmak için:

```bash
# Docker imajını oluştur ve çalıştır
docker-compose -f docker-compose.pc.yml up --build
```

Sonra tarayıcından şu adrese git: http://localhost:9029

Bu mod sahte algae tespitleri oluşturur, böylece gerçek bir kamera olmadan bile UI'yı test edebilirsin.

### Raspberry Pi Modu

Yarışma sırasında veya robotla çalışırken:

```bash
# Docker imajını oluştur ve çalıştır
docker-compose -f docker-compose.rpi.yml up --build
```

Raspberry Pi'nin IP adresinden erişim sağlayabilirsin: http://[rpi-ip]:9029

Bu mod EdgeTPU'yu kullanır (varsa) ve CameraServer entegrasyonu ile çalışır.

## Sistem Özellikleri (Detaylı)

### Algae Tespiti

Sistemimiz TensorFlow Lite kullanarak algae'leri tespit eder. Raspberry Pi'de daha iyi performans için Google Coral EdgeTPU desteği ekledik.

Tespit ayarlarını `config/pc_config.json` veya `config/rpi_config.json` dosyalarından düzenleyebilirsin:

```json
"detection": {
    "model_path": "models/model.tflite",
    "labels_path": "labels.txt",
    "threshold": 0.5,
    "use_coral": false,
    "input_size": [300, 300]
}
```

### Nesne Takibi

Sistem, tespit edilen algae'leri kamera görüntüsünde takip eder. Birkaç farklı algoritma arasından seçim yapabilirsin:

- **Kalman Filtresi**: Hız tahminli pozisyon takibi (varsayılan)
- **IoU Tracker**: Basit ve hızlı bounding box eşleştirmesi
- **SORT**: Daha karmaşık takip algoritması
- **OpenCV**: OpenCV'nin dahili takip algoritmaları

Takip algoritmasını config dosyasından değiştirebilirsin:

```json
"tracking": {
    "algorithm": "kalman",
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3
}
```

### 3B Konum Tespiti (PnP)

PnP (Perspective-n-Point) algoritması, bilinen boyuttaki nesnelerin 3B konumunu hesaplar. Algae'lerin fiziksel çapını bildiğimiz için (39 cm), sistemimiz:

- Kameradan mesafesini (cm)
- Yatay ve dikey açıları
- Tam 3B konum verilerini hesaplayabilir

Bu özelliği `--enable-pnp` parametresiyle etkinleştirebilirsin, veya UI üzerinden açabilirsin.

### NetworkTables Entegrasyonu

Robot koduna veri göndermek için NetworkTables kullanıyoruz. Tespit edilen algae'lerin konumları, mesafeleri ve açıları otomatik olarak gönderilir.

NetworkTables ayarlarını config dosyasından değiştirebilirsin:

```json
"networktables": {
    "team_number": 9029,
    "server_ip": "",
    "table_name": "VisionTracking"
}
```

## Web Arayüzü

Sistemimiz kullanıcı dostu bir web arayüzü içerir. Bu arayüz sayesinde:

- Kamera görüntüsünü canlı izleyebilirsin
- Algae tespitlerini ve takip bilgilerini görebilirsin
- Tüm ayarları grafik arayüzden değiştirebilirsin
- İşlem adımlarını (gri tonlama, kenar tespiti vb.) görebilirsin
- Karanlık ve aydınlık tema arasında geçiş yapabilirsin

Arayüz, takım numaramız olan 9029 portunda çalışır ve hem PC hem de Raspberry Pi'de kullanılabilir.

### Arayüz Özellikleri

- **Canlı Video Akışı**: Kamera görüntüsünü gerçek zamanlı izle
- **Tespit Bilgileri**: Algılanan nesnelerin listesi ve özellikleri
- **Ayarlar Menüsü**: Tüm sistem ayarlarını düzenle
- **İşlem Görselleştirmesi**: Görüntü işleme adımlarını ayrı ayrı gör
- **Tema Seçimi**: Koyu (siyah+yeşil) veya açık (gri+lacivert) tema

## Kendi Modelini Eğitme

Eğer kendi algae tespit modelini eğitmek istersen:

1. `training/collect_training_data.py` kullanarak veri topla
2. Verileri etiketle (Roboflow gibi bir araç kullanabilirsin)
3. TensorFlow Object Detection API ile model eğit
4. Modeli TFLite formatına dönüştür
5. EdgeTPU kullanıyorsan, EdgeTPU derleyicisi ile derle
6. Modeli `models/` klasörüne koyup config dosyasını güncelle

## Sorun Giderme

Eğer bir sorunla karşılaşırsan:

- Docker loglarını kontrol et
- UI'dan ayarları gözden geçir
- NetworkTables bağlantısını kontrol et
- Kamera erişimini doğrula

---

FRC Team 9029 tarafından 💚 ile yapıldı. 
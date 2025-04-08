# FRC 2025 - Algae and Coral Detector

Bu Docker container, [FRC 2025 - Algae and Coral](https://universe.roboflow.com/frc-team-503-frog-force/frc-2025-algae-and-coral) Roboflow modelini kullanarak kameranızla yosun ve mercan tespiti yapar.

## Gereksinimler

- Docker ve Docker Compose kurulu olmalı
- Bilgisayarınıza bağlı bir webcam veya kamera cihazı
- API anahtarı artık otomatik olarak dahil edilmiştir, ek yapılandırma gerekmez

## Kurulum

1. Bu repo'yu klonlayın
2. Containeri çalıştırın:

```bash
./run.sh
```

veya manuel olarak:

```bash
# X sunucusu bağlantılarına izin verme
xhost +local:docker
# Docker Compose ile çalıştırma
docker-compose up --build
```

## Yapılandırma

Kamera indeksini veya tespit eşiğini `docker-compose.yml` dosyasında değiştirebilirsiniz:

```yaml
command: --camera 1 --confidence 0.6
```

Burada:
- `--camera` kamera cihazının indeksidir (varsayılan: 0)
- `--confidence` tespit eşiğidir (varsayılan: 0.5)

## Farklı Bir Kamera Kullanmak

Kameranız `/dev/video0` konumunda değilse, doğru cihazı belirtmek için `docker-compose.yml` dosyasını değiştirin:

```yaml
devices:
  - /dev/video1:/dev/video0  # Kameranız /dev/video1'deyse
```

## X11 Ekran Yapılandırması

Kamera görüntüsünü göstermek için Docker'ın X sunucunuza bağlanmasına izin vermeniz gerekir:

```bash
xhost +local:docker
```

## Sorun Giderme

- Kamera görüntüsünü göremiyorsanız, Docker'ın X sunucunuza bağlanmasına izin verdiğinizden emin olun.
- Kamera algılanmıyorsa, kamera cihaz yolunu ve izinlerini doğrulayın.
- Docker Desktop kullanıcıları için cihaz erişimi için ek yapılandırma gerekebilir.

## Modeller Hakkında

Bu uygulama, indirilen modeli kullanarak tamamen yerel olarak çalışır. Model ilk çalıştırmada indirilir ve sonraki çalıştırmalarda tekrar indirilmesine gerek kalmaz. Bu, uygulamanın hızını artırır ve internet bağlantısına ihtiyaç duymadan çalışmasını sağlar. 
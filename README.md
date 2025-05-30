# Calorie Track API

Sistem API untuk mengenali jenis makanan dan melacak informasi nutrisi makanan Indonesia berdasarkan gambar atau nama makanan.

## Daftar Isi

- [Fitur](#fitur)
- [Teknologi](#teknologi)
- [Cara Penggunaan dengan Docker](#cara-penggunaan-dengan-docker)
- [Endpoint API](#endpoint-api)
- [Struktur Project](#struktur-project)
- [Kontributor](#kontributor)

## Fitur

- Identifikasi makanan Indonesia dari gambar menggunakan machine learning
- Ekstraksi informasi nutrisi untuk berbagai makanan Indonesia
- Support untuk berbagai ukuran porsi makanan
- Pencarian nutrisi berdasarkan nama makanan

## Teknologi

- FastAPI
- TensorFlow
- Docker
- BeautifulSoup (web scraping)
- Python 3.9

## Cara Penggunaan dengan Docker

### Prasyarat

- [Docker](https://www.docker.com/products/docker-desktop) sudah terinstall di komputer Anda
- [Git](https://git-scm.com/downloads) (opsional, jika Anda ingin clone repository)

### Langkah 1: Clone Repository atau Download

```powershell
# Clone repository (jika menggunakan Git)
git clone <url-repository>
cd calorie-track

# Atau download dan ekstrak, kemudian buka folder
cd D:\ML\calorie-track
```

### Langkah 2: Build dan Jalankan dengan Docker

```powershell
# Build image dan jalankan container
docker-compose up --build
```

Perintah ini akan:

- Membuild Docker image berdasarkan Dockerfile
- Menjalankan container dan menampilkan log

Untuk menjalankan aplikasi di background:

```powershell
docker-compose up --build -d
```

### Langkah 3: Akses API

Setelah container berjalan, Anda dapat mengakses:

- API Documentation: http://localhost:8000/docs
- API Endpoint: http://localhost:8000

### Menghentikan Aplikasi

```powershell
# Jika container berjalan di foreground, tekan CTRL+C
# Jika container berjalan di background
docker-compose down
```

## Endpoint API

### 1. Identifikasi Makanan dari Gambar

**Endpoint:** `POST /scan_food`

**Request:**

```json
{
	"image_url": "https://example.com/path/to/food_image.jpg"
}
```

atau

```json
{
	"image_path": "path/to/local/food_image.jpg"
}
```

**Response:**

```json
{
	"food_name": "nasi_goreng",
	"confidence": 0.95,
	"nutrition_info": {
		"Kalori": "260 kcal",
		"Lemak": "14.55 g",
		"Karbohidrat": "10.76 g",
		"Protein": "21.93 g"
	},
	"volume": "100 gram"
}
```

### 2. Informasi Nutrisi dari Nama Makanan

**Endpoint:** `POST /food_nutrition`

**Request:**

```json
{
	"name": "nasi goreng"
}
```

**Response:**

```json
{
	"food_name": "nasi goreng",
	"nutrition_info": {
		"Kalori": "260 kcal",
		"Lemak": "14.55 g",
		"Karbohidrat": "10.76 g",
		"Protein": "21.93 g"
	},
	"volume": "100 gram"
}
```

### 3. Informasi Porsi Makanan

**Endpoint:** `POST /food_portions`

**Request:**

```json
{
	"name": "nasi goreng"
}
```

**Response:**

```json
{
	"portions": [
		{
			"porsi": "100 gram",
			"Kalori": "260 kcal",
			"Lemak": "14.55 g",
			"Karbohidrat": "10.76 g",
			"Protein": "21.93 g",
			"source": "fatsecret"
		},
		{
			"porsi": "1 porsi",
			"Kalori": "520 kcal",
			"Lemak": "29.1 g",
			"Karbohidrat": "21.52 g",
			"Protein": "43.86 g",
			"source": "fatsecret"
		}
	],
	"success": true,
	"message": "Found 2 portion types",
	"food_name": "nasi goreng"
}
```

## Struktur Project

```
calorie-track/
├── main.py            # File utama aplikasi FastAPI
├── wsgi.py            # Entry point untuk server
├── requirements.txt   # Dependensi Python
├── dockerfile         # Konfigurasi Docker
├── docker-compose.yml # Konfigurasi Docker Compose
├── helper/            # Package helper
│   ├── __init__.py    # Inisialisasi package
│   ├── food.py        # Daftar makanan yang didukung
│   ├── functions.py   # Fungsi utilitas
│   └── scrap.py       # Fungsi web scraping
└── model/             # Model machine learning
    ├── saved_model.pb # Model TensorFlow
    ├── variables/     # Variabel model
    └── assets/        # Aset model
```

## Pemecahan Masalah

### Container Tidak Berjalan

Periksa status container:

```powershell
docker ps -a
```

Lihat log container:

```powershell
docker logs calorie-track-calorie-app-1
```

### Error 'ModuleNotFoundError'

Pastikan semua dependensi sudah terinstall dengan benar. Coba rebuild container:

```powershell
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

Dibuat dengan ❤️ menggunakan Python dan TensorFlow

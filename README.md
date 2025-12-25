# Klasifikasi Jenis Spesifik Sampah Menggunakan Deep Learning


Repository ini dibuat untuk memenuhi **Tugas Akhir Praktikum (UAP)** Mata Kuliah **Pembelajaran Mesin** pada Program Studi Informatika, Universitas Muhammadiyah Malang. Proyek ini mengimplementasikan sistem klasifikasi gambar sampah menggunakan metode Deep Learning dengan pendekatan Neural Network dan Transfer Learning.

---

## Daftar Isi
- [Deskripsi Proyek](#-deskripsi-proyek)
- [Dataset](#-dataset)
- [Preprocessing Data](#-preprocessing-data)
- [Model yang Digunakan](#-model-yang-digunakan)
- [Hasil Evaluasi dan Analisis](#-hasil-evaluasi-dan-analisis)
- [Cara Menjalankan Program](#-cara-menjalankan-program)
- [Biodata](#-biodata)

---

## Deskripsi Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi otomatis yang mampu mengenali dan mengkategorikan berbagai jenis sampah berdasarkan gambar/foto. Sistem ini dibangun menggunakan teknik Deep Learning dengan implementasi tiga model berbeda:

1. **Convolutional Neural Network (CNN)** - Model dasar yang dibangun from scratch
2. **ResNet50** - Model pretrained dengan arsitektur residual learning
3. **MobileNetV2** - Model pretrained yang dioptimasi untuk efisiensi

### Tujuan Proyek:
- Memahami dan mengimplementasikan Neural Network dasar untuk klasifikasi gambar
- Menerapkan konsep Transfer Learning menggunakan model pretrained
- Membandingkan performa antara model non-pretrained dan pretrained
- Membangun sistem website sederhana menggunakan Streamlit untuk deployment model

### Manfaat:
Sistem ini dapat membantu dalam proses pengelolaan sampah yang lebih efisien dengan mengotomatisasi proses klasifikasi, sehingga memudahkan pemilahan sampah untuk daur ulang dan pengelolaan limbah yang lebih baik.

---

## Dataset

### Sumber Dataset
Dataset yang digunakan adalah **Garbage Classification (12 classes)** dari Kaggle:
- **Link Dataset**: [Kaggle - Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Jumlah Total Data**: 15.150 gambar
- **Jumlah Kelas**: 12 kategori sampah

### Kategori Sampah (12 Kelas)

| No | Kategori      | Deskripsi                    |
|:--:|:--------------|:-----------------------------|
| 1  | Paper         | Kertas dan produk kertas     |
| 2  | Cardboard     | Karton dan kemasan kardus    |
| 3  | Plastic       | Berbagai jenis plastik       |
| 4  | Metal         | Kaleng dan produk logam      |
| 5  | Glass         | Kaca berbagai warna          |
| 6  | Trash         | Sampah umum/campuran         |
| 7  | Battery       | Baterai bekas                |
| 8  | Shoes         | Sepatu bekas                 |
| 9  | Clothes       | Pakaian bekas                |
| 10 | Biological    | Sampah organik               |
| 11 | Brown-glass   | Kaca cokelat                 |
| 12 | White-glass   | Kaca putih                   |

### Pembagian Dataset

| Subset       | Jumlah Data | Persentase |
|:-------------|:-----------:|:----------:|
| Training     |   10.605   |    70%     |
| Validation   |   2.272    |    15%     |
| Testing      |   2.273    |    15%     |
| **Total**    | **15.150**  |  **100%**  |

---

## Preprocessing Data

### Konfigurasi Training

Untuk mengoptimalkan proses training, digunakan konfigurasi berikut:
- **Image Size**: 160 x 160 pixels (optimasi dari 224x224 default)
- **Batch Size**: 64
- **Epochs**: 15 (dengan Early Stopping patience=3)
- **Training Time**: ±30-60 menit untuk masing masing model

### Langkah-langkah Preprocessing:

#### 1. Split Dataset
Dataset dibagi secara random dengan seed 42 untuk reproducibility:
- **Training Set**: 70% untuk melatih model
- **Validation Set**: 15% untuk monitoring dan early stopping
- **Test Set**: 15% untuk evaluasi final model

#### 2. Image Augmentation (Training Set)
Augmentasi data diterapkan pada training set untuk meningkatkan generalisasi model:
- **Rescaling**: Normalisasi pixel values ke range [0, 1]
- **Rotation**: Rotasi gambar hingga 20 derajat
- **Zoom**: Zoom in/out hingga 20%
- **Horizontal Flip**: Flipping gambar secara horizontal

#### 3. Image Preprocessing (Validation & Test Set)
- **Rescaling Only**: Normalisasi pixel values ke range [0, 1]
- Tidak ada augmentasi untuk memastikan evaluasi objektif

#### 4. Resize dan Batch Processing
- **Target Size**: 160 x 160 pixels (optimized untuk kecepatan training)
- **Batch Size**: 64 gambar per batch
- **Color Mode**: RGB (3 channels)

---

## Model yang Digunakan

### 1. CNN Non-Pretrained (Baseline Model)

Model Convolutional Neural Network yang dibangun from scratch tanpa menggunakan bobot pretrained.

#### Arsitektur Model:
```
Input (160x160x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (12 units) + Softmax
```

#### Karakteristik:
- **Training Time**: ±20 menit
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: EarlyStopping (patience=3)

---

### 2. ResNet50 (Pretrained Model)

Model pretrained berbasis arsitektur Residual Network dengan 50 layer, menggunakan bobot ImageNet.

#### Pendekatan Transfer Learning:
```
ResNet50 Base (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (12 units) + Softmax
```

#### Karakteristik:
- **Pretrained Weights**: ImageNet (1000 classes)
- **Training Strategy**: Hanya melatih layer classifier baru
- **Training Time**: ±60 menit
- **Optimizer**: Adam (learning_rate=0.0001)
- **Keunggulan**: 
  - Transfer knowledge dari ImageNet
  - Training lebih cepat
  - Mengatasi masalah overfitting

---

### 3. MobileNetV2 (Pretrained Model)

Model pretrained yang dioptimasi untuk efisiensi komputasi, cocok untuk deployment pada perangkat dengan resource terbatas.

#### Pendekatan Transfer Learning:
```
MobileNetV2 Base (Frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense (128 units) + ReLU
    ↓
Dropout (0.5)
    ↓
Dense (12 units) + Softmax
```

#### Karakteristik:
- **Pretrained Weights**: ImageNet (1000 classes)
- **Training Time**: ±30 menit
- **Optimizer**: Adam (learning_rate=0.0001)
- **Keunggulan**:
  - Model lightweight (ukuran kecil)
  - Inference time cepat
  - Cocok untuk mobile deployment
  - Balance antara akurasi dan efisiensi

---

## Hasil Evaluasi dan Analisis

### Tabel Perbandingan Performa Model

| Nama Model         | Akurasi (%) | Precision (%) | Recall (%) | F1-Score (%) | 
|:-------------------|:-----------:|:-------------:|:----------:|:------------:| 
| CNN Non-Pretrained |   **76.32** |     75.97     |   76.32    |    75.75     | 
| ResNet50           |   **39.31** |     22.24     |   39.31    |    26.40     |  
| MobileNetV2        |   **90.72** |     90.88     |   90.72    |    90.68     | 

---

### Tabel Analisis Perbandingan Model

| Nama Model         | Akurasi | Hasil Analisis |
|--------------------|---------|----------------|
| CNN Non-Pretrained | 76.32%  | Model baseline yang dibangun dari nol mampu mempelajari pola dasar klasifikasi sampah dengan cukup baik. Performa stabil dan training relatif cepat, namun akurasinya masih kalah dibanding model pretrained karena keterbatasan generalisasi. |
| ResNet50           | 39.31%  | Model menunjukkan performa sangat rendah dan mengalami underfitting. Arsitektur terlalu kompleks untuk input 160×160, seluruh layer dibekukan, serta learning rate yang kurang optimal menyebabkan model gagal beradaptasi dengan dataset. |
| MobileNetV2        | 90.72%  | Model terbaik dengan akurasi tertinggi dan performa paling seimbang. Arsitektur ringan dan efisien sangat cocok untuk image size 160×160, menghasilkan klasifikasi yang akurat, stabil, dan optimal untuk deployment aplikasi. |

---

### Kesimpulan Analisis

#### Perbandingan Kinerja:

**1. Model Terbaik:**
- **MobileNetV2** (90.72%) - Unggul dalam segala aspek

**2. Model Baseline:**
- **CNN Non-Pretrained** (76.32%) - Solid baseline, cukup baik untuk model sederhana

**3. Model Gagal:**
- **ResNet50** (39.31%) - Performa buruk, tidak cocok dengan konfigurasi saat ini


---

## Cara Menjalankan Program

### Prerequisites

Pastikan Anda telah menginstall:
- Python 3.10 atau lebih tinggi
- PDM (Python Development Manager)
- Git

### 1. Download Dataset

Download dataset dari Kaggle:
```bash
# Gunakan Kaggle API atau download manual dari:
# https://www.kaggle.com/datasets/mostafaabla/garbage-classification

# Ekstrak file zip
unzip garbage_classification.zip
```

### 2. Training Model

Jalankan notebook untuk melatih model:

```bash
# Buka Jupyter Notebook atau Google Colab
jupyter notebook UAP_ML.ipynb
```

### 3. Menjalankan Website Streamlit

Setelah model selesai dilatih, jalankan aplikasi web:

```bash
# Jalankan aplikasi Streamlit
streamlit run app.py

# Aplikasi akan berjalan di:
# Local URL: http://localhost:8501
```

### 4. Menggunakan Aplikasi

1. Buka browser dan akses `http://localhost:8501`
2. Pilih model yang ingin digunakan (CNN, ResNet50, atau MobileNetV2)
3. Lihat metrik evaluasi model yang dipilih
4. Upload gambar sampah (format: JPG, JPEG, PNG)
5. Klik tombol "Prediksi Jenis Sampah"
6. Lihat hasil klasifikasi beserta confidence score

**Rekomendasi:** Gunakan model **MobileNetV2** untuk hasil terbaik (akurasi 90.98%)

---

## Biodata

**[Ayulexa Rifka Raihaningtyas]**
- **NIM**: [202210370311418]
- **Program Studi**: Informatika
- **Universitas**: Universitas Muhammadiyah Malang
- **GitHub**: [@ayulexa](https://github.com/ayulexa)

---


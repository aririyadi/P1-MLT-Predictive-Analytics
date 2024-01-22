# Laporan Proyek Machine Learning – ARI RIYADI
## Domain Proyek
### Latar Belakang
Dalam era globalisasi dan pasar keuangan yang dinamis, pengambilan keputusan yang tepat dalam investasi menjadi semakin penting. Salah satu alat yang telah menjadi fokus perhatian adalah analisis prediktif, yang memungkinkan para investor untuk membuat keputusan yang lebih informasional dan cerdas. Dalam konteks ini, proyek ini akan berfokus pada penerapan analisis prediktif terhadap data historis harga dan data terkait saham META Platforms, Inc.

Proyek ini bertujuan untuk menggali potensi analisis prediktif terhadap harga saham META Platforms, Inc., memberikan nilai tambah dalam pengambilan keputusan investasi, dan meningkatkan pemahaman tentang bagaimana data historis dapat digunakan untuk meramalkan perilaku pasar keuangan.

### Referensi
[1]. Brown, S. J., Rozeff, M. S., & Ball, R. (1976). The Influence of Institutional Investors on Myopic R&D Investment Behavior. Journal of Finance, 31(5), 1653-1665.

[2]. Chen, J., Fan, Y., & Li, Q. (2014). Online Daily Stock Trading with Regularized Linear Models. Journal of Business & Economic Statistics, 32(2), 267-287.

[3]. Siegel, E. (2013). Predictive Analytics: The Power to Predict Who Will Click, Buy, Lie, or Die. Wiley.

[4]. Heaton, J., Polson, N. G., & Witte, J. H. (2017). Deep Learning for Finance: Deep Portfolios. Applied Stochastic Models in Business and Industry, 33(1), 3-12.

## Business Understanding
### Problem Statements
- Bagaimana menganalisis secara efektif data historis harga saham META Platforms, Inc. Agar mendapatkan wawasan yang mendalam terkait dengan pergerakan dan faktor-faktor yang mempengaruhi harga saham tersebut?
- Bagaimana memproses data historis harga saham META Platforms, Inc. agar dapat digunakan secara optimal oleh model analisis prediktif?
- Bagaimana membangun model analisis prediktif yang dapat memprediksi pergerakan harga saham META Platforms, Inc. dengan tingkat akurasi yang tinggi?

### Goals
- Mengembangkan analisis data historis harga saham META Platforms, Inc. yang memberikan wawasan mendalam terkait dengan pola dan tren pergerakan harga saham tersebut.
- Memproses data historis dengan baik, termasuk menangani outlier, mengelola missing values, dan melakukan feature engineering yang relevan agar dapat menjadi input yang optimal untuk model prediktif.
- Membangun model analisis prediktif yang memiliki tingkat akurasi yang tinggi dalam meramalkan pergerakan harga saham META Platforms, Inc. Metrik evaluasi dapat mencakup Mean Squared Error (MSE) atau akurasi prediksi yang dapat diukur.

### Solution statements
#### 1. Menerapkan Teknik Analisis Prediktif dengan Machine Learning Algorithms
- Menggunakan algoritma seperti Support Vector Machine (untuk Support Vector Regression), K-Nearest Neighbors, dan Boosting Algorithm (untuk Gradient Boosting Regression) untuk melakukan analisis prediktif pada data historis harga saham META Platforms, Inc.
#### 2. Optimasi Data Preprocessing untuk Kualitas Dataset yang Lebih Baik
- Melakukan preprocessing data dengan cermat, termasuk identifikasi dan penanganan outlier secara efisien untuk mencegah pengaruh yang tidak diinginkan pada model.
- Menyusun strategi yang efektif dalam mengelola missing values, baik dengan metode imputasi atau penghapusan data yang tidak lengkap.
- Mengimplementasikan teknik feature engineering dengan menguji dan memilih variabel yang paling relevan untuk meningkatkan kualitas input data bagi model prediktif.
#### 3. Penyetelan Hyperparameter Menggunakan Teknik Grid Search
- Melakukan penyetelan hyperparameter agar model dapat beroperasi dengan performa terbaik, menggunakan teknik Grid Search.

## Data Understanding
Dalam proyek ini, data yang digunakan berasal dari dataset historis harga saham META Platforms, Inc.
Dataset ini dapat diunduh di Kaggle : META Stock Historical Prices & Data.

### Berikut informasi pada dataset:
- Dataset memiliki format CSV (Comma-Seperated Values).
- Dataset memiliki 1509 Data & 7 Kolom (Date, Open, High, Low, Close, Adj Close, Volume).
- Dataset memiliki 1 fitur bertipe object, 5 fitur bertipe float64 dan 1 fitur bertipe int64.
- Tidak ada missing value dalam dataset.

### Variabel-variabel pada Dataset: 
- Date : Tanggal transaksi saham. 
- Open : harga saham pada saat pembukaan pasar pada tanggal tertentu.
- High : harga saham tertinggi yang dicapai pada tanggal tertentu.
- Low : harga saham terendah yang dicapai pada tanggal tertentu.
- Close : harga saham pada saat penutupan pasar pada tanggal tertentu.
- Adj Close (Adjusted Close) : harga penutup yang telah disesuaikan dengan pembagian saham, dividen, atau perubahan struktur modal lainnya yang dapat mempengaruhi harga saham.
- Volume : jumlah saham yang diperdagangkan pada tanggal tertentu.

### Exploratory Data Analysis - Outliers
Berikut visualisasi data META dengan boxplot untuk mendeteksi outliers pada beberapa fitur numerik:

![Outliers](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/3219872e29cf2c104976997dc3b8440cbd4ef0f5/Gambar/1.png)

Dari visualisasi data, hanya fitur Volume saja yang memiliki outlier. Untuk menangani outlier kita akan menggunakan IQR Method yaitu dengan menghapus data yang berada diluar IQR yaitu antara 25% dan 75%.














## Data Preparation


## Modeling


## Evaluation















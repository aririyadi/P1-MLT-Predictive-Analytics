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
Dataset ini dapat diunduh di [Kaggle : META Stock Historical Prices & Data](https://www.kaggle.com/datasets/fhabibimoghaddam/meta-stock-historical-prices-and-data2).


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

### Exploratory Data Analysis - Univariate Analysis

![Univariate](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/40b4be608e02ef93142d23f4ba1363f77ca71a8d/Gambar/2.png)

Mari amati histogram di atas, gambar histogram di atas memberikan visualisasi distribusi univariat dari suatu variabel pada dataset. Histogram tersebut menggambarkan distribusi frekuensi dari nilai-nilai pada variabel tertentu.

### Exploratory Data Analysis - Multivariate Analysis

![Multivariate](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/40b4be608e02ef93142d23f4ba1363f77ca71a8d/Gambar/3.png)

Berdasarkan visualisasi di atas, kita memperoleh pemahaman yang lebih mendalam tentang interaksi dan ketergantungan antar variabel numerik dalam dataset. Hasil analisis ini dapat menjadi dasar untuk pemilihan fitur dalam pembangunan model prediktif, serta memberikan wawasan yang diperlukan untuk langkah-langkah analisis selanjutnya.

### Correlation Matrix

![Correlation](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/22ef231364015d7195c84ca6230777a11571d4e4/Gambar/4.png)

Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa semua variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya.

## Data Preparation

### Menghapus Fitur Yang Tidak Diperlukan
Penghapusan fitur-fitur (Date, Volume dan Close) bertujuan untuk menyederhanakan dataset dan fokus pada atribut yang dianggap lebih relevan atau akurat dalam konteks analisis atau pembuatan model yang dilakukan. Berikut potongan kode dan hasil outputnya:

```sh
meta = meta.drop(['Date', 'Volume', 'Close'], axis=1)
meta.head()
```

![Data-Preparation](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/87625f805e773ded0654bd22d67dfd0a0b820fcd/Gambar/5.png)


### Melakukan Pembagian Dataset
Setelah menghapus fitur yang tidak diperlukan, langkah selanjutnya adalah membagi dataset menjadi dua bagian: data pelatihan (80%) untuk melatih model dan data pengujian (20%) untuk menguji kinerja model pada data baru. Proporsi 80% untuk data pelatihan dipilih untuk memastikan model mendapatkan sejumlah besar data untuk belajar, sedangkan 20% sisanya digunakan untuk pengujian, memungkinkan evaluasi objektif terhadap kemampuan prediktif model. Berikut potongan kode dan hasil outputnya:

```sh
X = meta.iloc[:, :-1].values
y = meta.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Total Dataset: {len(X)}')
print(f'Total Train Dataset: {len(X_train)}')
print(f'Total Test Dataset: {len(X_test)}')
```
**Output:**
- Total Dataset: 1509
- Total Train Dataset: 1207
- Total Test Dataset: 302

### Data Normalization
Setelah membagi dataset, langkah selanjutnya adalah melakukan normalisasi data menggunakan Min-Max Scaling. Normalisasi ini dilakukan untuk memastikan bahwa nilai-nilai dalam fitur memiliki rentang yang seragam dan dapat membantu model mengatasi skala yang berbeda di antara fitur-fitur tersebut. Normalisasi data membantu memastikan bahwa model tidak terlalu dipengaruhi oleh skala absolut dari nilai-nilai dalam fitur-fitur, dan ini dapat meningkatkan performa model, terutama untuk algoritma yang sensitif terhadap skala. Berikut potongan kode untuk Data Normalization:

```sh
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Modeling
Dalam pemodelan ini, tiga algoritma machine learning yang digunakan untuk menyelesaikan permasalahan, yaitu Support Vector Regression (SVR), Gradient Boosting, dan K-Nearest Neighbors (KNN).

### Support Vector Regression (SVR)
#### Kode:
```sh
svr = SVR(C=10, gamma=0.3, kernel='rbf')
svr.fit(X_train, y_train)
```
#### Parameter yang Digunakan:
-	C=10 : Parameter penalti yang mengontrol toleransi terhadap kesalahan.
-	gamma=0.3 : Parameter kernel untuk 'rbf' yang mengontrol bentuk dari fungsi basis Gaussian.
-	kernel='rbf ': Jenis kernel yang digunakan.
#### Kelebihan dan Kekurangan:
-	Kelebihan: Mampu menangani data non-linier dan bekerja baik dengan data berdimensi tinggi.
-	Kekurangan: Sensitif terhadap pemilihan parameter dan dapat memerlukan waktu komputasi yang cukup besar.

### Gradient Boosting
#### Kode:
```sh
gradient_boost = GradientBoostingRegressor(
criterion='squared_error',
learning_rate=0.01,
n_estimators=1000
)
gradient_boost.fit(X_train, y_train)
```
#### Parameter yang Digunakan:
-	learning_rate=0.01: Tingkat pembelajaran yang mengontrol seberapa besar model beradaptasi terhadap kesalahan sebelumnya.
-	n_estimators=1000: Jumlah pohon keputusan yang dibangun.
-	criterion='squared_error': Kriteria untuk mengukur kualitas split.
#### Kelebihan dan Kekurangan:
o	Kelebihan: Tingkat akurasi tinggi, dapat menangani data berdimensi tinggi dan fitur yang tidak terstruktur.
o	Kekurangan: Rentan terhadap overfitting, dan hyperparameter tuning dapat memakan waktu.

### K-Nearest Neighbors (KNN)
#### Kode:
```sh
knn = KNeighborsRegressor(n_neighbors=9)
knn.fit(X_train, y_train)
```
#### Parameter yang Digunakan:
-	n_neighbors=9: Jumlah tetangga yang digunakan untuk memprediksi nilai.
#### Kelebihan dan Kekurangan:
-	Kelebihan: Sederhana, mudah diimplementasikan, bekerja baik untuk dataset kecil.
-	Kekurangan: Rentan terhadap noise dan outlier, performa dapat dipengaruhi oleh pemilihan jumlah tetangga.


## Evaluation
Dalam proyek ini, metrik evaluasi yang digunakan untuk mengukur performa model adalah Mean Squared Error (MSE). MSE digunakan karena tugas ini merupakan masalah regresi, di mana kita berfokus pada prediksi nilai numerik. MSE mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya.

### Formula Mean Squared Error (MSE):
$MSE = \frac{1}{N} \Sigma_{i=1}^N({y_i}- y\_pred_i)^2$

MSE mengukur seberapa dekat prediksi model dengan nilai sebenarnya. Semakin rendah MSE, semakin baik model dalam memprediksi nilai target.

### Hasil Proyek Berdasarkan Metrik Evaluasi:
-	Model dievaluasi menggunakan MSE pada data pengujian.
-	Semakin kecil nilai MSE, semakin baik kinerja model dalam menghasilkan prediksi yang akurat.
-	Model dengan MSE terendah dianggap sebagai solusi terbaik untuk menyelesaikan permasalahan regresi ini.

Dengan menggunakan MSE sebagai metrik evaluasi, kita dapat mengukur seberapa baik model dapat memprediksi nilai harga saham atau variabel numerik lainnya dalam dataset. Semakin kecil nilai MSE, semakin akurat prediksi model, dan sebaliknya. Berikut potongan kode dan hasil outputnya:

```sh
model_dict = {
    'SVR': svr,
    'GradientBoosting': gradient_boost,
    'KNN': knn,
}

for name, model in model_dict.items():
  models.loc[name, 'train_mse'] = mean_squared_error(y_train, model.predict(X_train))
  models.loc[name, 'test_mse'] = mean_squared_error(y_test, model.predict(X_test))

models.head()
```

![Evaluation](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/38efe068cfb92cdf9ac63cabc5dfed135e58e8a2/Gambar/6.png)















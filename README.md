# Laporan Proyek Machine Learning – _Ari Riyadi_
## Domain Proyek
Dataset META Platforms, Inc. dipilih untuk proyek ini karena keberagaman data yang mencakup teks, gambar, dan metadata, serta ketersediaan data yang luas. Relevansi platform ini dengan tujuan analisis prediktif terletak pada jumlah pengguna yang besar dan potensi untuk mendukung proyek seperti analisis sentimen, prediksi perilaku pengguna, dan pengembangan model kecerdasan buatan. Karakteristik dataset yang mendukung analisis prediktif melibatkan ukuran sampel besar, ketersediaan label, dan variabel yang relevan, serta penelitian terhadap periode waktu yang memadai. Dengan demikian, pemilihan dataset ini didasarkan pada potensi untuk memberikan wawasan mendalam dan mendukung proyek analisis prediktif yang diinginkan.

### Latar Belakang
Dalam era globalisasi dan pasar keuangan yang dinamis, pengambilan keputusan yang tepat dalam investasi menjadi semakin penting. Salah satu alat yang telah menjadi fokus perhatian adalah analisis prediktif, yang memungkinkan para investor untuk membuat keputusan yang lebih informasional dan cerdas. Dalam konteks ini, proyek ini akan berfokus pada penerapan analisis prediktif terhadap data historis harga dan data terkait saham META Platforms, Inc.

Proyek ini bertujuan untuk menggali potensi analisis prediktif terhadap harga saham META Platforms, memberikan nilai tambah dalam pengambilan keputusan investasi, dan meningkatkan pemahaman tentang bagaimana data historis dapat digunakan untuk meramalkan perilaku pasar keuangan.

## Business Understanding
### Problem Statements
- Bagaimana menganalisis secara efektif data historis harga saham META Platforms, Inc. Agar mendapatkan wawasan yang mendalam terkait dengan pergerakan dan faktor-faktor yang mempengaruhi harga saham tersebut?
- Bagaimana memproses data historis harga saham META Platforms, Inc. agar dapat digunakan secara optimal oleh model analisis prediktif?
- Bagaimana membangun model analisis prediktif yang dapat memprediksi pergerakan harga saham META Platforms, Inc. dengan tingkat akurasi yang tinggi?

### Goals
- Mengembangkan analisis data historis harga saham META Platforms, Inc. yang memberikan wawasan mendalam terkait dengan pola dan tren pergerakan harga saham tersebut.
- Memproses data historis dengan baik, termasuk menangani _outlier_, mengelola _missing values_, dan melakukan feature engineering yang relevan agar dapat menjadi input yang optimal untuk model prediktif.
- Membangun model analisis prediktif yang memiliki tingkat akurasi yang tinggi dalam meramalkan pergerakan harga saham META Platforms, Inc. Metrik evaluasi dapat mencakup _Mean Squared Error (MSE)_ atau akurasi prediksi yang dapat diukur.

### Solution statements
#### 1. Menerapkan Teknik Analisis Prediktif dengan Machine Learning Algorithms
- Menggunakan algoritma seperti _Support Vector Machine (untuk Support Vector Regression)_, _K-Nearest Neighbors_, dan _Boosting Algorithm (untuk Gradient Boosting Regression)_ untuk melakukan analisis prediktif pada data historis harga saham META Platforms, Inc.
#### 2. Optimasi Data Preprocessing untuk Kualitas Dataset yang Lebih Baik
- Melakukan preprocessing data dengan cermat, termasuk identifikasi dan penanganan outlier secara efisien untuk mencegah pengaruh yang tidak diinginkan pada model.
- Menyusun strategi yang efektif dalam mengelola missing values, baik dengan metode imputasi atau penghapusan data yang tidak lengkap.
- Mengimplementasikan teknik feature engineering dengan menguji dan memilih variabel yang paling relevan untuk meningkatkan kualitas input data bagi model prediktif.
#### 3. Penyetelan Hyperparameter Menggunakan Teknik _Grid Search_
- Melakukan penyetelan hyperparameter agar model dapat beroperasi dengan performa terbaik, menggunakan teknik _Grid Search_.

Kaitan antara ketiga elemen tersebut terletak pada fakta bahwa Problem Statements mengidentifikasi tantangan yang perlu diatasi, Goals menetapkan tujuan yang harus dicapai, dan Solution Statements memberikan solusi konkret untuk mencapai tujuan tersebut. Seluruh rangkaian ini membentuk kerangka kerja yang koheren untuk proyek analisis prediktif pada data historis harga saham META Platforms, Inc.

## Data Understanding
Dalam proyek ini, data yang digunakan berasal dari dataset historis harga saham META Platforms, Inc.
Dataset ini dapat diunduh di [Kaggle : META Stock Historical Prices & Data](https://www.kaggle.com/datasets/fhabibimoghaddam/meta-stock-historical-prices-and-data2).


### Berikut informasi pada dataset:
- Dataset memiliki format CSV (_Comma-Seperated Values_).
- Dataset memiliki 1509 rows & 7 columns seperti (_Date, Open, High, Low, Close, Adj Close, Volume_).
- Terdapat 1 kolom dengan tipe data _object_, 5 kolom numerik dengan tipe data _float64_ dan 1 kolom numerik dengan tipe data _int64_. Berikut potongan kode dan outputnya:
```sh
meta = pd.read_csv('META.csv')
meta.info()
```
![1](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/747fc8ae-9c1e-498d-aa36-0aac1b386a22)

**Gambar 1**. Informasi Mengenai Dataset

- Tidak ada missing value dalam dataset. Berikut potongan kode untuk mengidentifikasinya:
```sh
meta.isnull().sum()
```
![2](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/2218b7ac-a842-478e-afdf-8522ff40baa5)

**Gambar 2**. Indentifikasi _Missing Value_

- Deskripsi statistik data dengan fitur ```describe()```.
```sh
meta.describe()
```
![3](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/bc96475d-4429-4478-82a7-6671d10a5e25)

**Gambar 3**. Informasi statistik pada masing-masing kolom

Fungsi ```describe()``` memberikan informasi statistik pada masing-masing kolom, antara lain:

- **Count**  adalah jumlah sampel pada data.
- **Mean** adalah nilai rata-rata.
- **Std** adalah standar deviasi.
- **Min** yaitu nilai minimum setiap kolom. 
- **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- **75%** adalah kuartil ketiga.
- **Max** adalah nilai maksimum.

### Variabel-variabel pada Dataset: 
- _Date_ : Tanggal transaksi saham. 
- _Open_ : harga saham pada saat pembukaan pasar pada tanggal tertentu.
- _High_ : harga saham tertinggi yang dicapai pada tanggal tertentu.
- _Low_ : harga saham terendah yang dicapai pada tanggal tertentu.
- _Close_ : harga saham pada saat penutupan pasar pada tanggal tertentu.
- _Adj Close_ : harga penutup yang telah disesuaikan dengan pembagian saham, dividen, atau perubahan struktur modal lainnya yang dapat mempengaruhi harga saham.
- _Volume_ : jumlah saham yang diperdagangkan pada tanggal tertentu.

### Exploratory Data Analysis - Tren Waktu Saham META Platforms
Berikut adalah kode dan visualisasi yang menggambarkan tren waktu terhadap harga saham _META Platforms_:

```sh
meta['Date'] = pd.to_datetime(meta['Date'])
plt.figure(figsize=(12, 6))
plt.plot(meta['Date'], meta['Adj Close'], label='Adj Close', color='blue')
plt.title('Tren Waktu Harga Saham META')
plt.xlabel('Tahun')
plt.ylabel('Harga Penutupan')
plt.legend()
plt.show()
```
![4](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/ea6b1283-1087-454d-907d-fb0e8f142be3)

**Gambar 4**. Grafik Tren Waktu saham META Platforms, Inc

Grafik Tren Waktu ini dapat menjadi dasar untuk analisis lebih lanjut terhadap performa saham META Platforms, Inc. dan membantu dalam pengambilan keputusan investasi atau strategi perdagangan. Analisis lebih lanjut, baik dalam bentuk statistik atau model prediktif, mungkin diperlukan untuk memperdalam pemahaman tentang pergerakan harga saham ini.

### Exploratory Data Analysis - _Outliers_
Berikut potongan kode dan visualisasi data META dengan boxplot untuk mendeteksi _outliers_ pada beberapa fitur numerik:

```sh
plt.subplots(figsize=(10,7))
sns.boxplot(data=meta).set_title("META Platforms, Inc")
plt.show()
```
![5](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/604f16cb-ad9b-47c6-9fd6-8081abbff042)


**Gambar 5**. Visualisasi Mendeteksi _Outlier_

Dari visualisasi data, hanya fitur Volume saja yang memiliki _outlier_. Untuk menangani _outlier_ kita akan menggunakan _IQR Method_ yaitu dengan menghapus data yang berada diluar _IQR_. Berikut kode untuk menghapusnya:

```sh
numeric_columns = meta.select_dtypes(include=['float64', 'int64']).columns
Q1 = meta[numeric_columns].quantile(0.25)
Q3 = meta[numeric_columns].quantile(0.75)
IQR = Q3 - Q1
meta = meta[~((meta[numeric_columns] < (Q1 - 1.5 * IQR)) | (meta[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
print(meta.shape)
```
Berikut Penjelasan Kodenya:

- ```numeric_columns = meta.select_dtypes(include=['float64', 'int64']).columns```: Memilih hanya kolom-kolom numerik (_float64_ dan _int64_) dari dataset. Ini termasuk kolom seperti _'Open', 'High', 'Low', 'Close', 'Adj Close', dan 'Volume'_.
- ```Q1 = meta[numeric_columns].quantile(0.25)```: Menghitung nilai kuartil pertama (25th percentile) untuk setiap kolom numerik.
- ```Q3 = meta[numeric_columns].quantile(0.75)```: Menghitung nilai kuartil ketiga (75th percentile) untuk setiap kolom numerik.
- ```IQR = Q3 - Q1```: Menghitung rentang interkuartil (IQR) untuk setiap kolom numerik.
- ```((meta[numeric_columns] < (Q1 - 1.5 * IQR)) | (meta[numeric_columns] > (Q3 + 1.5 * IQR)))```: Menandai data sebagai _outlier_ jika berada di luar rentang ```(Q1 - 1.5 * IQR)``` hingga ```(Q3 + 1.5 * IQR)```.
- ```meta = meta[~((meta[numeric_columns] < (Q1 - 1.5 * IQR)) | (meta[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]```: Menghapus baris yang mengandung setidaknya satu _outlier_ dalam setiap kolom numerik.
- ```print(meta.shape)```: Menampilkan bentuk (jumlah baris dan kolom) dari dataset setelah menghapus _outlier_.

### _Exploratory Data Analysis - Univariate Analysis_
```sh
meta.hist(bins=50, figsize=(20,15))
plt.show()
```

![6](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/f96839e4-d4bf-4fb8-b448-d865e5fb2ee5)

**Gambar 6**. Histogram Fitur Numerik - _Univariate Analysis_

Mari amati histogram di atas, khususnya histogram untuk variabel "Adj Close" yang merupakan fitur target (label) pada data kita. Dari histogram "Adj Close", kita bisa memperoleh beberapa informasi, antara lain:

- Rentang harga saham META cukup tinggi yaitu dari skala ratusan dolar Amerika hingga sekitar $273000.
- Setengah harga saham META bernilai di bawah $203000.

### _Exploratory Data Analysis - Multivariate Analysis_
```sh
sns.pairplot(meta, diag_kind = 'kde')
```
![7](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/f2ece70e-cc59-4f35-be8a-23b3619af727)

**Gambar 7**. Visualisasi Hubungan Antar Fitur Numerik - _Multivariate Analysis_

Berdasarkan visualisasi di atas, kita memperoleh pemahaman yang lebih mendalam tentang interaksi dan ketergantungan antar variabel numerik dalam dataset. Hasil analisis ini dapat menjadi dasar untuk pemilihan fitur dalam pembangunan model prediktif, serta memberikan wawasan yang diperlukan untuk langkah-langkah analisis selanjutnya.

### _Correlation Matrix_
```sh
plt.figure(figsize=(10, 8))
correlation_matrix = meta.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=10)
```

![8](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/c9e460d2-2c4d-463c-afa1-5065b48c19b0)

**Gambar 8**. Visualisasi Matriks Korelasi

Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa semua variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya.

## Data Preparation

### Menghapus Fitur Yang Tidak Diperlukan
Penghapusan fitur-fitur (_Date, Volume dan Close_) bertujuan untuk menyederhanakan dataset dan fokus pada atribut yang dianggap lebih relevan atau akurat dalam konteks analisis atau pembuatan model yang dilakukan. Berikut potongan kode dan hasil outputnya dalam bentuk tabel:

```sh
meta = meta.drop(['Date', 'Volume', 'Close'], axis=1)
meta.head()
```
**Tabel 1**. Data Harga Saham META setelah Penghapusan Fitur
|   Open      |    High     |    Low      |  Adj Close  |
|-------------|-------------|-------------|-------------|
| 181.880005  | 184.779999  | 181.330002  | 184.669998  |
| 184.899994  | 186.210007  | 184.100006  | 184.330002  |
| 185.589996  | 186.899994  | 184.929993  | 186.850006  |
| 187.199997  | 188.899994  | 186.330002  | 188.279999  |
| 188.699997  | 188.800003  | 187.100006  | 187.869995  |

Pada Tabel 1, dapat dilihat bahwa fitur-fitur Date, Volume, dan Close telah dihapus dari dataset, meninggalkan hanya fitur-fitur Open, High, Low, dan Adj Close. Pembersihan ini bertujuan untuk menyederhanakan dataset dan memfokuskan perhatian pada atribut yang dianggap lebih relevan.

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
- Total Dataset: 1422
- Total Train Dataset: 1137
- Total Test Dataset: 285

### Data Normalization
Min-Max Scaling (MinMaxScaler) adalah salah satu teknik normalisasi yang digunakan untuk mengubah nilai-nilai dalam dataset ke dalam rentang tertentu, umumnya antara 0 dan 1. Pemilihan parameter pada Min-Max Scaling dapat mempengaruhi performa model dan penyesuaian normalisasi terhadap data.

**Kode :**

```sh
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
Dalam konteks proyek analisis prediktif pada data historis harga saham META Platforms, penggunaan Min-Max Scaling dengan penyesuaian parameter '_feature_range_' memiliki dampak penting pada normalisasi data. Dalam dataset harga saham, variabel seperti harga pembukaan (_Open_), harga tertinggi (_High_), harga terendah (_Low_), dan harga penutupan yang disesuaikan (_Adj Close_) mungkin memiliki rentang nilai yang cukup besar. Dengan menggunakan _Min-Max Scaling_, kita dapat mengonversi nilai-nilai ini ke dalam rentang tertentu, memastikan bahwa model dapat memahami dan memproses perbedaan skala antar fitur.

Misalnya, pada proyek ini, harga saham yang bervariasi dari ratusan dolar hingga puluhan ribu dolar. Dengan mengatur '_feature_range_' menjadi rentang yang lebih besar seperti (-1, 1), kita dapat mempertahankan perbedaan relatif antara nilai-nilai tersebut. Pemilihan rentang yang sesuai dapat menjadi kunci untuk mencegah kehilangan informasi yang signifikan dan memastikan model lebih responsif terhadap variasi nilai.

Dengan demikian, penggunaan _Min-Max Scaling_ dengan penyesuaian parameter '_feature_range_' menjadi strategi normalisasi yang relevan dan efektif dalam konteks proyek analisis harga saham META Platforms.

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

### Visualization of Model Comparison
Berikut potongan kode untuk membuat chart dan visualisasi perbandingan model:

```sh
fig, ax = plt.subplots()
models.sort_values(by='test_mse', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
```

![Plot](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/3e9c692e82b71be8e2b02bdf7438a51aa9dcd709/Gambar/7.png)


### Prediction
Melalui langkah-langkah ini, kita dapat melihat bagaimana setiap model yang telah dilatih merespons terhadap subset data uji yang telah dipilih. Hal ini membantu dalam mengevaluasi kemampuan prediktif model pada situasi yang belum pernah dilihat sebelumnya dan memberikan gambaran tentang sejauh mana model dapat menghasilkan prediksi yang mendekati nilai sebenarnya. Berikut potongan kode dan hasil outputnya:

```sh
num_rows_to_predict = 1
X_test_subset = X_test[:num_rows_to_predict, :]
y_test_subset = y_test[:num_rows_to_predict]

pred_dict = {'y_true': y_test_subset}

for name, model in model_dict.items():
    predictions = model.predict(X_test_subset).round(1)
    pred_dict['prediksi_' + name] = predictions

pd.DataFrame(pred_dict)
```

![Prediction](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/blob/de67fea791d39f125d1ae0a5655a9347b865d35c/Gambar/8.png)

## Referensi
[1]. Brown, S. J., Rozeff, M. S., & Ball, R. (1976). The Influence of Institutional Investors on Myopic R&D Investment Behavior. Journal of Finance, 31(5), 1653-1665.

[2]. Chen, J., Fan, Y., & Li, Q. (2014). Online Daily Stock Trading with Regularized Linear Models. Journal of Business & Economic Statistics, 32(2), 267-287.

[3]. Siegel, E. (2013). Predictive Analytics: The Power to Predict Who Will Click, Buy, Lie, or Die. Wiley.

[4]. Heaton, J., Polson, N. G., & Witte, J. H. (2017). Deep Learning for Finance: Deep Portfolios. Applied Stochastic Models in Business and Industry, 33(1), 3-12.








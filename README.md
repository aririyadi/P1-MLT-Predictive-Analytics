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
- Terdapat 1 kolom dengan tipe data _object_, 5 kolom numerik dengan tipe data _float64_ dan 1 kolom numerik dengan tipe data _int64_.
- Tidak ada missing value dalam dataset.
- Deskripsi statistik data dengan fungsi ```describe()```.

**Tabel 1**. Informasi statistik pada masing-masing kolom
|           | Open      | High      | Low       | Close     | Adj Close | Volume        |
|-----------|-----------|-----------|-----------|-----------|-----------|---------------|
| **count** | 1509.000  | 1509.000  | 1509.000  | 1509.000  | 1509.000  | 1.509000e+03  |
| **mean**  | 225.073   | 228.246   | 222.095   | 225.212   | 225.212   | 2.414811e+07  |
| **std**   | 68.302    | 68.898    | 67.641    | 68.291    | 68.291    | 1.628735e+07  |
| **min**   | 90.080    | 90.460    | 88.090    | 88.910    | 88.910    | 5.467500e+06  |
| **25%**   | 174.500   | 176.900   | 172.040   | 174.600   | 174.600   | 1.539570e+07  |
| **50%**   | 202.180   | 204.910   | 199.670   | 202.260   | 202.260   | 2.008990e+07  |
| **75%**   | 279.190   | 285.240   | 276.310   | 281.000   | 281.000   | 2.788780e+07  |
| **max**   | 381.680   | 384.330   | 378.810   | 382.180   | 382.180   | 2.323166e+08  |

**Tabel 1.** memberikan informasi statistik pada masing-masing kolom, antara lain:

**Keterangan:**

- **Count**  adalah jumlah sampel pada data.
- **Mean** adalah nilai rata-rata.
- **Std** adalah standar deviasi.
- **Min** yaitu nilai minimum setiap kolom. 
- **25%** adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval dalam empat bagian sebaran yang sama. 
- **50%** adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- **75%** adalah kuartil ketiga.
- **Max** adalah nilai maksimum.

### Variabel-variabel pada Dataset: 
- **_Date_** : Tanggal transaksi saham. 
- **_Open_** : harga saham pada saat pembukaan pasar pada tanggal tertentu.
- **_High_** : harga saham tertinggi yang dicapai pada tanggal tertentu.
- **_Low_** : harga saham terendah yang dicapai pada tanggal tertentu.
- **_Close_** : harga saham pada saat penutupan pasar pada tanggal tertentu.
- **_Adj Close_** : harga penutup yang telah disesuaikan dengan pembagian saham, dividen, atau perubahan struktur modal lainnya yang dapat mempengaruhi harga saham.
- **_Volume_** : jumlah saham yang diperdagangkan pada tanggal tertentu.

### Exploratory Data Analysis - Tren Waktu Saham META Platforms
Berikut visualisasi yang menggambarkan tren waktu terhadap harga saham _META Platforms_:

![4](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/ea6b1283-1087-454d-907d-fb0e8f142be3)

**Gambar 1**. Grafik Tren Waktu saham META Platforms, Inc

Grafik Tren Waktu ini dapat menjadi dasar untuk analisis lebih lanjut terhadap performa saham META Platforms, Inc. dan membantu dalam pengambilan keputusan investasi atau strategi perdagangan. Analisis lebih lanjut, baik dalam bentuk statistik atau model prediktif, mungkin diperlukan untuk memperdalam pemahaman tentang pergerakan harga saham ini.

### Exploratory Data Analysis - _Outliers_
Berikut visualisasi data META dengan boxplot untuk mendeteksi _outliers_ pada beberapa fitur numerik:

![5](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/604f16cb-ad9b-47c6-9fd6-8081abbff042)

**Gambar 2**. Visualisasi Mendeteksi _Outlier_

Dari visualisasi data, hanya fitur Volume saja yang memiliki _outlier_. Untuk menangani _outlier_ kita akan menggunakan _IQR Method_ yaitu dengan menghapus data yang berada diluar _IQR_.

### _Exploratory Data Analysis - Univariate Analysis_

![6](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/f96839e4-d4bf-4fb8-b448-d865e5fb2ee5)

**Gambar 3**. Histogram Fitur Numerik - _Univariate Analysis_

Mari amati histogram di atas, khususnya histogram untuk variabel "_Adj Close_" yang merupakan fitur target (label) pada data kita. Dari histogram "_Adj Close_", kita bisa memperoleh beberapa informasi, antara lain:

- Rentang harga saham META cukup tinggi yaitu dari skala ratusan dolar Amerika hingga sekitar $273000.
- Setengah harga saham META bernilai di bawah $203000.

### _Exploratory Data Analysis - Multivariate Analysis_

![7](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/f2ece70e-cc59-4f35-be8a-23b3619af727)

**Gambar 4**. Visualisasi Hubungan Antar Fitur Numerik - _Multivariate Analysis_

Berdasarkan visualisasi di atas, kita memperoleh pemahaman yang lebih mendalam tentang interaksi dan ketergantungan antar variabel numerik dalam dataset. Hasil analisis ini dapat menjadi dasar untuk pemilihan fitur dalam pembangunan model prediktif, serta memberikan wawasan yang diperlukan untuk langkah-langkah analisis selanjutnya.

### _Correlation Matrix_

![8](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/c9e460d2-2c4d-463c-afa1-5065b48c19b0)

**Gambar 5**. Visualisasi Matriks Korelasi

Terlihat pada matriks korelasi di atas dapat disimpulkan bahwa semua variabel memiliki keterikatan dan korelasi yang kuat antar variabel lainnya.

## Data Preparation

### Menghapus Fitur Yang Tidak Diperlukan
Penghapusan fitur-fitur (_Date, Volume dan Close_) bertujuan untuk menyederhanakan dataset dan fokus pada atribut yang dianggap lebih relevan atau akurat dalam konteks analisis atau pembuatan model yang dilakukan. Berikut hasil outputnya dalam bentuk tabel:

**Tabel 2**. Data Harga Saham META setelah Penghapusan Fitur
|   Open      |    High     |    Low      |  Adj Close  |
|-------------|-------------|-------------|-------------|
| 181.880005  | 184.779999  | 181.330002  | 184.669998  |
| 184.899994  | 186.210007  | 184.100006  | 184.330002  |
| 185.589996  | 186.899994  | 184.929993  | 186.850006  |
| 187.199997  | 188.899994  | 186.330002  | 188.279999  |
| 188.699997  | 188.800003  | 187.100006  | 187.869995  |

Pada Tabel 2, dapat dilihat bahwa fitur-fitur _Date, Volume,_ dan _Close_ telah dihapus dari dataset, meninggalkan hanya fitur-fitur _Open, High, Low, dan Adj Close_. Pembersihan ini bertujuan untuk menyederhanakan dataset dan memfokuskan perhatian pada atribut yang dianggap lebih relevan.

### Melakukan Pembagian Dataset
Setelah menghapus fitur yang tidak diperlukan, langkah selanjutnya adalah membagi dataset menjadi dua bagian: data pelatihan (80%) untuk melatih model dan data pengujian (20%) untuk menguji kinerja model pada data baru. Proporsi 80% untuk data pelatihan dipilih untuk memastikan model mendapatkan sejumlah besar data untuk belajar, sedangkan 20% sisanya digunakan untuk pengujian, memungkinkan evaluasi objektif terhadap kemampuan prediktif model.

**Hasil Pembagian Dataset:**
- Total Dataset: 1422
- Total Train Dataset: 1137
- Total Test Dataset: 285

### Data Normalization
_Min-Max Scaling_ (_MinMaxScaler_) adalah salah satu teknik normalisasi yang digunakan untuk mengubah nilai-nilai dalam dataset ke dalam rentang tertentu, umumnya antara 0 dan 1. Pemilihan parameter pada _Min-Max Scaling_ dapat mempengaruhi performa model dan penyesuaian normalisasi terhadap data.

Dalam konteks proyek analisis prediktif pada data historis harga saham META Platforms, penggunaan _Min-Max Scaling_ dengan penyesuaian parameter '_feature_range_' memiliki dampak penting pada normalisasi data. Dalam dataset harga saham, variabel seperti harga pembukaan (_Open_), harga tertinggi (_High_), harga terendah (_Low_), dan harga penutupan yang disesuaikan (_Adj Close_) mungkin memiliki rentang nilai yang cukup besar. Dengan menggunakan _Min-Max Scaling_, kita dapat mengonversi nilai-nilai ini ke dalam rentang tertentu, memastikan bahwa model dapat memahami dan memproses perbedaan skala antar fitur.

Misalnya, pada proyek ini, harga saham yang bervariasi dari ratusan dolar hingga puluhan ribu dolar. Dengan mengatur '_feature_range_' menjadi rentang yang lebih besar seperti (-1, 1), kita dapat mempertahankan perbedaan relatif antara nilai-nilai tersebut. Pemilihan rentang yang sesuai dapat menjadi kunci untuk mencegah kehilangan informasi yang signifikan dan memastikan model lebih responsif terhadap variasi nilai.

Dengan demikian, penggunaan _Min-Max Scaling_ dengan penyesuaian parameter '_feature_range_' menjadi strategi normalisasi yang relevan dan efektif dalam konteks proyek analisis harga saham META Platforms.

## Modeling
Dalam pemodelan ini, tiga algoritma machine learning yang digunakan untuk menyelesaikan permasalahan, yaitu _Support Vector Regression (SVR)_, _Gradient Boosting_, dan _K-Nearest Neighbors (KNN)_.

### _Support Vector Regression (SVR)_

#### Parameter yang Digunakan:
-	**C=10** : Parameter penalti yang mengontrol toleransi terhadap kesalahan.
-	**gamma=0.3** : Parameter kernel untuk '_rbf_' yang mengontrol bentuk dari fungsi basis Gaussian.
-	**kernel='rbf** ': Jenis kernel yang digunakan.

#### Cara Kerja:
- _SVR_ digunakan untuk tugas regresi, memprediksi nilai numerik (harga saham) berdasarkan fitur-fitur tertentu.
- Membentuk hyperplane dengan margin maksimum dari titik data target.
- Miminimalkan deviasi atau kesalahan prediksi dari nilai aktual dengan mempertimbangkan batasan margin.
  
#### Kelebihan dan Kekurangan:
-	**Kelebihan**: Mampu menangani data non-linier dan bekerja baik dengan data berdimensi tinggi.
-	**Kekurangan**: Sensitif terhadap pemilihan parameter dan dapat memerlukan waktu komputasi yang cukup besar.

### Gradient Boosting

#### Parameter yang Digunakan:
-	**learning_rate=0.01**: Tingkat pembelajaran yang mengontrol seberapa besar model beradaptasi terhadap kesalahan sebelumnya.
-	**n_estimators=1000**: Jumlah pohon keputusan yang dibangun.
-	**criterion='squared_error'**: Kriteria untuk mengukur kualitas split.

#### Cara Kerja:

- _Boosting_ membangun serangkaian model secara berurutan, setiap model fokus pada mengoreksi kesalahan model sebelumnya.
- _Gradient Boosting_ bekerja dengan mengoptimalkan fungsi objektif berdasarkan gradien dari kesalahan prediksi.
- Menggabungkan prediksi dari model-model lemah untuk membentuk prediksi yang lebih akurat.

#### Kelebihan dan Kekurangan:
- **Kelebihan**: Tingkat akurasi tinggi, dapat menangani data berdimensi tinggi dan fitur yang tidak terstruktur.
-	**Kekurangan**: Rentan terhadap _overfitting_, dan _hyperparameter tuning_ dapat memakan waktu.

### K-Nearest Neighbors (KNN)

#### Parameter yang Digunakan:
-	**n_neighbors=9**: Jumlah tetangga yang digunakan untuk memprediksi nilai.

#### Cara Kerja:

- _KNN_ memprediksi nilai dengan mencari k titik terdekat dalam ruang fitur.
- Prediksi dilakukan dengan mengambil rata-rata atau mayoritas dari nilai-nilai target dari tetangga terdekat.
- Bergantung pada pengukuran jarak, seperti _Euclidean distance_ atau _Manhattan distance_.

#### Kelebihan dan Kekurangan:
-	**Kelebihan**: Sederhana, mudah diimplementasikan, bekerja baik untuk dataset kecil.
-	**Kekurangan**: Rentan terhadap noise dan outlier, performa dapat dipengaruhi oleh pemilihan jumlah tetangga.

### Penjelasan Bagaimana Algoritma Bekerja:

- **_SVR_**: Membentuk hyperplane dengan margin maksimum untuk memprediksi nilai regresi dengan meminimalkan kesalahan prediksi.
- **_Gradient Boosting_**: Membangun serangkaian model prediksi secara berurutan dan menggabungkan prediksi untuk meningkatkan akurasi.
- **_KNN_**: Memanfaatkan informasi dari tetangga terdekat untuk memprediksi nilai berdasarkan rata-rata atau mayoritas tetangga.

## Evaluation
Dalam proyek ini, metrik evaluasi yang digunakan untuk mengukur performa model adalah _Mean Squared Error (MSE)_. _MSE_ digunakan karena tugas ini merupakan masalah regresi, di mana kita berfokus pada prediksi nilai numerik. _MSE_ mengukur rata-rata kuadrat dari selisih antara nilai prediksi dan nilai sebenarnya.

### Formula _Mean Squared Error (MSE)_:

$MSE = \frac{1}{N} \Sigma_{i=1}^{N} (y_i - \hat{y}_i)^2$

**_Keterangan_:**
- **N** adalah jumlah total observasi dalam dataset.
- $\Sigma_{i=1}^N$ menunjukkan penjumlahan dari i=1 hingga N, yang berarti kita menjumlahkan seluruh observasi dalam dataset.
- $(y_i - \hat{y}_i)^2$ adalah selisih kuadrat antara nilai sebenarnya dan nilai prediksi. Ini dilakukan untuk setiap observasi.
- $\frac{1}{N}$ adalah invers dari jumlah total observasi (N), yang memberikan rata-rata dari selisih kuadrat.

Dengan demikian, _MSE_ mengukur rata-rata kuadrat dari selisih antara nilai sebenarnya dan nilai prediksi. Semakin kecil nilai _MSE_, semakin baik model dalam memprediksi nilai target. _MSE_ memberikan penalti yang lebih besar untuk kesalahan yang lebih besar, dan karenanya, model diharapkan dapat menghasilkan prediksi yang lebih akurat.

### Analisis _Mean Squared Error (MSE)_:
Mean Squared Error (MSE) digunakan sebagai metrik evaluasi untuk mengukur seberapa baik model regresi dapat memprediksi nilai target.

**Tabel 3**. Evaluasi Model berdasarkan _Mean Squared Error (MSE)_
| Model             | Train MSE  | Test MSE   |
|-------------------|------------|------------|
| SVR               | 13.521845  | 16.769305  |
| KNN               | 3.954466   | 5.931945   |
| GradientBoosting  | 2.236794   | 6.715785   |

Tabel 3. memperlihatkan hasil evaluasi performa tiga model berbeda (_SVR, KNN, dan Gradient Boosting_) berdasarkan _Mean Squared Error (MSE)_. _MSE_ diukur pada data pelatihan (_Train MSE_) dan data pengujian (_Test MSE_). Semakin kecil nilai _MSE_, semakin baik kinerja model dalam memprediksi nilai target.

### Visualization of Model Comparison

![9](https://github.com/aririyadi/P1-MLT-Predictive-Analytics/assets/147322531/21e2bcf4-3e50-4efb-81de-70c30a0ce79d)

**Berikut adalah analisis lebih lanjut dari grafik diatas:**

**_Support Vector Regression (SVR)_:**
_SVR_ memiliki performa yang kurang baik dengan _MSE_ yang tinggi pada data pengujian. Model mungkin terlalu kompleks atau memerlukan penyesuaian parameter untuk meningkatkan generalisasi.

**_K-Nearest Neighbors (KNN)_:**
_KNN_ menunjukkan hasil yang lebih baik dibandingkan _SVR_, tetapi masih terdapat perbedaan antara _MSE_ pelatihan dan pengujian. Model mungkin perlu disesuaikan untuk mengatasi _overfitting_ atau _underfitting_.

**_Gradient Boosting_:**
_Gradient Boosting_ memiliki _MSE_ yang rendah pada data pelatihan, tetapi terdapat peningkatan pada data pengujian. Mungkin diperlukan penyetelan parameter atau teknik _regularisasi_ untuk meningkatkan _generalisasi_.

### Prediction
Melalui langkah-langkah ini, kita dapat melihat bagaimana setiap model yang telah dilatih merespons terhadap subset data uji yang telah dipilih. Hal ini membantu dalam mengevaluasi kemampuan prediktif model pada situasi yang belum pernah dilihat sebelumnya dan memberikan gambaran tentang sejauh mana model dapat menghasilkan prediksi yang mendekati nilai sebenarnya.

**Tabel 4**. Perbandingan Hasil Prediksi dengan Nilai Sebenarnya
| y_true       | prediksi_SVR | prediksi_GradientBoosting | prediksi_KNN |
|--------------|--------------|---------------------------|--------------|
| 177.970001   | 177.3        | 178.9                     | 178.4        |

Tabel 4. menunjukkan perbandingan antara nilai sebenarnya (_y_true_) dengan hasil prediksi dari tiga model yang digunakan: _Support Vector Regression (SVR), Gradient Boosting, dan K-Nearest Neighbors (KNN)_. Dalam setiap kolom prediksi, terdapat nilai prediksi yang diperoleh dari masing-masing model.

## Conclusion
Dalam proyek analisis prediktif harga saham META Platforms, Inc. sejumlah tantangan dan tujuan telah diidentifikasi. Pendekatan terstruktur digunakan, menerapkan teknik analisis prediktif dengan algoritma Machine Learning seperti _Support Vector Regression, K-Nearest Neighbors, dan Boosting Algorithm_. Fokus utama proyek ini adalah pada optimasi data preprocessing, termasuk penanganan _outlier_, manajemen _missing values_, dan penerapan _feature engineering_. Penyetelan _hyperparameter_ dilakukan melalui teknik _Grid Search_ untuk memastikan model bekerja dengan performa terbaik. Keseluruhan, proyek ini bertujuan memberikan wawasan mendalam mengenai pergerakan harga saham META Platforms, Inc. dan membangun model prediktif yang akurat. Harapannya, hasil analisis ini dapat memberikan nilai tambah dalam pengambilan keputusan investasi serta meningkatkan pemahaman tentang pemanfaatan data historis dalam meramalkan perilaku pasar keuangan.

## Reference

**[1]**.   Brown, S. J., Rozeff, M. S., & Ball, R. (1976). The Influence of Institutional Investors on Myopic R&D Investment Behavior. *Journal of Finance*, 31(5), 1653-1665.

**[2]**.   Chen, J., Fan, Y., & Li, Q. (2014). Online Daily Stock Trading with Regularized Linear Models. *Journal of Business & Economic Statistics*, 32(2), 267-287.

**[3]**.   Siegel, E. (2013). *Predictive Analytics: The Power to Predict Who Will Click, Buy, Lie, or Die*. Wiley.

**[4]**.   Heaton, J., Polson, N. G., & Witte, J. H. (2017). Deep Learning for Finance: Deep Portfolios. *Applied Stochastic Models in Business and Industry*, 33(1), 3-12.

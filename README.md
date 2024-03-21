# Laporan Proyek Machine Learning - Joseph Tedja Nugraha Wibawa

## Domain Proyek

Dalam pengumpulan data yang dilakukan oleh GA Roth dan kawan-kawan pada artikel berjudul [Global, regional, and national age-sex-specific mortality for 282 causes of death in 195 countries and territories, 1980–2017: a systematic analysis for the Global Burden of Disease Study 2017](<https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(18)32203-7/fulltext>) dan pemaparan yang oleh Dong Zhao pada artikel berjudul [Epidemiological Features of Cardiovascular Disease in Asia](https://www.jacc.org/doi/epdf/10.1016/j.jacasi.2021.04.007) dipaparkan bahwa meskipun terdapat kemajuan pada bidang medis terutama pada pencegahan secara efektif dan aman, Penyakit jantung atau yang lebih dikenal dengan Cardiovascular Disease (CVD) tetap menjadi salah satu penyebab kematian terbesar di seluruh dunia.

### Mengapa masalah ini harus diselesaikan

Menghadapi masalah ini, deteksi dini adalah kunci untuk berhasil dalam pencegahan dan manajemen penyakit. Walau begitu, deteksi dini secara konvensional seringkali memerlukan serangkaian tes mahal dan prosedur invasif. Hal ini menjadi masalah apalagi pada daerah yang memiliki keterbatasan medis diagnosis bahkan tidak mungkin dilakukan.

### Bagaimana masalah ini harus diselesaikan

Dalam hal ini, menggunakan algoritme machine learning memungkinkan intervensi yang tepat waktu dan tepat sasaran untuk mengurangi risiko komplikasi serius seperti serangan jantung dan gagal jantung. Selain itu, penggunaan pengembangan model prediktif yang dapat mengidentifikasi pola tersembunyi dalam data pasien, membuahkan cara yang ekonomis dan non-invasif dalam mendiagnosa penyakit jantung.

Dalam sudut pandang medis, penerapan algoritme machine learning yang memanfaatkan data pasien dan informasi klinis untuk memprediksi risiko penyakit jantung dapat membantu para praktisi kesehatan untuk lebih cepat dan efisien dalam menentukan strategi pencegahan dan pengobatan. Pada daerah dengan ketersediaan medis yang minim, ini tidak hanya meningkatkan hasil kesehatan untuk pasien tetapi juga dapat meringankan beban sistem kesehatan dengan lebih efektif dalam mengalokasikan sumber daya ke pasien yang paling membutuhkan.

Oleh karena itu, saya tertarik mengangkat proyek klasifikasi biner penyakit jantung dalam rangka meningkatkan kualitas kesehatan masyarakat.

## Business Understanding

### Problem Statements

- Metode diagnosis konvensional yang berbiaya tinggi dan invasif, dan
- Metode diagnosis konvensional yang tidak dapat dilakukan pada daerah berketerbatasan medis.

### Goals

- Pembuatan suatu metode diagnosis yang ekonomis dan non-invasif, dan
- Pembuatan suatu metode diagnosis yang dapat diterapkan pada daerah berketerbatasan medis.

### Solution statements

- Membuat model dengan algoritma `KNNClassifier` sebagai pembanding untuk mencapai hasil paling ideal,
- Membuat model dengan algoritma `RandomForestClassifier` sebagai pembanding untuk mencapai hasil paling ideal,
- Membuat model dengan algoritma `DecisionTreeClassifier` sebagai pembanding untuk mencapai hasil paling ideal,
- Membuat model dengan algoritma `LogisticRegression` sebagai pembanding untuk mencapai hasil paling ideal,

## Data Understanding

Dataset yang digunakan untuk project ini adalah dataset Behavioral Risk Factor Surveillance System (BRFSS) yang dapat diunduh dari [Kaggle](https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset). BRFSS adalah sistem survei telepon terkemuka di Amerika Serikat (AS) yang mengumpulkan data perilaku berisiko terkait kesehatan, kondisi kesehatan kronis, dan penggunaan layanan pencegahan penduduk AS.

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- `General_Health` : Kesehatan responden secara mendasar.
- `Checkup` : Terakhir kali responden melakukan pemeriksaan kesehatan.
- `Exercise` : Responden melakukan aktivitas fisik dalam jangka waktu satu bulan terakhir (Ya/Tidak).
- `Heart_Disease` : Responden melaporkan memiliki riwayat penyakit jantung (Ya/Tidak).
- `Skin_Cancer` : Responden melaporkan memiliki riwayat kanker kulit (Ya/Tidak).
- `Other_Cancer` : Responden melaporkan memiliki riwayat kanker lainnya (Ya/Tidak).
- `Depression` : Responden melaporkan mengalami depresi (Ya/Tidak).
- `Diabetes` : Responden melaporkan riwayat diabetes, disertai tipe diabetes jika diperlukan.
- `Arthritis` : Responden melaporkan riwayat arthritis, disertai tipe diabetes jika diperlukan.
- `Sex` : Jenis kelamin responden.
- `Age_Category` : Kategori umur responden.
- `Height_(cm)` : Tinggi responden dalam cm.
- `Weight_(kg)` : Berat responden dalam kg.
- `BMI` : Body Mass Index (BMI) responden.
- `Smoking_History` : Adanya riwayat merokok pada responden.
- `Alcohol_Consumption` : Angka konsumsi alkohol responden.
- `Fruit_Consumption` : Angka konsumsi buah responden.
- `Green_Vegetables_Consumption` : Angka konsumsi sayuran hijau responden.
- `FriedPotato_Consumption` : Angka konsumsi kentang goreng responden.

## Exploratory Data Analysis

### Data Visualization

Data awal divisualisasikan dalam projek ini dengan memanggil `diseases.head()` yang menghasilkan sebuah tabel berisikan 5 data teratas sebagai berikut:

| General_Health | Checkup                 | Exercise | Heart_Disease | Skin_Cancer | Other_Cancer | Depression | Diabetes | Arthritis | Sex    | Age_Category | Height\_(cm) | Weight\_(kg) | BMI   | Smoking_History | Alcohol_Consumption | Fruit_Consumption | Green_Vegetables_Consumption | FriedPotato_Consumption |
| -------------- | ----------------------- | -------- | ------------- | ----------- | ------------ | ---------- | -------- | --------- | ------ | ------------ | ------------ | ------------ | ----- | --------------- | ------------------- | ----------------- | ---------------------------- | ----------------------- |
| Poor           | Within the past 2 years | No       | No            | No          | No           | No         | No       | Yes       | Female | 70-74        | 150.0        | 32.66        | 14.54 | Yes             | 0.0                 | 30.0              | 16.0                         | 12.0                    |
| Very Good      | Within the past year    | No       | Yes           | No          | No           | No         | Yes      | No        | Female | 70-74        | 165.0        | 77.11        | 28.29 | No              | 0.0                 | 30.0              | 0.0                          | 4.0                     |
| Very Good      | Within the past year    | Yes      | No            | No          | No           | No         | Yes      | No        | Female | 60-64        | 163.0        | 88.45        | 33.47 | No              | 4.0                 | 12.0              | 3.0                          | 16.0                    |
| Poor           | Within the past year    | Yes      | Yes           | No          | No           | No         | Yes      | No        | Male   | 75-79        | 180.0        | 93.44        | 28.73 | No              | 0.0                 | 30.0              | 30.0                         | 8.0                     |
| Good           | Within the past year    | No       | No            | No          | No           | No         | No       | No        | Male   | 80+          | 191.0        | 88.45        | 24.37 | Yes             | 0.0                 | 8.0               | 4.0                          | 0.0                     |

### Check Missing Values

Pengecekan keberadaan nilai kosong atau _Missing Values_ (`NaN`) dilakukan dengan memanggil `diseases.info()`, dan `diseases.describe()`

Dari hasil memanggil beberapa metode tersebut, diketahui bahwa tidak ada data kosong.

### Check Outliers

Pengecekan keberadaan nilai asing atau _Outliers_ dilakukan dengan menampilkan box plot menggunakan library _seaborn_

Pengecekan menghasilkan bahwa ada beberapa _outlier_ dalam data.

### Outlier Removal

Penghapusan outlier dilakukan dengan metode _Interquartile Range_ yang mengidentifikasi outlier sebagai data yang berada di luar Q1 dan Q3. Dalam [Experimental Design Analysis](https://www.stat.cmu.edu/~hseltman/309/Book/Book.pdf) karya Seltman, dijelaskan bahwa data yang memiliki nilai `1.5*IQR` di atas Q3 atau `1.5*IQR` di bawah Q1 dapat dikatakan sebagai outlier.

Dari penghapusan outlier, ukuran data yang tadinya `(308853, 19)` menjadi `(184533, 19)`.

### Univariate Analysis

Univariate Analysis merupakan metode analisa variable satu per satu.

*Categorical*

- `General_Health` dengan proporsi berikut:
  - `Very Good` => 35.5%
  - `Good` => 31.9%
  - `Excellent` => 16.8%
  - `Fair` => 12.0%
  - `Poor` => 3.8%
- `Checkup` dengan proporsi berikut:
  - `Within the past year` => 77.7%
  - `Within the past 2 years` => 12.0%
  - `Within the past 5 years` => 5.6%
  - `5 or more years ago` => 4.2%
  - `Never` => 0.5%
- `Exercise` dengan proporsi berikut:
  - `Yes` => 75.5%
  - `No` => 24.5%
- `Heart_Disease` dengan proporsi berikut:
  - `Yes` => 8.5%
  - `No` => 91.5%
- `Skin_Cancer` dengan proporsi berikut:
  - `Yes` => 9.3%
  - `No` => 90.7%
- `Other_Cancer` dengan proporsi berikut:
  - `Yes` => 9.7%
  - `No` => 90.3%
- `Depression` dengan proporsi berikut:
  - `Yes` => 19.9%
  - `No` => 80.1%
- `Diabetes` dengan proporsi berikut:
  - `Yes` => 13.5%
  - `No` => 83.3%
  - `No, pre-diabetes` => 2.3%
  - `Yes, but only during pregnancy` => 0.8%
- `Arthritis` dengan proporsi berikut:
  - `Yes` => 32.4%
  - `No` => 67.6%
- `Sex` dengan proporsi berikut:
  - `Female` => 52.2%
  - `Male` => 47.8%
- `Age_Category` dengan proporsi berikut:
  - `65-69` => 10.7%
  - `60-64` => 10.4%
  - `70-74` => 10.1%
  - `55-59` => 9.0%
  - `50-54` => 8.1%
  - `80+` => 7.5%
  - `75-79` => 6.8%
  - `45-49` => 6.7%
  - `40-44` => 6.6%
  - `18-24` => 6.6%
  - `35-39` => 6.4%
  - `30-34` => 5.9%
  - `25-29` => 5.2%
- `Smoking_History` dengan proporsi berikut:
  - `Yes` => 40.4%
  - `No` => 59.6%

Dari beberapa porsi di atas, dapat disimpulkan bahwa mayoritas variabel yang ada tidak memiliki proporsi yang seimbang.

*Numerical*

Terdapat beberapa data `Weight_(kg)`, `BMI`, dan `Height_(cm)` yang terdistribusi normal atau _bell curved_. Data `Alcohol_Consumption`, `Green_Vegetables_Consumption`, dan `FriedPotato_Consumption` memiliki distribusi miring kanan atau _skewed right_. Data `Fruit_Consumption` memiliki distribusi miring kiri atau _skewed left_.

### Multivariate Analysis

Perbandingan antara dua atau lebih variabel sejenis untuk mencari korelasi

*Categorical*

Pada bagian kategorikal, dilakukan perbandingan antara semua variabel kategorikal dengan label.

- `Heart_Disease` terhadap `General_Health`
  Menunjukkan bahwa sampel dengan kesehatan buruk dan biasa memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Checkup`
  Menunjukkan bahwa sampel dengan yang terakhir checkup dalam satu tahun terakhir memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Exercise`
  Menunjukkan bahwa sampel yang tidak berolahraga memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Skin_Cancer`
  Menunjukkan bahwa sampel yang beriwayat kanker kulit memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Other_Cancer`
  Menunjukkan bahwa sampel yang beriwayat kanker lain memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Depression`
  Menunjukkan bahwa sampel yang beriwayat depresi memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Diabetes`
  Menunjukkan bahwa sampel yang beriwayat diabetes (kecuali diabetes pada wanita mengandung) memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Arthritis`
  Menunjukkan bahwa sampel yang beriwayat arthritis memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Sex`
  Menunjukkan bahwa sampel yang berjenis kelamin laki laki memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.
- `Heart_Disease` terhadap `Age_Category`
  Menunjukkan bahwa sampel yang berumur di atas 55 memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung. Tendensi ini terus meningkat hingga puncaknya pada umur di atas 80 tahun.
- `Heart_Disease` terhadap `Smoking_History`
  Menunjukkan bahwa sampel yang beriwayat merokok memiliki tendensi lebih tinggi untuk teridentifikasi mengidap penyakit jantung.

*Numerical*

Pada bagian numerikal, semua variabel numerik dibandingkan satu dengan yang lain untuk mencari korelasi baik pada grafik _pairplot_ dan _correlation matrix_

Dari pembandingan menggunakan grafik grafik di atas, didapatkan beberapa variabel numerik yang memiliki persebaran seruba dan korelasi tinggi:

- `Weight_(kg)` dengan `BMI` => 0.86
- `Weight_(kg)` dengan `Height_(cm)` => 0.47
- `Fruit_Consumption` dengan `Green_Vegetable_Consumption` => 0.27

## Data Preparation

### One Hot Encoding Multivalued Categorical Columns

Pada bagian ini dilakukan transformasi variabel kategorikal dengan lebih dari 2 kategori menggunakan metode _One Hot Encoding_. Beberapa variabel yang memenuhi kriteria ini adalah `General_Health`, `Checkup`, `Age_Category`, dan `Diabetes`.

_One Hot Encoding_ membuat masing-masing kategori menjadi kolom-kolom baru yang berisikan 0 dan 1. Dimana letak 1 diantara 0 digunakan untuk menandai kategori aktif untuk sampel tersebut.

Alasan melakukan proses ini adalah untuk merubah data kategorikal bernilai banyak dari yang mudah untuk dipahami manusia, menjadi mudah dipahami oleh algoritma machine learning yang akan digunakan.

### Binary Encoding Bivalued Categorical Columns

Pada bagian ini dilakukan transformasi variabel kategorikal dengan 2 kategori menggunakan metode _Binary Encoding_. Beberapa variabel yang memenuhi kriteria ini adalah `Exercise`, `Heart_Disease`, `Skin_Cancer`, `Other_Cancer`, `Depression`, `Arthritis`, `Sex`, dan `Smoking_History`.

_Binary Encoding_ mengubah suatu kolom yang berisikan dua kategori dalam bentuk string menjadi 1 dan 0. Dimana 1 pada umumnya merepresentasikan nilai positif seperti yes, dan 0 sebaliknya.

Alasan melakukan proses ini adalah untuk merubah data kategorikal bernilai dua dari yang mudah untuk dipahami manusia, menjadi mudah dipahami oleh algoritma machine learning yang akan digunakan.

### PCA Dimension Reductibility Test

Dari hasil _pair plot_ dan _correlation matrix_ yang didapat pada bagian _numerical multivariate analysis_ sebelumnya, diketahui 3 kandidat variabel untuk reduksi dimensi. Diantaranya adalah `Weight_(kg)`, `BMI`, dan `Height_(cm)`.

Terdapat dua skenario dalam melakukan reduksi disini, yaitu:

- Mereduksi ketiga variabel menjadi satu, atau
- Mereduksi dua variabel dengan korelasi tertinggi menjadi satu.

Dari hasil explained_variance_ratio, didapat bahwa:

- Mereduksi ketiga variabel menghasilkan tiga PC dengan proporsi informasi ```[0.825, 0.174, 0.01]```.
- Mereduksi dua variabel berkolerasi tertinggi menghasilkan dua PC dengan proporsi informasi ```[0.977, 0.023]```.

Mereduksi dua variabel berkolerasi tertinggi dirasa lebih baik karena percobaan pada dua variabel `Weight_(kg)`, dan `BMI` menghasilkan persentase penyimpanan informasi lebih tinggi.

Alasan melakukan proses ini adalah untuk mengurangi dimensionalitas yang ada namun tetap mempertahankan keutuhan informasi.

### Train Test Split

Pada tahap ini, dilakukan pemisahan dataset pelatihan dan dataset penujian. Pemisahan dilakukan dengan rasio pelatihan : pengujian, 9 : 1. Sehingga merubah total sampel dataset yang berjumlah *184533* menjadi *166079* data pelatihan dan *18454* data pengujian.

Alasan melakukan proses ini adalah untuk memisahkan data yang akan digunakan untuk menilai performa model dengan data yang akan digunakan untuk pelatihan. Tujuannya agar model dapat dinilai secara lebih objektif yaitu dengan data yang tidak pernah dilihat selama pelatihan.

### Standarization

```StandardScaler``` digunakan pada tahap ini untuk merubah data numerik yang awalnya memiliki nilai yang teramat besar dan kecil menjadi terstandarisasi. 

Standarisasi dengan ```StandardScaler``` dilakukan dengan mencari nilai Z<sub>score</sub> menggunakan rumus:

Z<sub>score</sub> = (_x_ - _μ_)/_s_

Dimana:
- _x_ adalah nilai data yang ingin dirubah,
- _μ_ adalah rata rata nilai data, dan
- _s_ adalah standar deviasi data.

Standarisasi awalnya dilakukan pada dataset pelatihan. Lalu, obyek ```StandardScaler``` yang sudah digunakan pada train dataset, digunakan pada dataset pengujian. Hal ini dilakukan untuk mencegah kebocoran data pelatihan ke data pengujian.

Alasan melakukan standarisasi adalah memastikan setiap fitur memberikan kontribusi yang setara dalam perhitungan jarak pada algoritma yang sensitif terhadap skala fitur, sehingga meningkatkan kinerja model dan mempercepat konvergensi algoritma.

## Modeling

Pada tahap ini, digunakan 4 buah model yaitu, ```KNNClassifier```, ```RandomForestClassifier```, ```DecisionTreeClassifier```, dan ```LogisticRegression```. Metrik evaluasi yang digunakan pada modelling disini adalah *f1_score*.

### KNN Classifier

*Kelebihan:*
- Sederhana dan mudah diimplementasikan.
- Tidak perlu asumsi statistik tentang distribusi data.
- Efektif jika dataset memiliki banyak fitur.

*Kekurangan:*
- Sensitif terhadap data yang tidak standar dan outlier.
- Komputasi intensif saat dataset besar karena memerlukan penyimpanan semua data pelatihan.
- Memerlukan pemilihan parameter k (_nearest neighbour_) yang tepat.

### RandomForest Classifier

*Kelebihan:*
- Mengurangi overfitting dengan membangun banyak _desicion trees_ dan menggunakan rata-rata hasil atau _bagging_.
- Sangat fleksibel dan memiliki kinerja yang baik secara umum.
- Dapat dengan mudah menangani fitur kategorikal dan numerik.

*Kekurangan:*
- Model yang dihasilkan bisa menjadi cukup kompleks dan memerlukan lebih banyak sumber daya komputasi.
- Lebih sulit untuk diinterpretasi dibandingkan  _desicion tree_ tunggal.
- Waktu latih yang relatif lama karena membangun banyak _desicion trees_.

### DecisionTree Classifier

*Kelebihan:*
- Mudah dipahami dan diinterpretasikan.
- Dapat menangani data numerik dan kategorikal.
- Memerlukan sedikit pra-pemrosesan data, tidak perlu penskalaan atau normalisasi.

*Kekurangan:*
- Cenderung terjadi overfitting terutama jika pohon terlalu dalam.
- Rentan terhadap varians, kecilnya perubahan pada data bisa menghasilkan pohon yang sangat berbeda.
- Tidak selalu menghasilkan batas keputusan yang paling optimal.

### Logistic Regression

*Kelebihan:*
- Sederhana dan efisien untuk masalah klasifikasi biner.
- Memberikan probabilitas prediksi selain klasifikasi biner, yang berguna untuk penilaian risiko.
- Hasil model mudah diinterpretasikan.

*Kekurangan:*
- Kurang cocok untuk kompleksitas hubungan non-linear tanpa transformasi atau penambahan fitur.
- Tidak dirancang untuk menangani fitur yang berhubungan kuat atau multikolinearitas.
- Tidak efektif jika jumlah fitur sangat besar dibandingkan jumlah sampel atau _high-dimensional spaces_.

### Best Theoretical Model
Secara teori, model terbaik berdasarkan pemaparan di atas adalah _Random Forest Classifier_ karena dapat mencegah overfitting dengan menggunakan beberapa _DecisionTree_ dalam metode _bagging_. Selain itu, jumlah komputasi yang lebih banyak tidak terlalu berpengaruh apabila menggunakan Google Collab ataupun device lokal yang cukup.

## Evaluation

### Evaluation Metric

Metrik evaluasi yang digunakan di projek ini adalah _f1<sub>score</sub>_. _f1<sub>score</sub>_ sendiri adalah sebuah metrik yang melibatkan _Precision_ dan _Recall_ dalam rumusnya. Sehingga, _f1<sub>score</sub>_ dapat cukup mewakili performa klasifikasi dari _Precision_ dan _Recall_ yang dikenal dapat lebih baik dalam memaparkan performa pada data yang tidak seimbang. Rumus dari _f1<sub>score</sub>_ sendiri adalah:

F1 = 2/(1/P + 1/R)

F1 = 2*_P_*_R_ / _P_+_R_

Pada rumus di atas, dapat diperhatikan bahwa _f1<sub>score</sub>_ adalah bentuk _Harmonic Mean_ dari _Precision_ dan _Recall_. _Harmonic Mean_ digunakan karena F1Score mencari keseimbangan dari kedua metrik dengan memberi beban lebih pada metrik yang bernilai kecil.

### Results Based on Evaluation Metric
DecisionTree Classifier merupakan model terbaik karena dalam pengujian terbukti memiliki nilai ```train_f1_score``` dan ```test_f1_score``` tertinggi.

Namun, tidak bisa dipungkiri bahwa performa pada ```test_f1_score``` jauh di bawah ```train_f1_score``` pada seluruh model. Hal ini bisa disebabkan oleh _imbalance data_ atau data yang tidak seimbang seperti yang telah didiskusikan di Univariate Analysis.

# Laporan Proyek Machine Learning Roshied Mohammad

## Domain Proyek
Stroke merupakan salah satu masalah kesehatan yang serius dengan dampak yang signifikan pada masyarakat. Penyebab dari stroke adalah gangguan pada suplai darah pada beberapa bagian dari otak. Stroke merupakan kondisi kesehatan yang membahayakan sehingga diperlukannya tindakan cepat untuk mencegah kondisi stroke yang lebih parah (Kaur, 2022).  

Penanganan stroke dipengaruhi oleh kemampuan mendeteksi dini serta intervensi yang cepat, Hal ini harus dilakukan dalam waktu periode golden time, yaitu dalam waktu kurang dari 3,5 jam setelah munculnya gejala stroke (Kustanti & Widyarani, 2023). Stroke dapat diketahui dengan tanda fisik dengan gejala awal seperti adanya wajah yang terkulai (facial droop), lengan yang sulit untuk digerakkan, dan kesulitan untuk berbicara.
 
Dengan perkembangannya teknologi, stroke dapat di deteksi dengan penggunaan machine learning. Penggunaan rekam medis dari pasien didapatkan prediksi kemungkinan terjadinya stroke (Dev et. al., 2022).

## Business Understanding
### Problem Statement
1. Bagaimana cara untuk memprediksi kemungkinan stroke ?
2. Data apa saja yang diperlukan untuk memprediksi kemungkinan stroke ?
3. Model prediksi apa yang memiliki error paling rendah ?
### Goals
1. Mengetahui cara untuk memprediksi kemungkinan stroke.
2. Mengetahui data yang diperlukan untuk memprediksi kemungkinan stroke.
3. Mengetahui model prediksi yang memiliki error paling rendah.
- Solution Statement
1. Menggunakan 3 model prediksi yaitu K-Nearest Neighbor (KNN), Random Forest, dan Algoritma Boosting
2. Menyederhanakan data-data yang masih memiliki variabel kategorik dengan _One-Hot-Encoding_.
3. Menggunakan MSE sebagai evaluasi model yang mudah dipahami dan dinterpretasikan. MSE memiliki kekurangan yaitu sensitif dengan outliers. Sehingga diperlukannya preparasi data pada bagian outliers.

## Data Understanding
Pada repository ini digunakan dataset [Stroke Prediction Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Dataset ini memeliki 5.510 pasien. Data tersebut merupakan data non-numerik seperti _work type, residence type, dan smoking status_. Serta fitur numerik seperti _age, body mass index, dan average glucose level_.
### Data Loading
Memunculkan deskripsi setiap fitur dari dataset dengan tipedatanya.

Tabel 1. Menampilkan deskripsi setiap fitur dari dataset dengan tipedatanya.
|   | Column            | Non-Null Count | Dtype   |
|---|-------------------|-----------------|---------|
| 0 | id                | 5110 non-null   | int64   |
| 1 | gender            | 5110 non-null   | object  |
| 2 | age               | 5110 non-null   | float64 |
| 3 | hypertension      | 5110 non-null   | int64   |
| 4 | heart_disease     | 5110 non-null   | int64   |
| 5 | ever_married      | 5110 non-null   | object  |
| 6 | work_type         | 5110 non-null   | object  |
| 7 | Residence_type    | 5110 non-null   | object  |
| 8 | avg_glucose_level | 5110 non-null   | float64 |
| 9 | bmi               | 4909 non-null   | float64 |
|10 | smoking_status    | 5110 non-null   | object  |
|11 | stroke            | 5110 non-null   | int64   |
- Terdapat 5110 baris (recods atau jumlah pengamatan) dalam dataset.
- Terdapat 11 kolom yaitu : id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke.
  
Pada dataset ini dilakukan exploratory data analysis untuk mengetahui lebih lanjut data-data yang nantinya akan digunakan
### Variabel-variabel pada Stroke Prediction Dataset
Deskripsi Variabel
  Deskripsi variabel dilakukan untuk mengetahu jenis-jenis variabel dan nilai-nilai dari variabel tersebut. Data tersebut kemudian dapat membantu memahami pengelolaan data yang akan dipersiapkan.

Tabel 2. Menampilkan data head dari dataset.
|id|gender|age|hypertension|heart_disease|ever_married|work_type|Residence_type|avg_glucose_level|bmi|smoking_status|stroke|
|-|-|-|-|-|-|-|-|-|-|-|-|
|9046|Male|67.0|0|1|Yes|Private|Urban|228.69|36.6|formerly smoked|1|
|51676|Female|61.0|0|0|Yes|Self-employed|Rural|202.21|NaN|never smoke|1|
|31112|Male|80.0|0|1|Yes|Private|Rural|105.92|32.5|never smoked|1|
|60182|Female|49.0|0|0|Yes|Private|Urban|171.23|34.4|smokes|1|
|1665|Female|79.0|1|0|Yes|Self-employed|Rural|174.12|24.0|never smoked|1|

Tabel 3. Menampilkan statistik deskriptif dari data numerik.
||id|age|hypertension|heart_disease|avg_glucose_level|bmi|stroke|
|-|-|-|-|-|-|-|-|
|count|5110.000000|5110.000000|5110.000000|5110.000000|5110.000000|4909.000000|5110.000000|
|mean|36517.829354|43.226614|0.097456|0.054012|106.147677|28.893237|0.048728|
|std|21161.721625|22.612647|0.296607|0.226063|45.283560|7.854067|0.215320|
|min|67.000000|0.080000|0.000000|0.000000|55.120000|10.300000|0.000000|
|25%|17741.250000|25.000000|0.000000|0.000000|77.245000|23.500000|0.000000|
|50%|36932.000000|45.000000|0.000000|0.000000|91.885000|28.100000|0.000000|
|75%|54682.000000|61.000000|0.000000|0.000000|114.090000|33.100000|0.000000|
|max|72940.000000|82.000000|1.000000|1.000000|271.740000|97.600000|1.000000|


- Gender : merupakan jenis kelamin dari pasien yaitu male, female atau other
- Age : merupakan umur atau usia dari pasien 0.8 sampai 82 tahun
- Hypertension : merupakan pengidap hipertensi dari pasien yaitu 0 (tidak mengalami hypertension) dan 1 (mengalami hypertension)
- Heart Disease : merupakan pengidap penyakit jantung dari pasien yaitu 0 (tidak mengalami penyakit jantung) dan 1 (mengalami penyakit jantung)
- Ever Married : merupakan kondisi pasien telah menikah yaitu 0 (belum menikah) dan 1 (telah menikah)
- Work Type : merupakan jenis pekerjaan yang dilakukan pasien yaitu private, self employed, never_worked, govt_job, dan children.
- Residence Type : merupakan jenis tempat tinggal pasien yaitu urban dan rural
- Average Glucose Level : merupakan nilai rata-rata level gula darah pasien dari 55 sampai 271.24
- Body Mass Index : merupakan nilai berat badan ideal dari pasien dari 10.3 sampai 97.6

dengan mendeskripsikan variabel-variabel tersebut dapat diketahui  Variabel numerik yaitu _age, body mass index, dan average glucose level_ dan variabel kategori yaitu _work type, residence type, dan smoking status_.
- Missing Value
  Dari data yang sudah diketahui jenis variabel tersebut dapat dilihat bahwa terdepat beberapa sampel yang memiliki variabel dengan nilai nol atau tidak memiliki nilai (missing value). Untuk mengurangi error dalam penggunaan dataset sampel-sampel yang memiliki missing value akan dikeluarkan dari dataset.
Sehingga,
Nilai 0 di kolom avg_glucose_level:  0
Nilai 0 di kolom bmi:  201

Tabel 4. Menampilkan dataset head dan tail dari dataset yang sudah tidak memiliki missing value. 
|| id | gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi | smoking_status | stroke |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 9046 | Male | 67.0 | 0 | 1 | Yes | Private | Urban | 228.69 | 36.6 | formerly smoked | 1 |
| 31112 | Male | 80.0 | 0 | 1 | Yes | Private | Rural | 105.92 | 32.5 | never smoked | 1 |
| 60182 | Female | 49.0 | 0 | 0 | Yes | Private | Urban | 171.23 | 34.4 | smokes | 1 |
| 1665 | Female | 79.0 | 1 | 0 | Yes | Self-employed | Rural | 174.12 | 24.0 | never smoked | 1 |
| 56669 | Male | 81.0 | 0 | 0 | Yes | Private | Urban | 186.21 | 29.0 | formerly smoked | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5104 | Female | 13.0 | 0 | 0 | No | children | Rural | 103.08 | 18.6 | Unknown | 0 |
| 44873 | Female | 81.0 | 0 | 0 | Yes | Self-employed | Urban | 125.20 | 40.0 | never smoked | 0 |
| 19723 | Female | 35.0 | 0 | 0 | Yes | Self-employed | Rural | 82.99 | 30.6 | never smoked | 0 |
| 37544 | Male | 51.0 | 0 | 0 | Yes | Private | Rural | 166.29 | 25.6 | formerly smoked | 0 |
| 44679 | Female | 44.0 | 0 | 0 | Yes | Govt_job | Urban | 85.28 | 26.2 | Unknown | 0 |

4904 rows x 12 columns

- Outliers
  Kemudian sampel-sample yang memiliki nilai yang sangat jauh dari cakupan umum juga dikeluarkan dari dataset.

### Fitur age
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/c307b0e0-c1df-49a9-9d86-971ba75add43)

Gambar 1. IQR Fitur Age

Pada gambar 1, menampilkan rentang IQR pada fitur age yang berada pada rentang 20 sampai 60.

### Fitur avg_glucose_level
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/4fb5a353-3aad-4620-a6e7-5fe6d419966e)

Gambar 2. IQR Fitur avg_glucose_level

Pada gambar 2, menampilkan rentang IQR pada fitur avg_glucose_level yang berada pada rentang 50 sampai 150 dengan outliers setelah 150.

### Fitur bmi
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/842d8d14-1c5e-49ef-818f-d2402cc89464)

Gambar 3. IQR Fitur bmi

Pada gambar 3, menampilkan rentang IQR pada fitur bmi yang berada pada rentang 20 sampai 40 dengan outliers setelah 40.
### Membuat batas bawah dan atas
Menggunakan metode IQR untuk mengidentifikasikan outliers yang berada di luar Q1 dan Q3. Nilai apapun yang berada di luar batas ini dianggap sebagai outlier.
Didapatkan batas bawah dan atas yaitu (1372, 22) setelah menggunakan perhitungan IQR=Q3-Q1

## Multivariate Analysis
### Categorical Features
Pada tahap ini, pengecekan rata-rata umur terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap umur.

![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/921f0bc8-6e6e-4a3d-acec-9859fde62848)
Gambar 4. Rata-rata 'age' relatif terhadap gender
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/6a9774b9-92ac-46a6-8d02-b142502f2ff5)
Gambar 5. Rata-rata 'age' relatif terhadap ever_married
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/7d698ad6-5e47-404a-b732-0cc91171cd0a)
Gambar 6. Rata-rata 'age' relatif terhadap work_type
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/9a69c0a6-a709-419f-a095-e9893bca5fc8)
Gambar 7. Rata-rata 'age' relatif terhadap Residence_type

- Pada fitur gender umur cederung memiliki jumlah data yang hampir sama pada rentang 35-40.
- Pada fitur ever_married terlihat perbedaan yaitu pada umur 20-50 telah menikah.
- Pada fitur residence_type terbagi rata pada rural dan urban.

### Numerical Features
Untuk mengamati hubungan antara fitur numerik menggunakan fungsi fairplot().

![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/5835592b-d743-4b9f-9306-9038403af442)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/01bf5be2-3ec5-4afb-abd4-e86c5fa2aac1)
Gambar 8. Grafik Numerical Features

### Correlation Matrix
Membuat dan menampilkan matriks korelasi antara fitur-fitur numerik dalam dataset.

![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/e1e92353-08ff-4f31-bc33-796d7e2d40ea)

Gambar 9. Grafik Correlation Matrix

## Data Preparation
Pada dataset ini dilakukan tiga tahap persiapan data, yaitu:
### Encoding fitur kategori 
Terdapat 5 data kategori yaitu 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status' yang akan dilakukan proses encoding menggunakan teknik _one-hot-encoding_. Teknik ini memberikan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Metode ini dilakukan dengan fungsi pd.get_dummies()
Fitur kategori tidak dapat langsung digunakan dalam model predictive analytics. Oleh karena itu, perlu dilakukan proses encoding untuk mengubah fitur kategori menjadi numerik.
Alasan dilakukannya encoding fitur kategori pada dataset ini adalah:
- Model predictive analytics umumnya lebih mudah bekerja dengan fitur numerik daripada fitur kategori.
- Encoding fitur kategori dapat membantu model untuk membedakan antar kategori.

Tabel 5. Menampilkan dataset dengan fitur kategori yang telah dikonversi menjadi fitur numerik.
| age | hypertension | heart_disease | avg_glucose_level | bmi | stroke | gender_Female | gender_Male | gender_Other | ever_married_No | ... | work_type_Never_worked | work_type_Private | work_type_Self-employed | work_type_children | Residence_type_Rural | Residence_type_Urban | smoking_status_Unknown | smoking_status_formerly_smoked | smoking_status_never_smoked | smoking_status_smokes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 249 | 3.0 | 0 | 95.12 | 18.0 | 0 | 0 | 1 | 0 | 1 | ... | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| 251 | 8.0 | 0 | 110.89 | 17.6 | 0 | 1 | 0 | 0 | 1 | ... | 0 | 1 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| 252 | 70.0 | 0 | 69.04 | 35.9 | 0 | 1 | 0 | 0 | 0 | ... | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| 253 | 14.0 | 0 | 161.28 | 19.1 | 0 | 0 | 1 | 0 | 1 | ... | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| 255 | 52.0 | 0 | 77.59 | 17.7 | 0 | 1 | 0 | 0 | 0 | ... | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 5104 | 13.0 | 0 | 103.08 | 18.6 | 0 | 1 | 0 | 0 | 1 | ... | 0 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 0 |
| 5106 | 81.0 | 0 | 125.20 | 40.0 | 0 | 1 | 0 | 0 | 0 | ... | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 |
| 5107 | 35.0 | 0 | 82.99 | 30.6 | 0 | 1 | 0 | 0 | 0 | ... | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| 5108 | 51.0 | 0 | 166.29 | 25.6 | 0 | 0 | 1 | 0 | 0 | ... | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| 5109 | 44.0 | 0 | 85.28 | 26.2 | 0 | 1 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |

3755 rows x 22 columns

### Train-Test-Split
Pembagian dataset dengan fungsi train_test_split dari library sklearn
Dataset dibagi menjadi data train dan data test untuk menguji seberapa baik generalisasi model terhadap data baru. Pembagian yang dilakukan yaitu 80:20. Metode ini menggunakan fungsi sklearn.model_selection.train_test_split()
Alasan dilakukannya train-test-split pada dataset ini adalah:
- Data train digunakan untuk membangun model.
- Data test digunakan untuk mengevaluasi model.
- Pembagian 80:20 merupakan pembagian yang umum digunakan dalam predictive analytics.
Sehingga, didapat data yaitu :
Total # of sample in whole dataset: 3755
Total # of sample in train dataset: 3004
Total # of sample in test dataset: 751

### Standarisasi
Standardisasi merupakan teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan dengan menggunakan fungsi sklearn.preprocessing.StandardScaler()
Alasan dilakukannya standarisasi pada dataset ini adalah:
- Standarisasi dapat membantu model untuk lebih cepat konvergen.
- Standarisasi dapat membantu model untuk lebih akurat.

Tabel 6. Menampilkan head data avg_glucose_level dan bmi yang telah standarisasi.
||avg_glucose_level|bmi|
|-|-|-|
|520|-0.799319|2.571418|
|1814|-0.139998|0.750773|
|1938|-0.348437|-0.365105|
|1777|0.269374|-0.805584|
|1720|-0.916787|0.354343|

Tabel 7. Menampilkan statistik deskriptif ringkas dari fitur-fitur numerik.
| | avg_glucose_level | bmi |
|---|---|---|
|count | 3004.0000 | 3004.0000 |
|mean | -0.0000 | 0.0000 |
|std | 1.0002 | 1.0002 |
|min | -1.6026 | -2.5235 |
|25% | -0.7273 | -0.7175 |
|50% | -0.1561 | -0.0715 |
|75% | 0.5585 | 0.6333 |
|max | 3.3889 | 2.9385 |

## Modeling
Pada Tahapan modeling digunakan tiga algoritma yaitu :
### K-Nearest Neighbor
KNN adalah algoritma pembelajaran mesin yang bekerja dengan cara mencari k tetangga terdekat dari titik data yang akan diprediksi. Titik data yang paling dekat dengan titik data yang akan diprediksi akan memiliki nilai prediksi yang paling mungkin.
KNN memiliki kelebihan mudah diimplementasikan, tidak memerlukan pra-pemrosessan data yang kompleks, dan dapat menangani data numerik dan kategorikal. KNN memiliki kekurangan yaitu sensitif terhadap noise dan menjadi lambat untuk data yang besar.
Pada tahap ini, dilakukan proses berikut:
- Memilih nilai k
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
Nilai k yang dipilih adalah 10. Nilai k ini dipilih berdasarkan eksperimen yang dilakukan. Hasil eksperimen menunjukkan bahwa nilai k 10 memberikan akurasi terbaik.

### Random Forest
RF adalah algoritma pembelajaran mesin yang bekerja dengan cara membangun sejumlah pohon keputusan secara acak. Hasil prediksi dari sejumlah pohon keputusan tersebut kemudian digabungkan untuk menghasilkan prediksi akhir.
Pada tahap ini, dilakukan proses berikut:
- Memilih jumlah pohon
- Memilih metode bootstrap
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
Jumlah n yang dipilih adalah 50. Jumlah ini dipilih berdasarkan eksperimen yang dilakukan. Hasil ekseperimen menunjukan bahwa jumlah n 50 memberikan akurasi terbaik.
RF memiliki kelebihan dalam menangani data yang kompleks, data yang tidak seimbang, dan tidak rentan terhadap overfitting. RF memiliki kekurangan yaitu dapat menjadi lambat untuk data yang besar, dan menghasilkan prediksi yang tidak konsisten.

### Boosting Algorithm
Boosting memiliki kelebihan yaitu dapat meningkatkan akurasi model pembelajaran mesin, dapat menangani data yang kompleks, dan tidak rentan terhadap overfitting. Boosting memiliki kekurangan yaitu dapat menjadi lambat untuk data yang besar dan dapat menghasilkan prediksi yang tidak konsisten.
Pada tahap ini, dilakukan proses berikut:
- Memilih algoritma boosting
- Memilih learning rate
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
Algoritma boosting yang dipilih adalah AdaBoost. Learning rate yang dipilih adalah 0,05.
Penjelasan di atas menjelaskan proses dan tahapan yang dilakukan pada tahap modeling. Penjelasan tersebut juga menjelaskan bagaimana algoritma tersebut bekerja pada data secara detail, termasuk parameter beserta nilai parameter di setiap algoritma yang diajukan.

Penggunaan tiga model prediksi diharapkan model KNN memberikan hasil terbaik karena dataset yang dimiliki merupakan data yang kompleks, dan data dataset yang dimiliki berupa data numerik dan kategorik.

## Evaluation
Pada evaluasi metrik yang digunakan pada model regresi merupakan metrik Mean Squared Error (MSE) hal ini dikarenakan MSE lebih mudah untuk dipahami dan diinterpretasikan. Akan tetapi, MSE sensitif terhadap outlier. 
MSE merupakan metrik evaluasi yang umum digunakan untuk mengukur kinerja mnodel regresi. MSE mengukur rata-rata kuadrat kesalahan antara prediksi model dan nilai sebenarnya.
MSE dipilih karena sesuai dengan tujuan dari pemodelan, yaitu untuk memprediksi nilai stroke dengan akurat. MSE mengukur seberapa dekat prediksi model dengan nilai sebenarnya. Semakin kecil nilai MSE, maka semakin akurat model dalam memprediksi nilai stroke.

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Pengukuran performa dari 3 Model dengan menghitung mean squared error (MSE)

Tabel 8. Hasil performa 3 model.
||train|test|
|-|-|-|
|KNN|0.14536|0.18468|
|RF|0.038683|0.198566|
|Boosting|0.170451|0.168157|

![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/f2b86993-9a52-4241-8329-c02eb2e1740f)

Gambar 10. Grafik batang hasil performa 3 model.

Dari model yang digunakan model K-Nearest Neighbor (KNN) memberikan nilai error yang paling kecil yaitu 26.1. Sedangkan model random forest memiliki error yang paling besar yaitu 30.8. Model KNN merupakan model terbaik untuk melakukan prediksi stroke.

Tabel 9. Pengukuran prediksi 3 model.
||y_true|prediksi_KNN|prediksi_RF|prediksi_Boosting|
|-|-|-|-|-|
|3653|14.0|26.1|30.8|30.6|

Berdasarkan hasil evaluasi, dapat disimpulkan bahwa model KNN merupakan model terbaik untuk melakukan prediksi stroke.
## Reference
Kaur, M., Sakhare, S. R., Wanjale, K., & Akter, F. (2022). Early stroke prediction methods for prevention of strokes. Behavioural Neurology, 2022.

Kustanti, C., & Widyarani, L. (2023). Evaluasi Efektivitas Metode FAST Flipbook dalam Meningkatkan Deteksi Dini Stroke: Studi Pendidikan Pra-Rumah Sakit di Indonesia. NERS Jurnal Keperawatan, 19(2), 68-75.

Dev, S., Wang, H., Nwosu, C. S., Jain, N., Veeravalli, B., & John, D. (2022). A predictive analytics approach for stroke prediction using machine learning and neural networks. Healthcare Analytics, 2, 100032.

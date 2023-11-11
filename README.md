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
Import Library yang dibutuhkan
````
import zipfile, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
````
Kemudian, memasukan dataset
````
local_zip = '/content/Stroke Prediction Dataset.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content')
zip_ref.close()
stroke = pd.read_csv('healthcare-dataset-stroke-data.csv')
len(stroke)
stroke.info()
````
![image](https://github.com/Roshied/Dicoding-ML/assets/68040731/a2ab966c-f9b2-42bc-bdf3-738eafeaa6ba)
- Terdapat 5110 baris (recods atau jumlah pengamatan) dalam dataset.
- Terdapat 11 kolom yaitu : id, gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status, stroke.
  
Pada dataset ini dilakukan exploratory data analysis untuk mengetahui lebih lanjut data-data yang nantinya akan digunakan
### Variabel-variabel pada Stroke Prediction Dataset
Deskripsi Variabel
  Deskripsi variabel dilakukan untuk mengetahu jenis-jenis variabel dan nilai-nilai dari variabel tersebut. Data tersebut kemudian dapat membantu memahami pengelolaan data yang akan dipersiapkan.
````
stroke.head()
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/e25cf203-101c-4f67-ac71-96c48d5ff1bb)
````
stroke.describe()
````
![image](https://github.com/Roshied/Dicoding-ML/assets/68040731/30fc8c45-af35-444e-ac01-e3fe99b7a6b3)

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
````
stroke['bmi'] = stroke['bmi'].replace(np.NaN, 0)
y = (stroke.avg_glucose_level == 0).sum()
z = (stroke.bmi == 0).sum()

print("Nilai 0 di kolom avg_glucose_level: ", y)
print("Nilai 0 di kolom bmi: ", z)

stroke = stroke.loc[(stroke['bmi'] != 0)]
stroke
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/d7847326-e9a3-4b9e-a307-6861585607af)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/3e13b31f-9eda-42a0-bc4a-a140f1583ae8)

- Outliers
  Kemudian sampel-sample yang memiliki nilai yang sangat jauh dari cakupan umum juga dikeluarkan dari dataset.

### Fitur age
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/c307b0e0-c1df-49a9-9d86-971ba75add43)

### Fitur avg_glucose_level
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/4fb5a353-3aad-4620-a6e7-5fe6d419966e)

### Fitur bmi
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/842d8d14-1c5e-49ef-818f-d2402cc89464)

### Membuat batas bawah dan atas
````
Q1 = stroke.quantile(0.25)
Q3 = stroke.quantile(0.75)
IQR=Q3-Q1
stroke=stroke[~((stroke<(Q1-1.5*IQR))|(stroke>(Q3+1.5*IQR))).any(axis=1)]

stroke.shape
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/30b10fa1-c615-4633-8223-4df9608de0c3)

## Multivariate Analysis
### Categorical Features
Pada tahap ini, pengecekan rata-rata umur terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap umur.
````
cat_features = stroke.select_dtypes(include='object').columns.to_list()

for col in cat_features:
  sns.catplot(x=col, y="age", kind="bar", dodge=False, height = 4, aspect = 3,  data=stroke, palette="Set3")
  plt.title("Rata-rata 'age' Relatif terhadap - {}".format(col))
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/921f0bc8-6e6e-4a3d-acec-9859fde62848)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/6a9774b9-92ac-46a6-8d02-b142502f2ff5)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/7d698ad6-5e47-404a-b732-0cc91171cd0a)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/9a69c0a6-a709-419f-a095-e9893bca5fc8)
- Pada fitur gender umur cederung memiliki jumlah data yang hampir sama pada rentang 35-40.
- Pada fitur ever_married terlihat perbedaan yaitu pada umur 20-50 telah menikah.
- Pada fitur residence_type terbagi rata pada rural dan urban.

### Numerical Features
Untuk mengamati hubungan antara fitur numerik menggunakan fungsi fairplot().
````
sns.pairplot(stroke, diag_kind = 'kde')
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/5835592b-d743-4b9f-9306-9038403af442)
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/01bf5be2-3ec5-4afb-abd4-e86c5fa2aac1)

### Correlation Matrix
````
plt.figure(figsize=(10, 8))
correlation_matrix = stroke.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/e1e92353-08ff-4f31-bc33-796d7e2d40ea)

## Data Preparation
Pada dataset ini dilakukan tiga tahap persiapan data, yaitu:
### Encoding fitur kategori 
Terdapat 5 data kategori yaitu 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status' yang akan dilakukan proses encoding menggunakan teknik _one-hot-encoding_. Teknik ini memberikan fitur baru yang sesuai sehingga dapat mewakili variabel kategori. Metode ini dilakukan dengan fungsi pd.get_dummies()
Fitur kategori tidak dapat langsung digunakan dalam model predictive analytics. Oleh karena itu, perlu dilakukan proses encoding untuk mengubah fitur kategori menjadi numerik.
Alasan dilakukannya encoding fitur kategori pada dataset ini adalah:
- Model predictive analytics umumnya lebih mudah bekerja dengan fitur numerik daripada fitur kategori.
- Encoding fitur kategori dapat membantu model untuk membedakan antar kategori.
````
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in categorical_columns:
    dummies = pd.get_dummies(stroke[column], prefix=column)
    stroke = pd.concat([stroke, dummies], axis=1)
    stroke = stroke.drop(columns=column)


stroke['bmi'] = stroke['bmi'].replace(np.NaN, 0)
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/6b078403-eff5-472d-a926-74ee322b1d79)

### Train-Test-Split
Pembagian dataset dengan fungsi train_test_split dari library sklearn
Dataset dibagi menjadi data train dan data test untuk menguji seberapa baik generalisasi model terhadap data baru. Pembagian yang dilakukan yaitu 80:20. Metode ini menggunakan fungsi sklearn.model_selection.train_test_split()
Alasan dilakukannya train-test-split pada dataset ini adalah:
- Data train digunakan untuk membangun model.
- Data test digunakan untuk mengevaluasi model.
- Pembagian 80:20 merupakan pembagian yang umum digunakan dalam predictive analytics.
````
from sklearn.model_selection import train_test_split

x = stroke.drop(["age"], axis=1)
y = stroke["age"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123)
````
````
print(f'Total # of sample in whole dataset: {len(x)}')
print(f'Total # of sample in train dataset: {len(x_train)}')
print(f'Total # of sample in test dataset: {len(x_test)}')
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/61d3a83e-e1f3-4d7a-a914-fa7a3fe46471)

### Standarisasi
Standardisasi merupakan teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan dengan menggunakan fungsi sklearn.preprocessing.StandardScaler()
Alasan dilakukannya standarisasi pada dataset ini adalah:
- Standarisasi dapat membantu model untuk lebih cepat konvergen.
- Standarisasi dapat membantu model untuk lebih akurat.
````
from sklearn.preprocessing import StandardScaler
numerical_features = ['avg_glucose_level', 'bmi']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/6a1566f7-44f1-42dd-96d8-1672b8d75fd4)
````
x_train[numerical_features].describe().round(4)
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/8db29635-e747-4e65-bdea-422c5e6074df)

## Modeling
Pada Tahapan modeling digunakan tiga algoritma yaitu :
### K-Nearest Neighbor
KNN adalah algoritma pembelajaran mesin yang bekerja dengan cara mencari k tetangga terdekat dari titik data yang akan diprediksi. Titik data yang paling dekat dengan titik data yang akan diprediksi akan memiliki nilai prediksi yang paling mungkin.
KNN memiliki kelebihan mudah diimplementasikan, tidak memerlukan pra-pemrosessan data yang kompleks, dan dapat menangani data numerik dan kategorikal. KNN memiliki kekurangan yaitu sensitif terhadap noise dan menjadi lambat untuk data yang besar.
Pada tahap ini, dilakukan proses berikut:
- Memilih nilai k
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
````
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(x_train), y_true=y_train)
````
Nilai k yang dipilih adalah 10. Nilai k ini dipilih berdasarkan eksperimen yang dilakukan. Hasil eksperimen menunjukkan bahwa nilai k 10 memberikan akurasi terbaik.
### Random Forest
RF adalah algoritma pembelajaran mesin yang bekerja dengan cara membangun sejumlah pohon keputusan secara acak. Hasil prediksi dari sejumlah pohon keputusan tersebut kemudian digabungkan untuk menghasilkan prediksi akhir.
Pada tahap ini, dilakukan proses berikut:
- Memilih jumlah pohon
- Memilih metode bootstrap
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
````
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)
````
Jumlah n yang dipilih adalah 50. Jumlah ini dipilih berdasarkan eksperimen yang dilakukan. Hasil ekseperimen menunjukan bahwa jumlah n 50 memberikan akurasi terbaik.
RF memiliki kelebihan dalam menangani data yang kompleks, data yang tidak seimbang, dan tidak rentan terhadap overfitting. RF memiliki kekurangan yaitu dapat menjadi lambat untuk data yang besar, dan menghasilkan prediksi yang tidak konsisten.

### Boosting Algorithm
Boosting memiliki kelebihan yaitu dapat meningkatkan akurasi model pembelajaran mesin, dapat menangani data yang kompleks, dan tidak rentan terhadap overfitting. Boosting memiliki kekurangan yaitu dapat menjadi lambat untuk data yang besar dan dapat menghasilkan prediksi yang tidak konsisten.
Pada tahap ini, dilakukan proses berikut:
- Memilih algoritma boosting
- Memilih learning rate
- Membangun model dengan menggunakan data train
- Melakukan prediksi pada data test
````
from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(x_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)
````
Algoritma boosting yang dipilih adalah AdaBoost. Learning rate yang dipilih adalah 0,05.
Penjelasan di atas menjelaskan proses dan tahapan yang dilakukan pada tahap modeling. Penjelasan tersebut juga menjelaskan bagaimana algoritma tersebut bekerja pada data secara detail, termasuk parameter beserta nilai parameter di setiap algoritma yang diajukan.

Penggunaan tiga model prediksi diharapkan model KNN memberikan hasil terbaik karena dataset yang dimiliki merupakan data yang kompleks, dan data dataset yang dimiliki berupa data numerik dan kategorik.

## Evaluation
Pada evaluasi metrik yang digunakan pada model regresi merupakan metrik Mean Squared Error (MSE) hal ini dikarenakan MSE lebih mudah untuk dipahami dan diinterpretasikan. Akan tetapi, MSE sensitif terhadap outlier. 
MSE merupakan metrik evaluasi yang umum digunakan untuk mengukur kinerja mnodel regresi. MSE mengukur rata-rata kuadrat kesalahan antara prediksi model dan nilai sebenarnya.
MSE dipilih karena sesuai dengan tujuan dari pemodelan, yaitu untuk memprediksi nilai stroke dengan akurat. MSE mengukur seberapa dekat prediksi model dengan nilai sebenarnya. Semakin kecil nilai MSE, maka semakin akurat model dalam memprediksi nilai stroke.

![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/5df18aa8-c44b-4467-9695-cd99be139ce4)
````
x_test.loc[:, numerical_features] = scaler.transform(x_test[numerical_features])
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(x_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(x_test))/1e3

mse
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/2827d1b7-f1f0-41b9-b972-8260db3f1520)
````
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/f2b86993-9a52-4241-8329-c02eb2e1740f)
````
prediksi = x_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
````
![image](https://github.com/Roshied/Stroke-Prediction-Data-Analytics/assets/68040731/46a1aba0-8a7b-4464-a3a4-0f3cf51bdd0f)

Dari model yang digunakan model K-Nearest Neighbor (KNN) memberikan nilai error yang paling kecil yaitu 26.1. Sedangkan model random forest memiliki error yang paling besar yaitu 30.8. Model KNN merupakan model terbaik untuk melakukan prediksi stroke.

Berdasarkan hasil evaluasi, dapat disimpulkan bahwa model KNN merupakan model terbaik untuk melakukan prediksi stroke.
## Reference
Kaur, M., Sakhare, S. R., Wanjale, K., & Akter, F. (2022). Early stroke prediction methods for prevention of strokes. Behavioural Neurology, 2022.

Kustanti, C., & Widyarani, L. (2023). Evaluasi Efektivitas Metode FAST Flipbook dalam Meningkatkan Deteksi Dini Stroke: Studi Pendidikan Pra-Rumah Sakit di Indonesia. NERS Jurnal Keperawatan, 19(2), 68-75.

Dev, S., Wang, H., Nwosu, C. S., Jain, N., Veeravalli, B., & John, D. (2022). A predictive analytics approach for stroke prediction using machine learning and neural networks. Healthcare Analytics, 2, 100032.

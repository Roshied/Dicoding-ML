# Laporan Proyek Machine Learning
## Domain Proyek
Stroke merupakan salah satu masalah kesehatan yang serius dengan dampak yang signifikan pada masyarakat. Penyebab dari stroke adalah gangguan pada suplai darah pada beberapa bagian dari otak. Stroke merupakan kondisi kesehatan yang membahayakan sehingga diperlukannya tindakan cepat untuk mencegah kondisi stroke yang lebih parah (Kaur, 2022).  

Penanganan stroke dipengaruhi oleh kemampuan mendeteksi dini serta intervensi yang cepat, Hal ini harus dilakukan dalam waktu periode golden time, yaitu dalam waktu kurang dari 3,5 jam setelah munculnya gejala stroke (Kustanti & Widyarani, 2023). Stroke dapat diketahui dengan tanda fisik dengan gejala awal seperti adanya wajah yang terkulai (facial droop), lengan yang sulit untuk digerakkan, dan kesulitan untuk berbicara.
 
Dengan perkembangannya teknologi, stroke dapat di deteksi dengan penggunaan machine learning. Penggunaan rekam medis dari pasien didapatkan prediksi kemungkinan terjadinya stroke (Dev et. al., 2022).

## Business Understanding
### Problem Statement
1. Bagaimana cara untuk memprediksi kemungkinan stroke ?
2. Data apa saja yang diperlukan untuk memprediksi kemungkinan stroke?
3. 
### Goals
1. Mengetahui cara untuk memprediksi kemungkinan stroke.
2. Mengetahui data yang diperlukan untuk memprediksi kemungkinan stroke.
3. 
- Solution Statement
## Data Understanding
Pada repository ini digunakan dataset [Stroke Prediction Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). Dataset ini memeliki 5.510 sampel yang mengalami stroke dan tidak dengan berbagai data. Data tersebut merupakan data non-numerik seperti_ work type, residence type, dan smoking status_. Serta fitur numerik seperti _age, body mass index, dan average glucose level_.
### Variabel-variabel pada Stroke Prediction Dataset
- Gender : merupakan jenis kelamin dari pasien
- Age : merupakan umur atau usia dari pasien
- Hypertension : merupakan pengidap hipertensi dari pasien 
- Heart Disease : merupakan pengidap penyakit jantung dari pasien
- Ever Married : merupakan kondisi pasien telah menikah
- Work Type : merupakan jenis pekerjaan yang dilakukan pasien
- Residence Type : merupakan jenis tempat tinggal pasien
- Average Glucose Level : merupakan nilai rata-rata level gula darah pasien
- Body Mass Index : merupakan nilai berat badan ideal dari pasien

Pada dataset ini dilakukan exploratory data analysis untuk mengetahui lebih lanjut data-data yang nantinya akan digunakan

## Data Preparation
Pada dataset ini dilakukan empat tahap persiapan data, yaitu:
- Encoding fitur kategori 
Terdapat 5 data kategori yaitu 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status' yang akan dilakukan proses encoding menggunakan teknik _one-hot-encoding_. Teknik ini memberikan fitur baru yang sesuai sehingga dapat mewakili variabel kategori.
````
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for column in categorical_columns:
    dummies = pd.get_dummies(stroke[column], prefix=column)
    stroke = pd.concat([stroke, dummies], axis=1)
    stroke = stroke.drop(columns=column)


stroke['bmi'] = stroke['bmi'].replace(np.NaN, 0)
````
- Pembagian dataset dengan fungsi train_test_split dari library sklearn
Dataset dibagi menjadi data train dan data test untuk menguji seberapa baik generalisasi model terhadap data baru.
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
- Standarisasi
````
from sklearn.preprocessing import StandardScaler
numerical_features = ['avg_glucose_level', 'bmi']
scaler = StandardScaler()
scaler.fit(x_train[numerical_features])
x_train[numerical_features] = scaler.transform(x_train.loc[:, numerical_features])
x_train[numerical_features].head()
````
````
x_train[numerical_features].describe().round(4)
````

## Modeling
Pada Tahapan modeling digunakan tiga algoritma yaitu :
- K-Nearest Neighbor
````
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(x_train), y_true=y_train)
````

- Random Forest
````
from sklearn.ensemble import RandomForestRegressor

RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(x_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(x_train), y_true=y_train)
````

- Boosting Algorithm
````
from sklearn.ensemble import AdaBoostRegressor

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(x_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(x_train), y_true=y_train)
````
  

## Evaluation
## Bibliography
Kaur, M., Sakhare, S. R., Wanjale, K., & Akter, F. (2022). Early stroke prediction methods for prevention of strokes. Behavioural Neurology, 2022.

Kustanti, C., & Widyarani, L. (2023). Evaluasi Efektivitas Metode FAST Flipbook dalam Meningkatkan Deteksi Dini Stroke: Studi Pendidikan Pra-Rumah Sakit di Indonesia. NERS Jurnal Keperawatan, 19(2), 68-75.

Dev, S., Wang, H., Nwosu, C. S., Jain, N., Veeravalli, B., & John, D. (2022). A predictive analytics approach for stroke prediction using machine learning and neural networks. Healthcare Analytics, 2, 100032.

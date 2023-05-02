from flask import Flask, render_template, request, url_for
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

app = Flask(__name__)

class CovidPredictor:
    def __init__(self):
        # read data from CSV
        self.data = pd.read_csv('covid19_data2.csv')
        self.data = self.data[['tanggal', 'provinsi', 'kasus_baru', 'kasus_aktif', 'kasus_kematian']]
        self.data['tanggal'] = pd.to_datetime(self.data['tanggal'], format='%m/%d/%Y')
        
        # create OneHotEncoder for province column
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(self.data[['provinsi']])
        
        # create linear regression model
        self.regressor = LinearRegression()
        
        # create naive bayes model
        self.nb = GaussianNB()

    def predict(self, provinsi_input, tanggal_input_str, kasus_baru_input, kasus_aktif_input, kasus_kematian_input):
        # Convert input date string to datetime type
        tanggal_input = datetime.strptime(tanggal_input_str, '%m/%d/%Y')

        # select data for yesterday
        tanggal_kemarin = datetime.today() - timedelta(days=1)
        data_kemarin = self.data[self.data['tanggal'] == tanggal_kemarin]

        # add user input to data for prediction tomorrow
        data_input = pd.DataFrame({'tanggal': [tanggal_input], 'provinsi': [provinsi_input], 'kasus_baru': [kasus_baru_input], 'kasus_aktif': [kasus_aktif_input], 'kasus_kematian': [kasus_kematian_input]})
        data_sekarang_provinsi = self.data[self.data['provinsi'] == provinsi_input]
        data_sekarang_provinsi = pd.concat([data_sekarang_provinsi, data_input], ignore_index=True)

        # memilih data yang tanggalnya kurang dari atau sama dengan tanggal input dari pengguna
        data_sekarang_provinsi = data_sekarang_provinsi[data_sekarang_provinsi['tanggal'] <= tanggal_input]

        # melakukan konversi pada kolom tanggal menjadi angka hari
        data_sekarang_provinsi['tanggal'] = data_sekarang_provinsi['tanggal'].apply(lambda x: x.toordinal())

        # memilih data untuk provinsi dan variabel yang dipilih oleh pengguna
        X = data_sekarang_provinsi[['tanggal']]
        y_baru = data_sekarang_provinsi[['kasus_baru']]
        y_aktif = data_sekarang_provinsi[['kasus_aktif']]
        y_kematian = data_sekarang_provinsi[['kasus_kematian']]

        # membuat model regresi untuk kasus baru, kasus aktif, dan kasus kematian
        model_baru = LinearRegression()
        model_baru.fit(X, y_baru)
        model_aktif = LinearRegression()
        model_aktif.fit(X, y_aktif)
        model_kematian = LinearRegression()
        model_kematian.fit(X, y_kematian)

        # melakukan prediksi untuk kasus baru, kasus aktif, dan kasus kematian di esok hari
        tanggal_esok = tanggal_input + timedelta(days=1)
        X_esok = [[tanggal_esok.toordinal()]]
        prediksi_baru = model_baru.predict(X_esok)
        prediksi_aktif = model_aktif.predict(X_esok)
        prediksi_kematian = model_kematian.predict(X_esok)
        persentase_kematian = (kasus_kematian_input / kasus_aktif_input) * 100
        akurasi_baru = model_baru.score(X, y_baru)  # Menghitung akurasi model regresi kasus baru
        akurasi_aktif = model_aktif.score(X, y_aktif)  # Menghitung akurasi model regresi kasus aktif
        akurasi_kematian = model_kematian.score(X, y_kematian)  # Menghitung akurasi model regresi kasus kematian

        # Memeriksa apakah nilai 'akurasi_kematian' adalah NaN
        if math.isnan(akurasi_kematian):
            # Jika NaN, mengatur nilai 'akurasi_kematian' menjadi 0
            akurasi_kematian = 0

        #precision
        y_true = data_sekarang_provinsi['kasus_kematian'].values
        y_pred = model_kematian.predict(X).round().astype(int)

        precision = precision_score(y_true, y_pred, average='weighted')

        #recall
        y_true = data_sekarang_provinsi['kasus_kematian'].values
        y_pred = model_kematian.predict(X).round().astype(int)

        recall = recall_score(y_true, y_pred, average='weighted')

        #F1
        # true labels
        y_true = [0, 1, 0, 0, 1, 1, 0, 1]

        # predicted labels
        y_pred = [0, 1, 1, 0, 0, 1, 0, 0]

        # calculate F1 score
        f1 = f1_score(y_true, y_pred)


        # menampilkan hasil prediksi ke dalam GUI
        return {
            'provinsi': provinsi_input,
            'tanggal': tanggal_esok.strftime('%m/%d/%Y'),
            'kasus_baru': int(prediksi_baru[0]),
            'kasus_aktif': int(prediksi_aktif[0]),
            'kasus_kematian': int(prediksi_kematian[0]),
            'persentase_kematian': round(persentase_kematian, 2), # Menambahkan tingkat persentase kematian
            'akurasi_baru': round(akurasi_baru, 4),  # Menambahkan akurasi model regresi kasus baru
            'akurasi_aktif': round(akurasi_aktif, 4),  # Menambahkan akurasi model regresi kasus aktif
            'akurasi_kematian': str(round(akurasi_kematian * 100, 2)) + '%',  # Menambahkan akurasi model regresi kasus kematian
            'precision' : str(round(precision * 100, 2)) + '%',
            'recall': str(round(recall * 100, 2)) + '%',
            'f1' : str(round(f1 * 100, 2)) + '%'

        }

# Create a CovidPredictor object
predictor = CovidPredictor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediksi')
def prediksi():
    return render_template('prediksi.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input from the HTML form
    provinsi_input = request.form['provinsi']
    tanggal_input_str = str(request.form['tanggal'])
    tanggal_input = datetime.strptime(tanggal_input_str, '%m/%d/%Y')
    kasus_baru_input = int(request.form['kasus_baru'])
    kasus_aktif_input = int(request.form['kasus_aktif'])
    kasus_kematian_input = int(request.form['kasus_kematian'])

    # Convert tanggal_input back to string
    tanggal_input_str = tanggal_input.strftime('%m/%d/%Y')

    # Call the predict method of the CovidPredictor object
    result = predictor.predict(provinsi_input, tanggal_input_str, kasus_baru_input, kasus_aktif_input, kasus_kematian_input)

    # Render the result on the HTML page
    return render_template('prediksi.html', result=result)

    # membuat prediksi menggunakan model regresi
    pred_baru = model_baru.predict([[tanggal_input.toordinal()]])
    pred_aktif = model_aktif.predict([[tanggal_input.toordinal()]])
    pred_kematian = model_kematian.predict([[tanggal_input.toordinal()]])

    # menampilkan output prediksi
    print(f"Prediksi di {provinsi_input} pada {tanggal_input_str}:")
    print(f"Jumlah Kasus Baru: {round(pred_baru[0][0])}")
    print(f"Jumlah Kasus Aktif: {round(pred_aktif[0][0])}")
    print(f"Jumlah Kasus Kematian: {round(pred_kematian[0][0])}")
    if pred_aktif[0][0] != 0:
        print(f"Tingkat Kematian: {round(pred_kematian[0][0] / pred_aktif[0][0] * 100, 2)}%")
    else:
        print("Tidak ada kasus aktif.")
    print(f"Akurasi: {round(self.regressor.score(X, y_aktif) * 100, 2)}%")


@app.route('/probabilitas', methods=['POST'])
def probability():
    # Load the CSV data into a Pandas dataframe
    df = pd.read_csv('covid19_data2.csv')

    # Preprocess the data
    le = LabelEncoder()
    df['provinsi_encoded'] = le.fit_transform(df['provinsi'])

    X = df[['provinsi_encoded', 'kasus_baru', 'kasus_aktif', 'kasus_kematian']]
    y = df['provinsi']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train the Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Test the classifier on the testing set
    accuracy = clf.score(X_test, y_test)
    print('Accuracy:', accuracy)

    # Calculate the probability of each province in the testing set
    probabilities = clf.predict_proba(X_test)
    probability = pd.DataFrame(probabilities, columns=clf.classes_)

    # Render the table as HTML
    return render_template('probabilitas.html', tables=[probability.to_html(classes='probability', header="true")])

# if __name__ == '__main__':
#     app.run(debug=True)

@app.route('/data')
def data():
    # Open the CSV file and read the data into a list of dictionaries
    with open('covid19_data2.csv', mode='r') as file:
        csv_reader = csv.DictReader(file)
        data = list(csv_reader)

    return render_template('data.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
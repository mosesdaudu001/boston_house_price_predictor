import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

df_1 = pd.read_csv('boston_data.csv')
scaler = StandardScaler()
scaler.fit(df_1.iloc[:, :12])


app = Flask(__name__)
model = joblib.load(open('rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    final_data = scaler.transform(final_features)
    prediction = model.predict(final_data)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='House Price is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
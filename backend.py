from flask import Flask, render_template, request, jsonify
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geopy.distance import geodesic
import joblib


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    data = request.json
    print(data)
    if isinstance(data, dict):
        data = [data]  
    
    
    
    df = pd.DataFrame(data)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['Month'] = df['Timestamp'].dt.month
 
    selected_features = ['Transaction_Amount', 'Amount_paid', 'Geographical_Location', 'Month']

 
    new_data = df[selected_features]


     
    model = joblib.load('model_FasttagFraudDetection.pkl')

    
    scaler = joblib.load('scaler.pkl')   
 
 
    reference_point = (13.059816123454882, 77.77068662374292)
    new_data['distance_from_city_center'] = new_data['Geographical_Location'].apply(
        lambda x: geodesic(reference_point, tuple(map(float, x.split(',')))).kilometers
    )

     
    new_data[['Transaction_Amount', 'Amount_paid']] = scaler.transform(
        new_data[['Transaction_Amount', 'Amount_paid']]
    )
 
    new_data = new_data[['Transaction_Amount', 'Amount_paid', 'Month', 'distance_from_city_center']]

     
    new_data.fillna(new_data.mean(), inplace=True)

     
    predictionbinary = model.predict(new_data)

    if predictionbinary == 0:
            prediction = "fraud"
    else:
            prediction = " not fraud"
    print(prediction)
    
     
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug=True)

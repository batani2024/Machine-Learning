from tensorflow import keras
import numpy as np
import joblib
from flask import Flask, jsonify, request
import plotly.graph_objects as go
import pandas as pd
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

tanaman_data = {
    'anggur': {
        'data': 'grape_avg_price.xlsx',
        'scaler': 'scaler_grape.pkl',
        'model':'best_grape_model.h5'
    },
    'apel': {
        'data': 'grouped_prices_apple.xlsx',
        'scaler': 'scaler_apple.pkl',
        'model':'best_apple_model1.h5'
    },
    'delima': {
        'data': 'pomegranate_avg_price.xlsx',
        'scaler': 'scaler_pomegranate.pkl',
        'model':'best_pomegranate_model.h5'
    },
    'goni': {
        'data': 'jute_avg_price.xlsx', #masalah
        'scaler': 'scaler_jute.pkl',
        'model':'best_jute_model.h5'
    },
    'jagung': {
        'data': 'maize_avg_price.xlsx', 
        'scaler': 'scaler_maize.pkl',
        'model':'best_maize_model.h5'
    },
    'jeruk': {
        'data': 'orange_avg_price.xlsx',
        'scaler': 'scaler_orange.pkl',
        'model':'best_orange_model.h5'
    },
    'kacang_lentil': {
        'data': 'lentil_avg_price.xlsx', #masalah
        'scaler': 'scaler_lentil.pkl',
        'model':'best_lentil_model.h5'
    },
    'kacang_lentil_hitam': {
        'data': 'grouped_prices_blackgram.xlsx', #masalah
        'scaler': 'scaler_blackgram.pkl',
        'model':'best_blackgram_model.h5'
    },
    'kapas': {
        'data': 'grouped_prices_cotton.xlsx',
        'scaler': 'scaler_cotton.pkl',
        'model':'best_cotton_model.h5'
    },
    'kelapa': {
        'data': 'grouped_prices_coconut.xlsx',
        'scaler': 'scaler_coconut.pkl',
        'model':'best_coconut_model.h5'
    },
    'kopi': {
        'data': 'grouped_prices_coffee.xlsx',
        'scaler': 'scaler_kopi.pkl',
        'model':'best_kopi_model.h5'
    },
    'mangga': {
        'data': 'mango_avg_price.xlsx',
        'scaler': 'scaler_mango.pkl',
        'model':'best_mango_model.h5'
    },
    'melon_musk': {
        'data': 'melonmusk_avg_price.xlsx', #masalah
        'scaler': 'scaler_melonmusk.pkl',
        'model':'best_melonmusk_model.h5'
    },
    'padi': {
        'data': 'rice_avg_price.xlsx', #masalah
        'scaler': 'scaler_rice.pkl',
        'model':'best_rice_model.h5'
    },
    'pepaya': {
        'data': 'papaya_avg_price.xlsx', #masalah
        'scaler': 'scaler_papaya.pkl',
        'model':'best_papaya_model.h5'
    },
    'pisang': {
        'data': 'grouped_prices_banana.xlsx',
        'scaler': 'scaler_banana.pkl',
        'model':'best_banana_model.h5'
    },
    'semangka': {
        'data': 'watermelon_avg_price.xlsx',
        'scaler': 'scaler_watermelon.pkl',
        'model':'best_watermelon_model.h5'
    }
}

model = None
scaler = None

def transform(x, scaler):
    data = scaler.transform(x.values)
    return data

def predict(data, days, model):
    forecast = []
    result_forecast = data[-days:].tolist()

    result_forecast = [item[0] if isinstance(item, list) else float(item) for item in result_forecast]

    for _ in range(days):
        input_data = np.array(result_forecast[-days:]).reshape(1, days, 1)
        prediksi = model.predict(input_data)[0][0]
        forecast.append(prediksi)
        result_forecast.append(prediksi)

    return forecast

def convert_result(initial_x, converted_x, forecast, days, scaler):
    if not isinstance(initial_x.index, pd.DatetimeIndex):
        initial_x.index = pd.to_datetime(initial_x.index)

    last_30_df = pd.DataFrame(
        scaler.inverse_transform(converted_x[-days:]),
        columns=[initial_x.columns[0]],
        index=initial_x.index[-days:]
    )

    new_dates = pd.date_range(start=last_30_df.index[-1] + pd.Timedelta(days=1), periods=days)

    forecast_df = pd.DataFrame(
        {'Modal Price (Rs./Quintal)': scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()},
        index=new_dates
    )

    extended_df = pd.concat([last_30_df, forecast_df])

    return extended_df, last_30_df

app = Flask(__name__)

@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    global model, scaler
    # Get the list of tanaman names and days to forecast from the query string
    tanaman_list = request.args.getlist('tanaman')
    days2forecast = request.args.get('days2forecast', type=int)

    # Ensure that at least one tanaman name is provided
    if not tanaman_list:
        return jsonify({'error': 'At least one tanaman name must be provided'}), 400
    
    results = []
    #return tanaman_list
    for tanaman in tanaman_list:
        # Ensure the tanaman name exists in the dictionary
        if tanaman not in tanaman_data:
            
            results.append({'error': f'Data {tanaman} tidak ada'})
            
            #return jsonify({'error': f'Tanaman name "{tanaman}" not found'}), 400
            #continue
        else:
            # Load the model and scaler for the selected tanaman
            model = keras.models.load_model(tanaman_data[tanaman]['model'], custom_objects={'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()})
            scaler = joblib.load(tanaman_data[tanaman]['scaler'])

            # Load the dataset from the provided URL
            file = tanaman_data[tanaman]['data']
        try:

            lastprice_df = pd.read_excel(file)
            lastprice_df = lastprice_df.set_index(lastprice_df.columns[0])
        except Exception as e:
            return jsonify({'error': f'Failed to download or read file for "{tanaman}": {e}'}), 400

        if 'Modal Price (Rs./Quintal)' not in lastprice_df.columns:
            return jsonify({'error': 'Excel file must contain a column named "Modal Price (Rs./Quintal)"'}), 400

        # Process the data and get the forecast
        initial_data = transform(lastprice_df, scaler)
        prediksi = predict(initial_data, days2forecast, model)
        true, predicted = convert_result(lastprice_df, initial_data, prediksi, days2forecast, scaler)

        # Generate the plot for the current tanaman
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=true.index,
            y=true['Modal Price (Rs./Quintal)'],
            mode='lines',
            name='True Price',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=predicted.index,
            y=predicted['Modal Price (Rs./Quintal)'],
            mode='lines',
            name='Predicted Price',
            line=dict(color='red')
        ))
        fig.update_layout(
            title=f'{tanaman} Modal Price Over Time',
            xaxis_title='Price Date',
            yaxis_title='Modal Price (Rs./Quintal)',
            legend_title='Legend',
            template='plotly_white'
        )
        fig_json = fig.to_json()

        # Append the plot to the results list
        results.append({'tanaman': tanaman, 'plot': fig_json})

    # Return all forecast plots in a list
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)

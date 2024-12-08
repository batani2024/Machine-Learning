from tensorflow import keras
import numpy as np
import joblib
from flask import Flask, jsonify, request
import plotly.graph_objects as go
import pandas as pd
import requests
from io import BytesIO
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError

tanaman_data = {
    'anggur': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSGOk1Dm2xn_GCIj1u8Wh_VBFmyEJQMuQlv_pP6HFFyHnnqkvipIcfhwrG8Yqkv4A/pub?output=xlsx',
        'scaler': 'scaler_grape.pkl',
        'model':'best_grape_model.h5'
    },
    'apel': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQ3XzG8gdELVbmSBVFPtU5LW7FitIBnCBYPQQRyxEJvKlPXpkb6H08cQXkZzohsJw/pub?output=xlsx',
        'scaler': 'scaler_apple.pkl',
        'model':'best_apple_model1.h5'
    },
    'delima': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSgy18YB4C6zt3hlRvVEpH6F6ObO1NxKa1V0INc3WDthfMfuVilzxTyDKQ-IzjlOg/pub?output=xlsx',
        'scaler': 'scaler_pomegranate.pkl',
        'model':'best_pomegranate_model.h5'
    },
    'goni': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTSwxoyeeukGDrc-375u0l-QDurCBCDYPcMC6UCT_xR-CuGQlYRd6IbWuL-PdF6IA/pub?output=xlsx',
        'scaler': 'scaler_jute.pkl',
        'model':'best_lentil_jute.h5'
    },
    'jagung': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRQKISrB0KM08IF9irqcGJBk2Xwd5_5N2-1yA7PSBgh0qorMDFaAXvzPDh_PIDaDQ/pub?output=xlsx',
        'scaler': 'scaler_maize.pkl',
        'model':'best_maize_model.h5'
    },
    'jeruk': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTr22wUpMdzdeXXW888ZyMi0ln0Sd8Yw5WuZzDTTQ1-XKpUGEi0vt0wERuNgPNdeA/pub?output=xlsx',
        'scaler': 'scaler_orange.pkl',
        'model':'best_orange_model.h5'
    },
    'kacang_lentil': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSnLvxw69gSvBWCOH0qPi7ZroSVJMCwky_psL4V9Rd4Dsg-3mzsELbC8g27Uhk2Lg/pub?output=xlsx',
        'scaler': 'scaler_lentil.pkl',
        'model':'best_lentil_model.h5'
    },
    'kacang_lentil_hitam': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR4UsWQnt1vZDLeXq8lDDSOXjUtgXw96haKFfkY_DJ-HRFI_JGiddyK7ssDUByt_w/pub?output=xlsx',
        'scaler': 'scaler_blackgram.pkl',
        'model':'best_blackgram_model.h5'
    },
    'kapas': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRTaDJ3xiEWZ0jxp-bOXUKfgERGweIAEP5xLaO4iJccuZGPu6G7l4MXoS8GZLi90g/pub?output=xlsx',
        'scaler': 'scaler_cotton.pkl',
        'model':'best_cotton_model.h5'
    },
    'kelapa': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRaSzRmZr5Oa_8JGeiBw6d03E4w5RlBzvgdNJfYl0fIvfP2RWMevHFPrlaAm1HmRQ/pub?output=xlsx',
        'scaler': 'scaler_coconut.pkl',
        'model':'best_coconut_model.h5'
    },
    'kopi': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vTmxWYDy6Wr4rCw7EJFG05lgWLG-giD30fXdrHDKtEc9sib5I96uUsb_ASSzHe1Mw/pub?output=xlsx',
        'scaler': 'scaler_kopi.pkl',
        'model':'best_kopi_model.h5'
    },
    'mangga': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQHpPhMrZQQONFW74AYoGe-POB8-Cic6NJ-LYYThV5e67uhBBxGGlHwvnTDxf66Dw/pub?output=xlsx',
        'scaler': 'scaler_mango.pkl',
        'model':'best_mango_model.h5'
    },
    'melon_musk': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRStYmZ2UydRN3LvI7QKedFu2O9kGoASNhWcNxeubj0oHUO9RRhNX3O8Zhupxt2PQ/pub?output=xlsx',
        'scaler': 'scaler_melonmusk.pkl',
        'model':'best_melonmusk_model.h5'
    },
    'padi': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRpIvYM-V8Zf_xwcfUFhrbq6Qo7PjvAysTX703ouc6QzbFAq6o5nqz-eRcPRx5HFw/pub?output=xlsx',
        'scaler': 'scaler_rice.pkl',
        'model':'best_rice_model.h5'
    },
    'pepaya': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vSgw9Bz8ywlTDrVSRlNEqECDeMvjIX6GQaEtPcOxZsJzdVElaYEuga7ZNUeaCkiIQ/pub?output=xlsx',
        'scaler': 'scaler_papaya.pkl',
        'model':'best_papaya_model.h5'
    },
    'pisang': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRohW9P2U40CCI_6s7qvpI1NTagc_PFNEE-yKah7Optu0TlW-KXaoWM2SAr5f9GdQ/pub?output=xlsx',
        'scaler': 'scaler_banana.pkl',
        'model':'best_banana_model.h5'
    },
    'semangka': {
        'data': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQRmadnz2HF7fl3aOISE8Oq-afIu-YKMsbehtCGt02xvK4L2A-MnZJ33iivlMWkMg/pub?output=xlsx',
        'scaler': 'scaler_watermelon.pkl',
        'model':'best_watermelon_model.h5'
    },
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
            #return jsonify({'error': f'Tanaman name "{tanaman}" not found'}), 400
            continue

        # Load the model and scaler for the selected tanaman
        model = keras.models.load_model(tanaman_data[tanaman]['model'], custom_objects={'mse': MeanSquaredError(), 'mae': MeanAbsoluteError()})
        scaler = joblib.load(tanaman_data[tanaman]['scaler'])

        # Load the dataset from the provided URL
        file_url = tanaman_data[tanaman]['data']
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            file = BytesIO(response.content)
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


from flask import Flask, render_template
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    # STEP 1: Load data
    df = pd.read_csv('Sample_Data.csv')
    df.rename(columns={'Values': 'Voltage'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

    # STEP 2: Moving average
    df['Moving_Avg'] = df['Voltage'].rolling(window=5).mean()

    # STEP 3: Plot voltage and moving average
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Timestamp'], df['Voltage'], label='Voltage', alpha=0.6)
    ax.plot(df['Timestamp'], df['Moving_Avg'], label='5-point Moving Avg', linewidth=2)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('Voltage')
    ax.set_title('Voltage over Time with Moving Average')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # STEP 4: Find peaks and valleys
    peaks, _ = find_peaks(df['Voltage'])
    valleys, _ = find_peaks(-df['Voltage'])
    peak_df = df.iloc[peaks][['Timestamp', 'Voltage']].rename(columns={'Voltage': 'Peak Voltage'}).head()
    valley_df = df.iloc[valleys][['Timestamp', 'Voltage']].rename(columns={'Voltage': 'Valley Voltage'}).head()

    # STEP 5: Voltage below 20
    low_voltage_df = df[df['Voltage'] < 20][['Timestamp', 'Voltage']]

    # STEP 6: Accelerated downward slopes
    df['First_Derivative'] = df['Voltage'].diff()
    df['Second_Derivative'] = df['First_Derivative'].diff()
    accel_down = df[(df['First_Derivative'] < 0) & (df['Second_Derivative'] < 0)]
    accel_down_timestamps = accel_down['Timestamp']

    # Convert tables to HTML for display
    peak_html = peak_df.to_html(index=False)
    valley_html = valley_df.to_html(index=False)
    low_html = low_voltage_df.to_html(index=False)
    accel_html = accel_down_timestamps.to_frame(name="Timestamp").to_html(index=False)

    return render_template('index.html', plot_url=plot_url, peak_html=peak_html,
                           valley_html=valley_html, low_html=low_html, accel_html=accel_html)

if __name__ == '__main__':
    app.run(debug=True)

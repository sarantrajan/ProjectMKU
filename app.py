from flask import Flask, render_template, request, redirect, url_for, flash
import requests
import pickle
import numpy as np
import mysql.connector
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'b9417269bf520c0ea8b9523d17c34586'

# Load trained model and season encoder
model = pickle.load(open('crop_model.pkl', 'rb'))
label_encoder = pickle.load(open('season_encoder.pkl', 'rb'))

API_KEY = '056423ddc0ce07a4fa420788baaf012d'

# Detect current season
def get_season():
    month = datetime.now().month
    if month in [3, 4, 5]:
        return "summer"
    elif month in [6, 7, 8]:
        return "rainy"
    elif month in [9, 10, 11]:
        return "spring"
    else:
        return "winter"

def get_ph_from_soilgrids(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lat={lat}&lon={lon}&property=phh2o&depth=0-5cm"
        headers = {"accept": "application/json"}
        response = requests.get(url, headers=headers)
        data = response.json()

        ph_value = data['properties']['phh2o']['mean'][0] / 10
        return round(ph_value, 2)
    except Exception as e:
        print("Soil pH fetch error:", e)
        return round(np.random.uniform(5.5, 7.5), 1)


def estimate_water_availability(temp, humidity):

    if humidity > 85:
        if temp < 20:
            return round(np.random.uniform(250, 270), 1)
        elif 20 <= temp < 25:
            return round(np.random.uniform(230, 250), 1)
        elif 25 <= temp < 30:
            return round(np.random.uniform(210, 230), 1)
        else:
            return round(np.random.uniform(190, 210), 1)

    elif 70 <= humidity <= 85:
        if temp < 20:
            return round(np.random.uniform(220, 240), 1)
        elif 20 <= temp < 25:
            return round(np.random.uniform(200, 220), 1)
        elif 25 <= temp < 30:
            return round(np.random.uniform(180, 200), 1)
        else:
            return round(np.random.uniform(160, 180), 1)

    elif 55 <= humidity < 70:
        if temp < 20:
            return round(np.random.uniform(180, 200), 1)
        elif 20 <= temp < 25:
            return round(np.random.uniform(160, 180), 1)
        elif 25 <= temp < 30:
            return round(np.random.uniform(140, 160), 1)
        else:
            return round(np.random.uniform(120, 140), 1)

    else:  # humidity < 55
        if temp < 20:
            return round(np.random.uniform(150, 170), 1)
        elif 20 <= temp < 25:
            return round(np.random.uniform(130, 150), 1)
        elif 25 <= temp < 30:
            return round(np.random.uniform(110, 130), 1)
        else:
            return round(np.random.uniform(90, 110), 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']

        # Connect to MySQL and store the message
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="crop_system"
        )
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO contact_messages (name, email, message) VALUES (%s, %s, %s)",
            (name, email, message)
        )
        conn.commit()
        cursor.close()
        conn.close()

        flash("Your message has been sent successfully!", "success")
        return redirect('/contact')

    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('prediction.html')

    lat = float(request.form['latitude'])
    lon = float(request.form['longitude'])

    if not lat or not lon:
        flash("Please select a location on the map!", "error")
        return redirect(url_for("prediction_page"))

    # Fetch temperature and humidity from OpenWeatherMap
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    response = requests.get(weather_url).json()
    temperature = response['main']['temp']
    humidity = response['main']['humidity']

    # Estimate other values
    ph = get_ph_from_soilgrids(lat, lon)
    water = estimate_water_availability(temperature, humidity)
    season_text = get_season()
    season = label_encoder.transform([season_text])[0]

    # Make prediction
    features = np.array([[temperature, humidity, ph, water, season]])
    prediction = model.predict(features)[0]

    # Save to MySQL
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="crop_system"
    )
    cursor = conn.cursor()
    cursor.execute(
    "INSERT INTO predictions (latitude, longitude, temperature, humidity, ph, water, season, predicted_crop) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
    (
        float(lat),
        float(lon),
        float(temperature),
        float(humidity),
        float(ph),
        float(water),
        season_text,
        str(prediction)
    )
)

    conn.commit()
    cursor.close()
    conn.close()

    return render_template("output.html", crop=prediction)

if __name__ == '__main__':
    app.run(debug=True)

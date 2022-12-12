from application import app
from flask import render_template, request, json, jsonify
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import requests
import numpy
import pandas as pd

#decorator to access the app
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")

#decorator to access the service
@app.route("/Acciclassify", methods=['GET', 'POST'])
def Acciclassify():

    #extract form inputs
    severity = request.form.get("severity")
    city = request.form.get("city")
    state = request.form.get("state")
    pressure = request.form.get("pressure")
    windspeed = request.form.get("windspeed")
    WeatherCondition = request.form.get("WeatherCondition")
    SunsetSunrise = request.form.get("SunsetSunrise")
    Humidity = request.form.get("Humidity")

   #convert data to json
    input_data = json.dumps({"severity": severity, "city": city, "state": state, "pressure": pressure, "windspeed": windspeed, "WeatherCondition": WeatherCondition,"SunsetSunrise":SunsetSunrise, "Humidity": Humidity})

    #url for accident predicting model
    url = "http://localhost:5000/api"
    
  
    #post data to url
    results =  requests.post(url, input_data)

    #send input values and prediction result to index.html for display
    return render_template("index.html", severity = severity, city = city, state = state, pressure = pressure, windspeed = windspeed, WeatherCondition = WeatherCondition, SunsetSunrise=SunsetSunrise, Humidity = Humidity,  results=results.content.decode('UTF-8'))
  
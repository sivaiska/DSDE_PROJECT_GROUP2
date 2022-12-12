from flask import Flask, request, Response, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

#load data
df = pd.read_csv("./data/US_Accidents.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

numeric_data = df.iloc[:, [1,2,3]].values
numeric_df = pd.DataFrame(numeric_data, dtype = object)
numeric_df.columns = ['severity', 'pressure','windspeed']


#standard scaling severity
severity_std_scale = StandardScaler()
numeric_df['severity'] = severity_std_scale.fit_transform(numeric_df[['severity']])

#standard scaling balance
balance_std_scale = StandardScaler()
numeric_df['pressure'] = balance_std_scale.fit_transform(numeric_df[['pressure']])

balance_std_scale = StandardScaler()
numeric_df['windspeed'] = balance_std_scale.fit_transform(numeric_df[['windspeed']])

X_categoric = df.iloc[:, [0,4,5,6,7]].values

#onehotencoding
ohe = OneHotEncoder()
categoric_data = ohe.fit_transform(X_categoric).toarray()
categoric_df = pd.DataFrame(categoric_data)
categoric_df.columns = ohe.get_feature_names()

#combine numeric and categorix
X_final = pd.concat([numeric_df, categoric_df], axis = 1)
#train model
rfc = RandomForestClassifier(n_estimators = 100)
rfc.fit(X_final, y)

#create flask instance
app = Flask(__name__)

#create api
@app.route('/api', methods=['GET', 'POST'])
def predict(): 
    #get the data from request
    data = request.get_json(force=True)
    data_categoric = np.array([data["city"], data["state"], data["WeatherCondition"], data["SunriseSunset"], data["Humidity"]])
    data_categoric = np.reshape(data_categoric, (1, -1))
    data_categoric = ohe.transform(data_categoric).toarray()
 
    data_severity = np.array([data["severity"]])
    data_severity = np.reshape(data_severity, (1, -1))
    data_severity = np.array(severity_std_scale.transform(data_severity))

    data_pressure = np.array([data["pressure"]])
    data_pressure= np.reshape(data_pressure, (1, -1))
    data_pressure = np.array(balance_std_scale.transform(data_pressure))

    data_windspeed = np.array([data["windspeed"]])
    data_windspeed= np.reshape(data_windspeed, (1, -1))
    data_windspeed = np.array(balance_std_scale.transform(data_windspeed))

    data_final = np.column_stack((data_severity, data_pressure,data_windspeed , data_categoric))
    data_final = pd.DataFrame(data_final, dtype=object)

    #make predicon using model
    prediction = rfc.predict(data_final)
    return Response(json.dumps(prediction[0]))

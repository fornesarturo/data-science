import googlemaps as gm
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from flask import Flask, render_template, request

app = Flask(__name__)

routes = {
    "Vallarta": [
        "Av. Ignacio L. Vallarta 6485, Granja, 45010 Zapopan, Jal.",
        "Av. Ignacio L. Vallarta 4845, Prados Vallarta, 45020 Zapopan, Jal."
    ],
    "Juarez": [
        "Av Juarez 30, Zona Centro, 44100 Guadalajara, Jal.",
        "Av Juarez 976, Col Americana, Americana, 44160 Guadalajara, Jal."
    ],
    "Hidalgo": [
        "Av. Miguel Hidalgo y Costilla 1540, Ladron de Guevara, Lafayette, 44600 Guadalajara, Jal.",
        "Av. Miguel Hidalgo y Costilla 400, Zona Centro, 44100 Guadalajara, Jal."
    ],
    "Federalismo": [
        "Calz del Federalismo Nte 781, Artesanos, 44200 Guadalajara, Jal.",
        "Calz. del Federalismo Sur 544, Mexicaltzingo, 44100 Guadalajara, Jal."
    ],
    "Alcalde": [
        "Av. Fray Antonio Alcalde 1011, Alcalde Barranquitas, 44270 Guadalajara, Jal.",
        "Av. Fray Antonio Alcalde 130, Centro, 44100 Guadalajara, Jal."
    ],
    "Enrique_Diaz_de_Leon": [
        "Av Enrique Díaz de León Nte 1743, Lagos del Country, 44210 Guadalajara, Jal.",
        "44160, Av Enrique Díaz de León Sur 132, Col Americana, Americana, 44160 Guadalajara, Jalisco"
    ],
    "Lopez_Mateos": [
        "2043, Av Adolfo López Mateos Nte, El mante, Vallarta Nte., 44690 Zapopan, Jal.",
        "Av. Adolfo López Mateos Sur 5946, Las Fuentes, 45070 Zapopan, Jal."
    ],
    "Avila_Camacho": [
        "Avenida Manuel Ávila Camacho 950, Conjunto Patria, 45160 Zapopan, Jal.",
        "Avenida Manuel Ávila Camacho 1015, La Normal, 44260 Guadalajara, Jal."
    ],
    "Independencia": [
        "Calz Independencia Sur 1085, Ferrocarril, 44460 Guadalajara, Jal.",
        "Calz Independencia Nte 2302, Colonia Belisario Domínguez, 44320 Guadalajara, Jal."
    ],
    "Lazaro_Cardenas": [
        "Calz. Lázaro Cárdenas 4249, Camino Real, 45040 Zapopan, Jal.",
        "Calz. Lázaro Cárdenas 1489, Computec Express, 44940 Guadalajara, Jal."
    ]
}

checkpoints = {
    "MUSA": "Av Juárez 975, Col Americana, Centro, 44100 Guadalajara, Jal.",
    "Hospicio Cabañas": "Calle Cabañas 8, Las Fresas, 44360 Guadalajara, Jal.",
    "Catedral de Guadalajara": "Av Alcalde 10, Zona Centro, 44100 Guadalajara, Jal.",
    "Plaza Galerías": "Av Rafael Sanzio 150, Camichines Vallarta, 45030 Zapopan, Jal.",
    "Plaza Patria": "Av. Patria, Plaza Patria, 45160 Zapopan, Jal.",
    "Andares": "Blvrd Puerta de Hierro 4965, 45116 Zapopan, Jal.",
    "Secretaría de Movilidad": "Avenida Prolongación Alcalde S/N, Jardines Alcalde, Santa Elena Alcalde, 44290 Guadalajara, Jal.",
    "La Minerva": "Av 8 de Julio 55, Zona Centro, 44100 Guadalajara, Jal.",
    "Centro Magno": "Av. Ignacio L. Vallarta 2425, Arcos Vallarta, 44130 Guadalajara, Jal.",
    "Basílica de Zapopan": "Calle Eva Briseño 152, Zapopan, 44250 Zapopan, Jal."
}

my_key = "AIzaSyBWN4K23yA7pPd1Qyw5hdxhyD9zTPtn2ws"
my_maps = gm.Client(my_key)

class TrafficFramer():
    def __init__(self, maps, checkpoints, routes):
        self.maps = maps
        self.checkpoints = checkpoints
        self.routes = routes
    
    def getTrafficDataframe(self):
        now = datetime.now()
        result_df = None

        for name, origin in self.checkpoints.items():
            destinations = {k: v for k, v in self.checkpoints.items() if k != name}
            for destination_name, destination in destinations.items():
                res = TrafficFramer.getTrafficData(
                        origin,
                        destination,
                        self.maps,
                        now,
                        name,
                        destination_name
                    )
                if result_df is None:
                    result_df = res
                else:
                    result_df = pd.merge(result_df, res, how="outer")
        self.df = result_df
        return result_df
    
    def getTrafficRoutesDataframe(self):
        now = datetime.now()
        result_df = None

        for name, route in self.routes.items():
            origin = route[0]
            destination = route[1]
            res = TrafficFramer.getTrafficRouteData(
                    origin,
                    destination,
                    self.maps,
                    now,
                    name
                )
            if result_df is None:
                result_df = res
            else:
                result_df = pd.merge(result_df, res, how="outer")
        self.df = result_df
        return result_df
    
    @staticmethod
    def getTrafficData(origin, destination, maps, time, origin_name, destination_name):
        res = maps.distance_matrix(
                origin,
                destination,
                mode="driving",
                departure_time=time,
                traffic_model="pessimistic"
            )
        if res and res is not None:
            return pd.DataFrame({
                "origin": [origin_name],
                "destination": [destination_name],
                "time": [time],
                "duration": [res["rows"][0]["elements"][0]["duration"]["value"] / 60],
                "duration_with_traffic": [res["rows"][0]["elements"][0]["duration_in_traffic"]["value"] / 60]
            })
        else:
            return None
    
    @staticmethod
    def getTrafficRouteData(origin, destination, maps, time, name):
        res = maps.distance_matrix(
                origin,
                destination,
                mode="driving",
                departure_time=time,
                traffic_model="pessimistic"
            )
        if res and res is not None:
            return pd.DataFrame({
                "name": [name],
                "time": [time],
                "duration": [res["rows"][0]["elements"][0]["duration"]["value"] / 60],
                "duration_with_traffic": [res["rows"][0]["elements"][0]["duration_in_traffic"]["value"] / 60],
                "diff": (res["rows"][0]["elements"][0]["duration_in_traffic"]["value"] / 60) - (res["rows"][0]["elements"][0]["duration"]["value"] / 60)
            })
        else:
            return None

global model, sample_entry
def trainModel():
    df_routes = pd.read_csv('route_categories.csv', index_col=0)
    df_routes.time = pd.to_datetime(df_routes.time, format='%Y-%m-%d')
    df_traffic_dummies = pd.get_dummies(df_routes.iloc[:,2:14])
    X = df_traffic_dummies
    y = df_routes[df_routes.columns[14:15]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestClassifier(criterion='entropy', n_estimators=1000)
    model.fit(X_train, y_train.values.ravel())
    return (model, X.head(1))

global grouped_data
def get_traffic_model():
    def get_filenames():
        filenames = []
        for _, _, files in os.walk("./routes"):
            for file in files:
                if file.endswith(".csv"):
                    filenames.append(file)
        return filenames
    csv_filenames = get_filenames()

    def to_time(datetime_string):
        d = datetime.strptime(datetime_string, "%Y-%m-%d %H:%M:%S.%f")+timedelta(hours=-5)
        return d.strftime("%H")

    def to_mexican(dt_string):
        d = datetime.strptime(dt_string, "%Y-%m-%d %H:%M:%S.%f")+timedelta(hours=-5)
        return d

    complete_data = None
    for i, filename in enumerate(csv_filenames):
        data = pd.read_csv(f"./routes/{filename}")
        data["hour"] = data["time"].apply(to_time)
        data["time"] = data["time"].apply(to_mexican)
        
        if(int(data['hour'][0]) > 21 or int(data['hour'][0]) < 5):
            continue
        if complete_data is None:
            complete_data = data
        else:
            complete_data = pd.merge(complete_data, data, how='outer')
    complete_data = complete_data.drop(['diff'], axis=1)
    complete_data['traffic_proportion'] = complete_data['duration_with_traffic']/complete_data['duration']

    def q1(x):
        return x.quantile(0.25)
    def q3(x):
        return x.quantile(0.75)
    def std_shift(x):
        return x.mean() + (0.75*x.std())
    f = {'traffic_proportion': ['mean', 'median', 'std', q1, q3, std_shift]}
    global grouped_data
    grouped_data = complete_data.groupby('name').agg(f)
    return grouped_data

def get_current_route_data(origin, destination):
    traffic = TrafficFramer(my_maps, checkpoints, routes)
    result_df = traffic.getTrafficRoutesDataframe()

    def to_time(timestamp):
        d = timestamp.to_pydatetime()+ timedelta(hours=-5)
        return d.strftime("%H")

    def to_mexican(timestamp):
        d = timestamp.to_pydatetime()+ timedelta(hours=-5)
        return d

    result_df["hour"] = result_df["time"].apply(to_time)
    result_df["time"] = result_df["time"].apply(to_mexican)
    result_df = result_df.drop(['diff'], axis=1)
    result_df['traffic_proportion'] = result_df['duration_with_traffic']/result_df['duration']
    result_df['has_traffic'] = [1 if row['traffic_proportion'] > grouped_data['traffic_proportion'].loc[row['name']]['std_shift'] else 0  for i, row in result_df.iterrows()] 
    indexed = result_df.set_index('time')
    indexed = indexed.drop(['duration', 'duration_with_traffic', 'traffic_proportion'], axis=1)
    indexed = pd.pivot_table(indexed, index=["time", "hour"], columns='name')['has_traffic']
    indexed.reset_index(inplace=True)
    indexed['origin'] = origin
    indexed['destination'] = destination
    return indexed


model, sample_entry = trainModel()
grouped_data = get_traffic_model()

@app.route('/')
def index():
    return render_template('index.html')
  
@app.route('/getRoute', methods=['GET', 'POST'])
def getRoute():
    origin = request.form.get("origin", None)
    destination = request.form.get("destination", None)
    data = get_current_route_data(origin, destination)
    
    data_dummies = pd.get_dummies(data.iloc[:,2:14])
    checkpoints = [
        "Hospicio Cabañas",
        "Plaza Galerías",
        "Andares",
        "Secretaría de Movilidad",
        "La Minerva",
        "Basílica de Zapopan"
    ]
    for ch in checkpoints:
        if ch != origin:
            o = "origin_"+ch
            data_dummies[o] = 0
        if ch != destination:
            d = "destination_"+ch
            data_dummies[d] = 0
    data_dummies = data_dummies[sample_entry.columns]
    y = model.predict(data_dummies)[0]
    print(y)
    if y!=None:
        return render_template("index.html", predicted = y, origin = origin, destination = destination)

if __name__ == '__main__':
    app.run()
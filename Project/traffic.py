import googlemaps as gm
import pandas as pd
import numpy as np
from datetime import datetime

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

my_key = "AIzaSyBWN4K23yA7pPd1Qyw5hdxhyD9zTPtn2ws"
my_maps = gm.Client(my_key)

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

from apscheduler.schedulers.blocking import BlockingScheduler

def traffic_job():
    print("Getting traffic at "+str(datetime.now())+"...\n")
    
    traffic = TrafficFramer(my_maps, checkpoints, routes)
    result_df = traffic.getTrafficDataframe()
    
    output_filename = str(datetime.now().timestamp())+"_traffic"
    result_df.to_csv(path_or_buf=output_filename, index=False)
    print("Successfully got traffic...\n")

def route_job():
    print("Getting traffic at "+str(datetime.now())+"...\n for routes")
    
    traffic = TrafficFramer(my_maps, checkpoints, routes)
    result_df = traffic.getTrafficRoutesDataframe()
    
    output_filename = str(datetime.now().timestamp())+"_routes.csv"
    result_df.to_csv(path_or_buf=output_filename, index=False)
    print("Successfully got traffic for routes...\n")

scheduler = BlockingScheduler()
scheduler.add_job(route_job, 'interval', hours=1)
scheduler.start()
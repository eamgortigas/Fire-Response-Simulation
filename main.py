from flask import Flask, send_from_directory, jsonify, request
import psycopg2 as sql
import pandas as pd
import numpy as np
import geopandas as gpd
import re
from shapely.geometry import Point,Polygon,LineString
from shapely.ops import nearest_points
import pickle
import networkx as nx
import osmnx as osx
import json

app = Flask(__name__, static_folder='static/')
debug = False
# BFP data names
with open('data/graph_qc2.pkl', 'rb') as f:
    graph = pickle.load(f)
# print('test')
# print(type(graph))
# firestationsFile = 'data/bfp_firestations31july2017.csv'
firetrucksFile = 'data/qc_coords_gdf.pkl'
multipolygonsFile = 'data/multipolygons_32651.pkl'

with open(multipolygonsFile, 'rb') as f:
    multipolygons_32651 = pickle.load(f) 
with open(firetrucksFile, 'rb') as f:
    qc_coords_ = pickle.load(f)


def shortest_distance(graph, o_c, dest_c, debug=False):
    """
    Creates a GeoDataFrame containing the edges of the shortest path
    from an origin to a destination point. These edges are represented 
    as linestrings that can be plotted on a map (folium works)
    
    Also prints the distance of the computed path.
    
    Parameters
    ----------
    graph : Graph 
        osmnx graph of the area containing the points
        
    o_c : tuple of floats
        coordinates of origin point in Long/Lat format
        
    dest_c : tuple of floats
        coordinates of destination point in Long/Lat format
        
    Returns
    -------
    path_gdf : GeoDataFrame
    
    shortest_dist : float
    """
    
    # Create a gdf to convert the geometry to standard crs
    
    try:
        gdf = gpd.GeoDataFrame([{"geometry": Point(o_c)},
                                {"geometry": Point(dest_c)}], 
                               crs={'init':'epsg:4326'})
        gdf.to_crs({'init': 'epsg:32651'}, inplace=True)
        origin_point = gdf.loc[0].geometry
        dest_point = gdf.loc[1].geometry
        
        # find the nearest node to the input points
        O = osx.get_nearest_node(graph, (origin_point.y, origin_point.x), 
                                 method='euclidean', return_dist=True)
        D = osx.get_nearest_node(graph, (dest_point.y, dest_point.x), 
                                 method='euclidean', return_dist=True)

        """
        create a dictionary of edges and their associated features
        note you can inspect the features by printing node_dict to see
        other available features""
        """   
        path = nx.shortest_path(graph, source=O[0], target=D[0])
        path_edges = []
        adj = graph.adj
        for i in range(0,len(path)-1):
            edges = adj[path[i]][path[i+1]]
            min_length = float("inf")
            min_edge = None
            for key in edges: 
                edge = edges[key]
                if edge["length"] < min_length:
                    min_length = edge["length"]
                    min_edge = edge
            path_edges.append((path[i],path[i+1],min_edge))
        node_dict = dict(graph.nodes)

        """
        generate a geodataframe to make it folium-ready
        """   
        gdf_builder = [] 
        for edge in path_edges:
            u = node_dict[edge[0]]
            v = node_dict[edge[1]]
            curr = edge[2] 
            if "geometry" not in curr: 
                geom = LineString([(u['x'],u['y']),(v['x'],v['y'])])
            else:
                geom = curr['geometry']
            res = {"source": edge[0], "target": edge[1], 
               "distance": edge[2]['length'],"geometry":geom}
            gdf_builder.append(res)
        path_gdf = gpd.GeoDataFrame(gdf_builder,crs={'init':'epsg:32651'})
        path_gdf.to_crs({'init':'epsg:4326'},inplace=True)
        path_gdf

        shortest_dist = path_gdf['distance'].sum()
    #     print('distance:', shortest_dist)

        # compute shortest path
    #     best_path = nx.shortest_path_length(graph, Point(o_c), Point(dest_c))   
        return path_gdf, shortest_dist
    except:
        print('Please choose a point closer to the road.')
def compute_time(speed, distance, return_minutes=True):
    """
    Parameters
    ----------
    speed : int
        speed in km/h
    distance : float
        distance in m
    """
    ms_speed = speed/3.6
    seconds = distance/ms_speed
    mins = seconds/60
    hours = mins/60
    if return_minutes:
        return round(mins, 2)wq
        
    else:
        return round(hours, 2)

def return_features(graph, dest, F_gdfs, geom_col, debug=False):
    print('Return Features')
    """
    NOTE: keep all inputs in lat/long format. 
    
    Parameters
    ----------
    dest : tuple
        destination of all pts
        
    F_gdfs : GeoDataFrame
        all origin points
    """
    gdfs = {}
    name_dists = []
    name = []:
    distance_ = []
    time_ = []
    index_ = []
    counter = 0
    
    for index,row in F_gdfs.iterrows():
        
        F = (row[geom_col].x, row[geom_col].y)
        try:  
            gdf, dist = shortest_distance(graph, F, dest[::-1])

            if compute_time(30, dist) <= 30:
                name_dists.append((row['name'], round(dist, 2), compute_time(30, dist)))
                name.append(row['name'])
                distance_.append(round(dist, 2))
                time_.append(compute_time(30, dist))
                gdfs[row['name']] = json.loads(gdf.to_json())
                index_.append(counter)
                counter += 1
            
            else:
                print(row['name'], 'was too far.')

        except:
            print('location outside graph')
            
    out = pd.DataFrame({'id': index_, 'name': name, 'distance': distance_, 'time': time_})
    
    return gdfs, out.sort_values(by='time')
#     return gdfs, name_dists, counter

def return_spread(multipolygons, stoptime, fireradius_init, firept, debug=False):
    """"
        Computes fire spreads of every fire started when a polygon is hit

        Inputs:
        ---------
        multipolygons     geodataframe of polygons of qc from OSM
        stoptime          earliest time the first fire truck arrives at    location
        fireradius_init   initial radius of fire
        firept            epicenter of initial fire

        Output:
        ----------
        dictionary with timestep as keys and values are list of epicenters , corresponding shapely object, corresponding geo object as geod dataframe, and corresponding radii of the fires triggered
    """
    print('Return Spread') 
    # types = {'nonburnable': 0, 'building': 1, 'house':2, 'natural': 3 }
    t = 0
    rateofburn = {0: 0, 1:0.333 , 2: 0.6667, 3: 0.8}
    ros = rateofburn[3]
    mean_load = multipolygons.fuel_load.mean()

    def spreadFire(center, radius, ros, t, reverse=True, is_shape=False, epsg=32651):
        """
            gets fire line circle polygon in 32651 given the inputs
        """
        mult = ros*t
        if reverse:
            c = center if is_shape else Point(center[::-1])
            gdf = (gpd.GeoSeries(c, crs={'init':f'epsg:{epsg}'})
                    .to_crs(epsg=32651))
            return gdf.values[0], gdf.buffer(radius+mult), radius+mult
        else:
            c = center if is_shape else Point(center)
            gdf = (gpd.GeoSeries(c, crs={'init':f'epsg:{epsg}'})
                    .to_crs(epsg=32651))
            return gdf.values[0], gdf.buffer(radius+mult), radius+mult
        

    #check if point is in multipolygon and which type
    fire_cntr_init, fire_init, rad_init = spreadFire(firept, 1, 0, 0, epsg=4326)
    flagdf = gpd.sjoin((gpd.GeoDataFrame(geometry=fire_init,
                                        crs={'init':'epsg:32651'})
                        .to_crs(epsg=32651))
            , multipolygons)
    firehp = (mean_load/rad_init)*ros
    gothit = flagdf.index_right.values.tolist()
    if flagdf.shape[0] > 0:
        ros = rateofburn[flagdf.ftype.values[0]]
        firehp = (rad_init/flagdf.iloc[0]['st_area'])*flagdf.iloc[0]['fuel_load']*ros
    ross = [ros]
    hps = [firehp]
    firess = [fire_cntr_init]
    results = {t:[fire_cntr_init, fire_init, fireradius_init]}
    # results
    while t<stoptime:
        # print(f'At {t}')
        results[t] = []
        for ic, fc in enumerate(firess):  
            firesscp = firess.copy()
            if ic == 0:
                fireCenter, fire, rad = spreadFire(fc, fireradius_init, ross[0], t,
                                            is_shape=True)
            else: 
                # print(ross[ic])
                fireCenter, fire, rad = spreadFire(fc, 1, ross[ic], t, is_shape=True)

            results[t].append((fireCenter, fire, rad))
            #check if it hits any multipolygons, if yes add to firess
            firegpd = gpd.GeoDataFrame(geometry=fire, crs={'init':'epsg:32651'})
            index_list = gpd.sjoin(firegpd, multipolygons).index_right.tolist()
    #         print(index_list)
            whatsnew =[x for x in index_list if x not in gothit]
            #only if there are new polygons to hit

            if whatsnew:
                # print(whatsnew)
                polygonhit = gpd.overlay(firegpd, multipolygons.iloc[whatsnew])
    #             print(polygonhit)
                hits = [nearest_points(firegpd.values[0][0], yy)[0] 
                            for yy in polygonhit['geometry']]
                hitTypes = polygonhit['ftype'].values.tolist()
                for ih, hc in enumerate(hits):
                    if hc not in firess:
                        firesscp.append(hc)
                        ross.append(rateofburn[hitTypes[ih]])
                        firehp = ((rad/polygonhit.iloc[ih]['st_area'])*
                              polygonhit.iloc[ih]['fuel_load']*ross[ic])
                        hps.append(firehp)
                gothit = gothit + whatsnew
                
                firess = firesscp
            else:
                # print('nothing new')
                continue
        t+=1
    return results, hps

def simulate_trucks(HP, df_sim, retain_data, debug=True):
    """
    Simulates the sequenced arrival of all trucks.
    """
    print('Simulate Trucks')
#     display(df_sim)
    # increases every step. Tracks minutes
    timer = 0

    # roller increases to index through time_steps
    # will increase only if condition is fulfilled
    roller = 0

    # power is initalized as a list and summed later on
    # we take each fire station's group of trucks as a single entity
    power = []
    limiter = np.array([])
    
    # place here all stations deployed
    stations = pd.DataFrame()

    # generate a list of all unique values in time
    time_steps = list(set(df_sim['time_round'].tolist()))

    while HP > 0:
        if debug:
            print("minutes elapsed:", timer)

        # will perform this action if there are still fire trucks that have 
        # not arrived
        try:
            if timer == time_steps[roller]:
                if debug:
                    print(timer)

                sliced = df_sim[df_sim['time_round']==time_steps[roller]]

                # creates a sliced dataframe so we can iterate through all values
                # needed in the df
                if debug:
                    print(sliced)
                stations = pd.concat((stations, sliced))

                power_add = []

                for index,row in sliced.iterrows():
                    """
                    PLEASE ADD SOMETHING TO MODIFY THE POWER VALUE
                    """
                    power_add.append(row['power'])
                    limiter = np.append(limiter, int(row['limit']))
                if debug:
                    print("power added:", sum(power_add))
                power.extend(power_add)
                # These track indexes from limiter to check
                # if it still has water in the tank
                if len(np.where(limiter==0)[0]) == 0:
                    if debug:
                        print("keep subtracting")
                else:
                    if debug:
                        print("something has reached 0")
                    indexes = list(np.where(limiter==0)[0])
                    if debug:
                        print(indexes)
                    replacements = [0 for i in range(len(indexes))]
                    if debug:
                        print(replacements)

                    for (index, replacement) in zip(indexes, replacements):
                        power[index] = replacement
                if debug:
                    print(limiter)

                # time passes

                limiter = limiter - 1
                roller += 1
                timer += 1
#                 if stations.empty:
#                     HP = increase_fire(HP, 0.02)
#                     if debug:
#                         print('fire grows')
            else:
                timer +=1
#                 if stations.empty:
#                     HP = increase_fire(HP, 0.02)
#                     if debug:
#                         print('fire grows')

        # when all fire trucks arrive, they need to exhaust all capacities
        except:
            print("All fire trucks have arrived")

            if len(np.where(limiter==0)[0]) == 0:
                    print("keep subtracting")
            else:
                print("something has reached 0")
                indexes = list(np.where(limiter==0)[0])
                print(indexes)
                replacements = [0 for i in range(len(indexes))]
                print(replacements)

                for (index, replacement) in zip(indexes, replacements):
                    power[index] = replacement

            if power == [0 for x in range(len(power))]:
                print('failed to extinguish fire :--(')
                break

            limiter = limiter - 1
            timer += 1
        if debug:
            print('current power:', power)
        HP -= sum(power)
        if debug:
            print('remaining HP:', HP)
            
    if debug:
        print("number of stations deployed:", len(stations))
        print(timer, "minutes to extinguish fire")
    
    return len(stations), timer, retain_data.merge(stations)

#routes
@app.route('/')
def mainload():
    return send_from_directory('', 'main.html')

@app.route('/passCentroid')
def passCentroid():
    """Get and return QC centroid from db or supply hard code for poc"""

    connect = sql.connect(
        dbname="eamgo",
        user="eamgo",
        password="shir0Kitsune08",
        port=5432,
        host="127.0.0.1"
    )

    gadmquery="""
        SELECT ST_X(ST_CENTROID(geom)) as long, ST_Y(ST_CENTROID(geom)) as lat
        FROM gadm.ph
        WHERE name_2 ~* 'quezon'
        AND name_1 ~* 'manila'
    """
    with connect.cursor() as cur:
        cur.execute(gadmquery)
        # print({list(cur)[0]})
        ct_long, ct_lat = list(cur)[0]
        return jsonify({"x": ct_long, "y": ct_lat})

@app.route('/getStations')
def getStations():
    """Retrieve QC firestation lat longs from BFP file and geo tagged from above"""
    def getLatLongStr(x):
        # print(x['dy'].x, x['dy'].y)
        x['lat_long'] = (x['dy'].y, x['dy'].x)
        return x
    coords = qc_coords_.apply(getLatLongStr, axis=1)
    res = coords[['name', 'lat_long']].to_dict(orient='list')
    # res_ = res.apply(getLatLongStr, axis=1)
    # print(res)
    return jsonify(res)

@app.route('/getShortestPaths', methods=['POST'])
def getShortestPaths():
    # print(json.loads(request.data)) 
    success = False
    firepoint = tuple(json.loads(request.data)['firepoint'].values())
    # points = qc_coords_['geometry']
    # print(type(firepoint[0]))
    # print(qc_coords_)
    paths_gdfs, df = return_features(graph, firepoint, qc_coords_, geom_col='dy')
    # print(df)
    # return jsonify(paths_gdfs)
    if len(paths_gdfs)>0:
        success = True
    # print(success)
    if success:
        return jsonify({'gdfs': paths_gdfs, 'df': df.to_dict(), 'dfid': str(df.iloc[0].id), 
        'dfname': df.iloc[0]['name'],'success': success})
    else:
        return jsonify({'success': success, 'msg': 'Could not compute shortest path. Location outside graph.'})

@app.route('/spreadFire', methods=['POST'])
def spreadFire():
    print('Simulate Spread')
    #fire spread variables

    reqdata = json.loads(request.data)
    # print(reqdata)
    stoptime = reqdata['stoptime']#50 #first truck time
    fireradius_init = int(reqdata['rad_init'])
    firept = tuple(reqdata['firepoint'].values()) #(14.644973,121.050858)
    # print(stoptime,fireradius_init, firept )
    
    
    # multipolygons_32651 
    results, hps = return_spread(multipolygons_32651, stoptime, fireradius_init, firept)
    centers, radii = [], []
    for i in results[np.floor(stoptime)]:
        center = i[0]
        to_4326 = (gpd.GeoSeries(center, crs={'init':f'epsg:{32651}'})
                    .to_crs(epsg=4326)).values[0]
        centers.append((to_4326.y, to_4326.x))
        radii.append(i[2])
    return jsonify({'epicenters': centers, 'radii': radii, 'total_rad': np.unique(radii).sum(), 'hp': np.sum(hps)})

@app.route('/simulateTrucks', methods=['POST'])
def simulateTrucks():
    #needed paths_gdfs, df , hp
    reqdata = json.loads(request.data)
    paths_gdfs = reqdata['path_gdfs']
    df = reqdata['df']
    fireres = reqdata['fireres']
    print(fireres)
    # print(pd.DataFrame(df))
    df_sim = pd.DataFrame(df).merge(qc_coords_)
    df_sim['time_round'] = df_sim['time'].apply(lambda x: round(x))
    # print(qc_coords_)
    # print(df_sim)
    stations_deployed, time_elapsed, fire_stations = simulate_trucks(fireres['hp'], df_sim, 
                qc_coords_, debug=False)
    # print(fire_stations)
    
    paths_gdfs_ = {k:v for k,v in paths_gdfs.items() if k in fire_stations['name'].tolist()}
    return jsonify({'paths_geojson': paths_gdfs_, 'fireres': fireres, 'stations_deployed':stations_deployed, 
    'gallons': fire_stations['power'].sum(),
    'stations': '\n\n'.join(fire_stations['name'].tolist()), 
        'time_elapsed': time_elapsed})


if __name__ == '__main__':
    app.run()





# old codes
# print(qc_coords_)

# fireStationsData = pd.read_csv(firestationsFile, engine='python').query("city_or_municipality=='QUEZON CITY'")
# firetrucks_power = (pd.read_csv(firetrucksFile, engine='python').query("city_or_municipality == 'QUEZON CITY'")
#     .groupby(['station_name','vehicle_type', 'vehicle_capacity'])['truck_id']
#     .count()
#     .reset_index()
#     .pivot_table(index='station_name', columns='vehicle_capacity', values='truck_id')
#     .drop('-', axis=1)
#     .reset_index())


# #QC firestation geotags
# qc_gm_coords_4326 = {'AGHAM FIRE SUBSTATION':(14.651923,121.038508), 
#                      'BAESA FIRE SUBSTATION':(14.671719,121.009174),
#                      'BAHAY TORO FIRE SUBSTATION': (14.666898,121.021191),
#                      'BFP NATIONAL HEADQUARTERS': (14.653407,121.037840),
#                      'COMMONWEALTH FIRE SUBSTATION': (14.697740,121.088307),
#                      'CONGRESS FIRE SUBSTATION':(14.693757, 121.0944050),
#                      'EASTWOOD (LIBIS) FIRE SUBSTATION': (14.611728,121.076502),
#                     'FAIRVIEW FIRE SUBSTATION': (14.706801,121.072979),
#                     'FRISCO FIRE SUBSTATION':(14.654766,121.018787),
#                     'GALAS FIRE SUBSTATION':(14.611947,121.009073),
#                     'HOLY SPIRIT FIRE SUBSTATION':(14.683850,121.076277),
#                     'LA LOMA FIRE SUBSTATION':(14.631767,120.994742),
#                     'LAGRO FIRE SUBSTATION':(14.722487,121.068227),
#                     'MARILAG FIRE SUBSTATION':(14.651344,121.058021),
#                     'MASAMBONG FIRE SUBSTATION':(14.640496,121.012170),
#                     'NEW ERA FIRE SUBSTATION':(14.667124,121.060724),
#                     'NOVALICHES FIRE SUBSTATION':(14.721418,121.035412),
#                     'ODFM - DISTRICT V': None, 
#                     'ORD - NATIONAL CAPITAL REGION':(14.627967,121.046283),
#                     'PALIGSAHAN FIRE SUBSTATION': (14.629973,121.025258),
#                     'PINAGKAISAHAN FIRE SUBSTATION':(14.627979,121.046390),
#                     'PROJECT 6 FIRE SUBSTATION': (14.662268,121.040709),
#                     'QUEZON CITY FIRE STATION': (14.611771,121.060002),
#                     'QUIRINO 2A FIRE SUBSTATION':(14.631393,121.059118),
#                     'RAMON MAGSAYSAY FIRE SUBSTATION':(14.659436,121.021551),
#                     'SAN BARTOLOME FIRE SUBSTATION':(14.710947, 121.128138),
#                     'SANTA LUCIA FIRE SUBSTATION':(14.707516,121.053329),
#                     'TALIPAPA FIRE SUBSTATION': (14.687718,121.025427)}

# qc_coords = pd.DataFrame(zip(qc_gm_coords_4326.keys(), qc_gm_coords_4326.values()), columns=['station_name','lat_long'])
# qc_coords_ = qc_coords[~qc_coords['lat_long'].isnull()].merge(firetrucks_power.reset_index(), on='station_name')
# geometry = [Point(x) for x in qc_coords_['lat_long']]
# qc_coords_ = gpd.GeoDataFrame(qc_coords_, geometry=geometry)


# functions
# def shortest_distance(graph, o_c, dest_c):
#     """
#     Creates a GeoDataFrame containing the edges of the shortest path
#     from an origin to a destination point. These edges are represented 
#     as linestrings that can be plotted on a map (folium works)
    
#     Also prints the distance of the computed path.
    
#     Parameters
#     ----------
#     graph : Graph 
#         osmnx graph of the area containing the points
        
#     o_c : tuple of floats
#         coordinates of origin point in Long/Lat format
        
#     dest_c : tuple of floats
#         coordinates of destination point in Long/Lat format
        
#     Returns
#     -------
#     path_gdf : GeoDataFrame
    
#     shortest_dist : float
#     """
    
#     # Create a gdf to convert the geometry to standard crs
    
#     try:
#         gdf = gpd.GeoDataFrame([{"geometry": Point(o_c)},
#                                 {"geometry": Point(dest_c)}], 
#                                crs={'init':'epsg:4326'})
#         gdf.to_crs({'init': 'epsg:32651'}, inplace=True)
#         origin_point = gdf.loc[0].geometry
#         dest_point = gdf.loc[1].geometry
#         # print(origin_point, dest_point)
#         # find the nearest node to the input points
#         O = osx.get_nearest_node(graph, (origin_point.y, origin_point.x), 
#                                 method='euclidean', return_dist=True)
#         D = osx.get_nearest_node(graph, (dest_point.y, dest_point.x), 
#                                 method='euclidean', return_dist=True)
#         # print()

#         """
#         create a dictionary of edges and their associated features
#         note you can inspect the features by printing node_dict to see
#         other available features""
#         """   
#         path = nx.shortest_path(graph, source=O[0], target=D[0])
#         path_edges = []
#         adj = graph.adj
#         for i in range(0,len(path)-1):
#             edges = adj[path[i]][path[i+1]]
#             min_length = float("inf")
#             min_edge = None
#             for key in edges: 
#                 edge = edges[key]
#                 if edge["length"] < min_length:
#                     min_length = edge["length"]
#                     min_edge = edge
#             path_edges.append((path[i],path[i+1],min_edge))
#         node_dict = dict(graph.nodes)

#         """
#         generate a geodataframe to make it folium-ready
#         """   
#         gdf_builder = [] 
#         for edge in path_edges:
#             u = node_dict[edge[0]]
#             v = node_dict[edge[1]]
#             curr = edge[2] 
#             if "geometry" not in curr: 
#                 geom = LineString([(u['x'],u['y']),(v['x'],v['y'])])
#             else:
#                 geom = curr['geometry']
#             res = {"source": edge[0], "target": edge[1], 
#                "distance": edge[2]['length'],"geometry":geom}
#             gdf_builder.append(res)
#         path_gdf = gpd.GeoDataFrame(gdf_builder,crs={'init':'epsg:32651'})
#         path_gdf.to_crs({'init':'epsg:4326'},inplace=True)
#         path_gdf

#         shortest_dist = path_gdf['distance'].sum()
#     #     print('distance:', shortest_dist)

#         # compute shortest path
#     #     best_path = nx.shortest_path_length(graph, Point(o_c), Point(dest_c))   
#         return path_gdf, shortest_dist
#     except:
#         print('Please choose a point closer to the road.')

# def compute_time(speed, distance, return_minutes=True):
#     """
#     Parameters
#     ----------
#     speed : int
#         speed in km/h
#     distance : float
#         distance in m
#     """
#     ms_speed = speed/3.6
#     seconds = distance/ms_speed
#     mins = seconds/60
#     hours = mins/60
#     if return_minutes:
#         return round(mins, 2)
#     else:
#         return round(hours, 2)

# def return_features(graph, dest, F_gdfs, geom_col):
#     """
#     NOTE: keep all inputs in lat/long format. 
    
#     Parameters
#     ----------
#     loc : tuple
#         center from where to make the map
        
#     dest : tuple
#         destination of all pts
        
#     F_gdfs : GeoDataFrame
#         all origin points
#     """
#     gdfs = {}
#     name_dists = []
#     name = []
#     distance_ = []
#     time_ = []
#     index_ = []
#     counter = 0
    
#     for index,row in F_gdfs.iterrows():
        
#         F = (row[geom_col].y, row[geom_col].x)
#         try:  
#             gdf, dist = shortest_distance(graph, F, dest[::-1])

#             if compute_time(30, dist) <= 20:
#                 name_dists.append((row['station_name'], round(dist, 2), compute_time(30, dist)))
#                 name.append(row['station_name'])
#                 distance_.append(round(dist, 2))
#                 time_.append(compute_time(30, dist))
#                 gdfs[counter] = json.loads(gdf.to_json())
#                 index_.append(counter)
#                 counter += 1
            
#             else:
#                 print(row['station_name'], 'was too far.')

#         except:
#             print('location outside graph')
            
#     out = pd.DataFrame({'id': index_, 'name': name, 'distance': distance_, 'time': time_})
#     return gdfs, out.sort_values(by='time')
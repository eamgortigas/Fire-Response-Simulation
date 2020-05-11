# Fire-Response-Simulation

A proof of concept project that uses OSMNx, Networkx, PostGIS, and Leaflet to visualize a fire location, its intensity progression, all deployed fire stations, and their respective paths. Distance computations are done in meters, EPSG: 32651 for the Philippines and converted back to 4326 for map visualizations.

### Fire Intensity Computation

Visualized as circles with a radius and rate of spread as a function of the initial intensity value set as an integer, building material, and polygon features.

### Fire Truck Paths

Visualized as lines using Networkx and OSMnx shortest path geojson objects. Travel time is computed with an assumed constant velocity of 20 km/h with no traffic congestion taken into account. 

### Simulating Truck Order of Arrival and Water Output

All trucks are assumed to have an output of 175 gallons per minute, the most commonly used hose. Trucks are assigned different capacities depending on their indicated class in fire station date. For this proof of concept, fire trucks output water the moment they arrive at the location, fire intensity stops growing once the first set of trucks arrives, and all actions are executed in 1-minute intervals.

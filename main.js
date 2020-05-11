//variables
// icons are from flaticon
const flameOn = L.icon({
    iconUrl: '../img/fire.svg',
    // shadowUrl: 'leaf-shadow.png',

    iconSize: [30, 30], // size of the icon
    // shadowSize: [50, 64], // size of the shadow
    // iconAnchor: [22, 94], // point of the icon which will correspond to marker's location
    // shadowAnchor: [4, 62],  // the same for the shadow
    // popupAnchor: [-3, -76] // point from which the popup should open relative to the iconAnchor
});
const fireStation = L.icon({
    iconUrl: '../img/fire-station.svg',
    iconSize: [15, 15]
})

map = L.map('resultsmap')
arsonLoc = null
flameCircle = null
flameCircles = []
thePaths = null
theShortestPath = null
fire = null

//storing needed ouputs
path_gdfs = null
df = null
dfid = null

//html elements
fireHp = document.querySelector('#gb_hp')
mask = document.querySelectorAll('#mask')[0]
simulateBtn = document.querySelectorAll('#simulate')[0]
fireResDiv = document.querySelector('#fire_res')
fire_hp = document.querySelector('#fire_hp')
time_elapsed = document.querySelector('#time_elapsed')
station_total = document.querySelector('#station_total')
gallons = document.querySelector('#gallons')
stations = document.querySelector('#stations')
fire_centers = document.querySelector('#fire_centers')


//leaflet mapping (4326 epsg)
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map)
//set centroid map
fetch('/passCentroid').then(resp => {
    return resp.json()
}).then(data => {
    // console.log(data['x'])
    // console.log(y)
    map.setView([data['y'], data['x']], 15);
    fetch('/getStations').then(respfs => {
        return respfs.json()
    }).then(datafs => {
        // console.log(datafs)
        for (i = 0; i < datafs['name'].length; i++) {
            L.marker(datafs['lat_long'][i], { icon: fireStation, title: datafs['name'][i] }).addTo(map)
        }
        // datafs['lat_long'].forEach(el => {
        //     L.marker(el, { icon: fireStation, title: 'fs' }).addTo(map)
        // });
    })
})

//fire starter events
fireHp.addEventListener('change', (event) => {
    // console.log(fireHp.value)
    if (flameCircle !== null) {
        flameCircle.remove();
    }
    // print(arsonLoc.getLatLng())
    flameCircle = L.circle(arsonLoc.getLatLng(), {
        color: 'orange',
        fillOpacity: 0.5,
        radius: fireHp.value
    }).addTo(map)
})
// console.log(fireHp)

map.on('click', (event) => {
    if (arsonLoc !== null) {
        arsonLoc.remove();
        flameCircle.remove();
    }


    arsonLoc = L.marker(event.latlng, { icon: flameOn }).addTo(map)
    flameCircle = L.circle(arsonLoc.getLatLng(), {
        color: 'red',
        // radialGradient: ['red', 'orange', 'yellow'],
        fillOpacity: 0.5,
        radius: fireHp.value
    }).addTo(map)
    fire = event.latlng
    console.log(fire)
})
// resolve(event.latlng)
// }).then(fire => {
simulateBtn.addEventListener('click', (event) => {
    //     console.log(event)
    // })
    // console.log(fire)
    mask.style.visibility = 'visible'
    fetch('/getShortestPaths', {
        method: "POST",
        body: JSON.stringify({
            firepoint: fire
        })
    }).then(resp => {
        return resp.json()
    }).then(datagdfs => {
        console.log(datagdfs.success)

        //visualize shortest paths
        new Promise((resolve, reject) => {
            if (datagdfs.success == false) {
                reject(datagdfs.msg)
            }
            path_gdfs = datagdfs['gdfs']
            df = datagdfs['df']
            // dfid = datagdfs['dfid']
            dfid = datagdfs['dfname']

            resolve(datagdfs['df']['time'][parseInt(datagdfs['dfid'])])
        }).then(stoptime => {

            //get firespreads
            fetch('/spreadFire', {
                method: "POST",
                body: JSON.stringify({
                    stoptime: stoptime,
                    rad_init: fireHp.value,
                    firepoint: fire
                })
            }).then(resp => {
                return resp.json()
            }).then(results => {
                // console.log(results)
                //visualize fire circles
                if (flameCircles.length > 0) {
                    // flameCircles.remove()
                    flameCircles.forEach(el => {
                        el.remove()
                    })
                }
                for (i = 0; i < results['epicenters'].length; i++) {
                    color = i > 0 ? 'purple' : 'orange'
                    flameCircles.push(L.circle(results['epicenters'][i], {
                        color: color,
                        fillOpacity: 0.5,
                        radius: results['radii'][i]
                    }))//.addTo(map)
                }
                flameCircles.forEach(el => {
                    el.addTo(map)
                })
                // console.log(results['total_rad'])
                return results
            }).then((results) => {

                // console.log(results)
                fetch('/simulateTrucks', {
                    method: "POST",
                    body: JSON.stringify({
                        path_gdfs: path_gdfs,
                        df: df,
                        fireres: results
                    })
                }).then(resp => {
                    return resp.json()
                }).then(output => {
                    console.log(output)
                    mask.style.visibility = 'hidden'
                    new Promise((resolve, reject) => {
                        console.log(dfid)
                        shortestpaths = output['paths_geojson']

                        //     // console.log(datagdfs['df'])
                        thePathsList = []
                        Object.keys(shortestpaths).forEach(key => {
                            // console.log(key == parseInt(datagdfs['df']))
                            if (key != dfid) {
                                // console.log(key)
                                thePathsList.push(shortestpaths[key])
                            }
                        })
                        // console.log(thePathsList)
                        if (thePaths !== null) {
                            thePaths.remove();
                        }

                        othershortestpaths = []
                        Object.values(thePathsList).forEach(props => {
                            // console.log(props)
                            othershortestpaths.push(props['features'])
                        })
                        thePaths = L.geoJSON(othershortestpaths.flat())
                        thePaths.addTo(map)
                        if (theShortestPath !== null) {
                            theShortestPath.remove();
                        }

                        theShortestPath = L.geoJSON(shortestpaths[dfid], {
                            style: function (feature) {
                                return { 'color': 'green' }
                            }
                        })
                        theShortestPath.addTo(map);

                        // resolve(datagdfs['df']['time'][parseInt(datagdfs['dfid'])])
                        fire_hp.innerText = output['fireres']['hp']
                        time_elapsed.innerText = output['time_elapsed'].toString() + ' min'
                        station_total.innerText = output['stations_deployed']
                        gallons.innerText = output['gallons']
                        stations.innerText = output['stations']
                        fire_centers.innerText = output['fireres']['epicenters'].length
                        fireResDiv.style.visibility = 'visible'

                    })
                })
            })
        }).catch((msg) => {
            mask.style.visibility = 'hidden'
            alert(msg)
        })

    })

})






// references
    // < div > Icons made by < a href = "https://www.flaticon.com/authors/freepik" title = "Freepik" > Freepik</a > from < a href = "https://www.flaticon.com/" title = "Flaticon" > www.flaticon.com</a ></div >
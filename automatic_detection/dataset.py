from .config import cfg
import copy as cp
import numpy as np
import datetime as dt


class Network():
    """Station data:
    Contains stations and geographical coordinates.

    network_file = station ascii file name.
    """
    def __init__(self, network_file):
        self.where = cfg.network_path + network_file

    def n_stations(self):
        return np.int32(len(self.stations))

    def n_components(self):
        return np.int32(len(self.components))

    def read(self):
        networks = []
        stations = []
        components = []
        with open(self.where, 'r') as file:
            # read in start and end dates
            columns = file.readline().strip().split()
            self.start_date = dt.date(
                int(columns[0][0:0 + 4]),
                int(columns[0][4:4 + 2]),
                int(columns[0][6:6 + 2]))
            self.end_date = dt.date(
                int(columns[1][0:0 + 4]),
                int(columns[1][4:4 + 2]),
                int(columns[1][6:6 + 2]))

            # read in component names
            columns = file.readline().strip().split()
            for component in columns[1:]:
                components.append(component)
            self.components = components
            
            data_centers = []
            networks     = []
            locations    = []

            # read in station names and coordinates
            latitude, longitude, depth = [], [], []
            for line in file:
                columns = line.strip().split()
                data_centers.append(columns[0])
                networks.append(columns[1])
                stations.append(columns[2])
                locations.append(columns[3])
                latitude.append(np.float32(columns[4]))
                longitude.append(np.float32(columns[5]))
                depth.append(-1.*np.float32(columns[6]) / 1000.)  # convert m to km

            self.data_centers = data_centers
            self.networks     = networks
            self.stations     = stations
            self.locations    = locations
            self.latitude     = np.asarray(latitude, dtype=np.float32)
            self.longitude    = np.asarray(longitude, dtype=np.float32)
            self.depth        = np.asarray(depth, dtype=np.float32)

    def datelist(self):
        dates = []
        date = self.start_date
        while date <= self.end_date:
            dates.append(date)
            date += dt.timedelta(days=1)

        return dates

    def stations_idx(self, stations):
        if not isinstance(stations, list) and not isinstance(stations, np.ndarray):
            stations = [stations]
        idx = []
        for station in stations:
            idx.append(self.stations.index(station))
        return idx

    def subset(self, stations, components):
        subnetwork = cp.deepcopy(self)

        if not isinstance(stations, list):
            stations = [stations]
        if not isinstance(components, list):
            components = [components]

        for station in stations:
            if station in self.stations:
                idx = subnetwork.stations.index(station)
                subnetwork.stations.remove(station)
                np.delete(subnetwork.latitude, idx)
                np.delete(subnetwork.longitude, idx)
                np.delete(subnetwork.depth, idx)
            else:
                print('{} not a network station'.format(station))

        for component in components:
            if component in self.components:
                idx = subnetwork.components.index(component)
                subnetwork.components.remove(component)
            else:
                print('{} not a network component'.format(station))

        return subnetwork


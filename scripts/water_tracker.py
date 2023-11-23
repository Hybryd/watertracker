import pandas as pd
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
import numpy as np
from datetime import date, timedelta


class WaterTracker():
    def __init__(self):
        self.data_folder = "data/"
        
        # self.df contains information about the stations:
        # ['id', 'bss_code', 'name', 'slug', 'latitude', 'longitude',
        # 'coordinates', 'altitude', 'is_active', 'is_open', 'is_forecast',
        # 'creation_date', 'created_at', 'updated_at', 'info', 'type',
        # 'indicators', 'decree_restrictions', 'forecast_watchers',
        # 'forecast_indicators', 'departement', 'departement_name',
        # 'departement_code'],
        self.df = pd.read_csv(f"{self.data_folder}stations.csv")

        
        # self.data_departements is a dictionary with information about departements.
        # The first entry corresponds to the Ain departement:
        # { 1: {"data" : {'id' : 1,
        #                 'code': "01",
        #                 'name': "Ain",
        #                 'geometry': {'type': 'Polygon',
        #                              'coordinates': [[[4.780208, 46.176676],
        #                                              [4.780243, 46.189052],
        #                                                ...
        #                                                ]]
        #                             },
        #                 'indicators': [...],
        #                 'locations': [...]
        # }}
        print("Loading data_departements")
        with open(f"{self.data_folder}data_departements.pkl", 'rb') as fp:
            self.data_departements = pickle.load(fp)
        
        # self.departements_names is the dictionary {1:"Ain", ...}
        self.departements_names = {departement_code:v["data"]["name"] for departement_code, v in self.data_departements.items()}
        
        
        # self.ids_departements is a dictionary whose keys are the departement codes (1,2,3...) and the values are the list of location ids in the corresponding departement.
        # The first entry corresponds to the Ain departement:
        # [8168, 2575, 8418,...]
        print("Loading ids_departements")
        with open(f"{self.data_folder}ids_departements.pkl", 'rb') as fp:
            self.ids_departements = pickle.load(fp)
        
        # self.dict_time_series_departement is used to store temporarily the time series for all stations in a departement
        self.dict_time_series_departement = {}
        
        # self.dict_indic contains the indicator_type:indicator_name pairs
        self.dict_indic = {"dryness-groundwater"     : "water-level-static",
                           "dryness-groundwater-deep": "water-level-static",
                           "dryness-stream-flow"     : 'stream-flow', # TODO: check if it is the correct indicator
                           "dryness-small-stream"    : "observation",
                           "dryness-meteo"           : "rain-level",
                           }
        
        self.dict_standardized_indicators = {"spli" : { "dryness-groundwater"     : "water-level-static",
                                                        "dryness-groundwater-deep": "water-level-static",
                                                        "dryness-stream-flow"     : 'stream-flow',
                                                        "dryness-small-stream"    : "observation",
                                                        },
                                             "spi":  {"dryness-meteo" : "rain-level"}
                                             }
        # # Indicators on which we can compute the SPLI
        # self.dict_indic_spli = {"dryness-groundwater"     : "water-level-static",
        #                         "dryness-groundwater-deep": "water-level-static",
        #                         "dryness-stream-flow"     : 'stream-height',
        #                         "dryness-small-stream"    : "observation",
        #                         }
        
        # # Indicators on which we can compute the SPI
        # self.dict_indic_spi = {"dryness-meteo" : "rain-level"}
        
        # self.dict_indic_names contains the indicator_type:real_names pairs
        self.dict_indic_names = {"dryness-groundwater"    : "Nappes",
                                "dryness-groundwater-deep": "Nappes profondes",
                                "dryness-stream-flow"     : "Cours d'eau",
                                "dryness-small-stream"    : "Ruisseaux",
                                "dryness-meteo"           : "Pluie",
                               }
        
        #self.dict_indic_rev = {"water-level-static": ["dryness-groundwater","dryness-groundwater-deep"]  }
        self.levels = { -1.78        : ["Très bas", 'tab:red'],
                        -0.84        : ["Bas", 'tab:orange'],
                        -0.25        : ["Modérément bas", 'tab:yellow'],
                         0.25        : ["Autour de la normale", 'tab:green'],
                         0.84        : ["Modérément haut", 'tab:cyan'],
                         1.28        : ["Haut", 'tab:blue'],
                         float('inf'): ["Très haut", 'tab:olive']
                      }
        self.levels_colors = ["#da442c", "#f28f00", "#ffdd55", "#6cc35a", "#30aadd", "#1e73c3", "#286172"]

        # TODO: adel
        self.spi_levels = self.levels
        
        # self.count_spli_levels_france is of the form:
        #      indicator_type        month   number of stations that are in the corresponding dryness level in France for each month
        # {'dryness-groundwater': {   0:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             1:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             2:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             3:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             4:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             5:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             6:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             7:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             8:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             9:    {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             10:   {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        #                             11:   {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}},
        #   'dryness-groundwater-deep': ...
        # }


        self.counts_standardized_indicator_levels_france = {indicator : {month : {level_code: 0 for level_code in range(len(self.levels))} for month in range(12)} for indicator in self.dict_indic_names.keys()}

        self.count_spli_levels_france = {indicator : {month : {spli_level: 0 for spli_level in range(len(self.levels))} for month in range(12)} for indicator in self.dict_indic.keys()}

        # same as previously, but for spi only
        self.count_spi_levels_france = {indicator : {month : {spli_level: 0 for spli_level in range(len(self.levels))} for month in range(12)} for indicator in self.dict_indic.keys()}
        
        self.dict_time_series_france = {}
        self.counts_standardized_indicator_levels_departements = {departement_code : {indicator : None for indicator in self.dict_indic_names.keys()} for departement_code in self.data_departements.keys()}

    def load_time_series_departement(self, departement_code):
        print(f"Loading dict_time_series_departement_{departement_code}")
        with open(f"{self.data_folder}dict_time_series_departement_{departement_code}.pkl", 'rb') as fp:
            self.dict_time_series_departement = pickle.load(fp)
            
    
    def keep_timeseries_with_enough_data(self, departement_code, min_number_years=15):
        """
        Returns a dictionnary of the form:
        {   
            "dryness-groundwater" : [...],
            "dryness-groundwater-deep": [...],
            "dryness-stream-flow": [...],
            "dryness-small-stream" : [...],
            "dryness-meteo" : [...]
        }
        Where the lists contain the location ids for which the stations gathered enough data (more than min_number_years years of cumulated data)
        """
        res = {   
            "dryness-groundwater"     : [],
            "dryness-groundwater-deep": [],
            "dryness-stream-flow"     : [],
            "dryness-small-stream"    : [],
            "dryness-meteo"           : []
        }
        for indicator_type, list_timeseries in self.dict_time_series_departement[departement_code].items():

            if len(list(list_timeseries.values())) > 0:
                indicator_name = self.dict_indic[indicator_type]
                first = list(list_timeseries.values())[0]
                df_tmp = pd.DataFrame(first[indicator_name])
                df_tmp["date"] = pd.to_datetime(df_tmp["date"])
                df_tmp = df_tmp.set_index("date")
                dt_index = df_tmp.index
                for loc_id, timeseries in list_timeseries.items():
                    if (dt_index.max() - dt_index.min()).days/365.25 > min_number_years:
                        res[indicator_type].append(loc_id)
        return res


    def compute_months_functions(self, standardized_indicator, indicator_type, location_id, roll_window='90D'):
        """
        standardized_indicator must be "spli" or "spi"
        The parameter indicator_type must be one of the following: 
        - "dryness-groundwater"
        - "dryness-groundwater-deep"
        - "dryness-stream-flow"
        - "dryness-small-stream"
        
        Computes the Standardized Piezometric Level Index function for a given location_id for each months.
        The rolling mean is computed over 90 days by default.

        The result is a dict
        {
         1 : [X_jan, Y_jan],
         2 : [X_feb, Y_feb],
         ...
         12: [X_dec, Y_dec]
        }
        """
        res = {}
        departement_code = self.df[self.df["id"] == location_id]["departement_code"].values[0]
        indicator = None
        for k,v in self.dict_indic.items():
            if indicator_type == v:
                indicator = k
                break
        
        time_series = self.dict_time_series_departement[departement_code][indicator][location_id][indicator_type]
        
        df_station = pd.DataFrame(time_series)[["date","value"]]
        df_station["date"] = pd.to_datetime(df_station["date"])
        df_station = df_station.set_index("date")

        if standardized_indicator == "spli":
            # Calcul de la moyenne glissante sur 3 mois
            rolling = pd.DataFrame(df_station.value.rolling(roll_window).mean())
        else:
            # Calcul de la somme glissante sur 3 mois
            rolling = pd.DataFrame(df_station.value.rolling(roll_window).sum())

        rolling.columns = ["roll3M"]

        rolling = rolling.reset_index()
        x = rolling.groupby([rolling['date'].dt.month, rolling['date'].dt.year]).mean()
        x.index = x.index.set_names(['mois', 'annees'])
        x = x.reset_index()

        for m in range(1,13):
            hist, bin_edges = np.histogram(x[x.mois == m]["roll3M"].tolist(), density=True)

            X = np.linspace(np.array(bin_edges).min(), np.array(bin_edges).max(), 1000)[:, np.newaxis]

            log_dens = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(bin_edges[:, None])

            density = np.exp(log_dens.score_samples(X))
            Y = np.cumsum(density)
            res[m] = [X.reshape(1,-1)[0], Y]

        return res   
        
    
    def project_on_normal_law(self, X, Y, x):
        y = np.interp(x, X, Y)/(Y.max()-Y.min())
        return norm.ppf(y)
    
    def compute_standardized_indicator_for_station(self, standardized_indicator, indicator_type, location_id):
        departement_code = self.df[self.df["id"] == location_id]["departement_code"].values[0]
        indicator = None
        for k,v in self.dict_indic.items():
            if indicator_type in v:
                indicator = k
                break
        time_series = self.dict_time_series_departement[departement_code][indicator][location_id][indicator_type]
        standardized_indicator_station = self.compute_months_functions(indicator_type=indicator_type,
                                                                       location_id=location_id, 
                                                                       standardized_indicator=standardized_indicator, 
                                                                       roll_window='90D'
                                                                       )
        df_station = pd.DataFrame(time_series).drop(["min", "max", "depth"],axis=1)
        df_station["date"] = pd.to_datetime(df_station["date"])
        df_station['month'] = pd.DatetimeIndex(df_station['date']).month
        df_station["x"] = np.arange(len(df_station))
        df_station = df_station.set_index("date")
        df_station[standardized_indicator] = df_station.apply(lambda x: self.project_on_normal_law(standardized_indicator_station[x.month][0],
                                                                                                   standardized_indicator_station[x.month][1], 
                                                                                                   x.value
                                                                                                   ), 
                                                                                                   axis=1
                                                                                                   )
        return df_station

    def standardized_indicator_to_level_description(self, standardized_indicator_value):
        for k,v in self.levels.items():
            if standardized_indicator_value < k:
                return v
 
    def standardized_indicator_to_level_code(self, standardized_indicator_value):
        for i, (k, v) in enumerate(self.levels.items()):
            if standardized_indicator_value < k:
                return i
    
    def plot_spli(self, indicator_type, location_id):
        fig, ax = plt.subplots(figsize=(10,5))
        y1 = -1000
        y2 = -1.28
        y3 = -0.84
        y4 = -0.25
        y5 = 0.25
        y6 = 0.84
        y7 = 1.28
        df_station = self.compute_standardized_indicator_for_station(self, "spli", indicator_type, location_id)#self.spli_station(indicator_type, location_id)
        
        plt.fill_between(list(df_station["x"].values),
                         y2,
                         list(df_station["spli"].values),
                         color='red',
                         alpha=0.5,
                         where=df_station["spli"].values < y2
                        )
        plt.fill_between(list(df_station["x"].values),
                         y3,
                         list(df_station["spli"].values),
                         color='orange',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y2) & (df_station["spli"].values < y3))
                        )
        plt.fill_between(list(df_station["x"].values),
                         y4,
                         list(df_station["spli"].values),
                         color='yellow',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y3) & (df_station["spli"].values < y4))
                        )
        plt.fill_between(list(df_station["x"].values),
                         y5,
                         list(df_station["spli"].values),
                         color='green',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y4) & (df_station["spli"].values < y5))
                        )
        plt.fill_between(list(df_station["x"].values),
                         y6,
                         list(df_station["spli"].values),
                         color='grey',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y5) & (df_station["spli"].values < y6))
                        )
        plt.fill_between(list(df_station["x"].values),
                         y7,
                         list(df_station["spli"].values),
                         color='grey',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y6) & (df_station["spli"].values < y7))
                        )
        plt.fill_between(list(df_station["x"].values),
                         y7,
                         list(df_station["spli"].values),
                         color='grey',
                         alpha=0.5,
                         where= ((df_station["spli"].values >= y7))
                        )
        
    def standardized_indicator_mean_last_year(self, standardized_indicator, indicator_type, location_id):
        end_date = date.today().replace(day=1)
        one_year_before = (end_date.today()-timedelta(days=365)).strftime("%Y-%m-%d")
        df_station = self.compute_standardized_indicator_for_station(standardized_indicator, indicator_type, location_id)
        df_station = df_station.reset_index()
        df_station["date"] = pd.to_datetime(df_station["date"])
        df_station = df_station[df_station["date"] >= one_year_before]
        d = dict(df_station[standardized_indicator].groupby(df_station['date'].dt.month).mean().apply(lambda x: self.standardized_indicator_to_level_code(x)))
        return {k-1:v for k,v in d.items()}

    def compute_standardized_indicator_levels_for_departement(self, standardized_indicator, departement_code, indicator_name):
        indicator_type = self.dict_indic[indicator_name]
        counts = {i:{k:0 for k in range(len(self.levels))} for i in range(0,12)}
        ids = self.keep_timeseries_with_enough_data(departement_code, min_number_years=15)[indicator_name]
        if len(ids) > 0:
            for location_id in ids:
                location_standardized_indicator = self.standardized_indicator_mean_last_year(standardized_indicator, indicator_type, location_id)
                for month, standardized_indicator_code in location_standardized_indicator.items():
                    if not np.isnan(month) and not np.isnan(standardized_indicator_code):
                        counts[month][standardized_indicator_code] += 1
 
        return counts

    def compute_standardized_indicator_levels_in_france(self, standardized_indicator, save = True):
        """
        Populates self.counts_standardized_indicator_levels_departements
        Needed if you want to call plot_counts_france
        """
        if standardized_indicator not in ["spi", "spli"]:
            print("Error: standardized_indicator must be 'spi' (for rain level) or 'spli' (for the other indicators)")

        for departement_code in tqdm(self.data_departements.keys()):
            print(f"Traitement du département: {departement_code}")
            self.load_time_series_departement(departement_code)
            
            for indicator_name in self.dict_standardized_indicators[standardized_indicator].keys():
                print(f" Indicateur: {indicator_name}")
                self.counts_standardized_indicator_levels_departements[departement_code][indicator_name] = self.compute_standardized_indicator_levels_for_departement(standardized_indicator,
                                                                                                                                                                      departement_code,
                                                                                                                                                                      indicator_name)
            del self.dict_time_series_departement[departement_code]
        
        if save:
            with open(f"{self.data_folder}_{standardized_indicator}_counts_standardized_indicator_levels_departements.pkl", 'wb') as fp:
                pickle.dump(self.counts_standardized_indicator_levels_departements)
     

    def plot_counts_departement(self, departement_code, indicator_name):
        """
        Plots the BRGM representation of the proportions of stations of the given departement for each dryness levels month by month since 1 year
        Returns also the corresponding dataframe
        """
        #standardized_indicator = "spli" if indicator_name in self.dict_standardized_indicators["spli"].keys() else "spi"

        if self.counts_standardized_indicator_levels_departements[departement_code]:
            df_levels = pd.DataFrame(self.counts_standardized_indicator_levels_departements[departement_code][indicator_name]).transpose().reset_index()
            df_levels.columns = ["Mois"] + [x[0] for x in self.spi_levels.values()]
            df_levels["Mois"]= df_levels["Mois"].replace({  0: "Janvier",
                                                            1: "Février",
                                                            2: "Mars",
                                                            3: "Avril",
                                                            4: "Mai",
                                                            5: "Juin",
                                                            6: "Juillet",
                                                            7: "Août",
                                                            8: "Septembre",
                                                            9: "Octobre",
                                                            10: "Novembre",
                                                            11: "Décembre",
                                                            })
            # Shift rows
            df_levels = df_levels.reindex(index=np.roll(df_levels.index,12-(date.today().replace(day=1).month-1)))

            df_values = df_levels.drop("Mois", axis=1)
            df_values = df_values.div(df_values.sum(axis=1), axis=0)*100
            df_levels = pd.concat([df_levels["Mois"], df_values], axis=1)

            ax = df_levels.plot.bar(x='Mois',  stacked=True, title=f'Niveaux {self.dict_indic_names[indicator_name].lower()} {self.departements_names[departement_code]}', color=self.levels_colors)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(reversed(handles), reversed(labels), loc='upper left',bbox_to_anchor=(1.0, 0.5))


    def plot_counts_france(self, indicator_name):
        """
        Plots the BRGM representation of the proportions of stations in France for each dryness levels month by month since 1 year.
        Returns also the corresponding dataframe
        """
        df_levels = pd.DataFrame(self.counts_standardized_indicator_levels_france[indicator_name]).transpose().reset_index()
        df_levels.columns = ["Mois"] + [x[0] for x in self.spi_levels.values()]
        df_levels["Mois"]= df_levels["Mois"].replace({  0: "Janvier",
                                                        1: "Février",
                                                        2: "Mars",
                                                        3: "Avril",
                                                        4: "Mai",
                                                        5: "Juin",
                                                        6: "Juillet",
                                                        7: "Août",
                                                        8: "Septembre",
                                                        9: "Octobre",
                                                        10: "Novembre",
                                                        11: "Décembre",
                                                        })
        # Shift rows
        df_levels = df_levels.reindex(index=np.roll(df_levels.index,12-(date.today().replace(day=1).month-1)))

        df_values = df_levels.drop("Mois", axis=1)
        df_values = df_values.div(df_values.sum(axis=1), axis=0)*100
        df_levels = pd.concat([df_levels["Mois"], df_values], axis=1)

        ax = df_levels.plot.bar(x='Mois',  stacked=True, title=f'Niveaux {self.dict_indic_names[indicator_name].lower()} en France', color=self.levels_colors)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles), reversed(labels), loc='upper left',bbox_to_anchor=(1.0, 0.5))
        return df_levels
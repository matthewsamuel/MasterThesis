# -*- coding: utf-8 -*-
"""
Created on Jan 2021

Abstract model evaluation framework components. Provides abstract definitions 
for the classes that will be used in value-based model evaluation.

Multi threading library to perform data valuation tasks, including:
    - Control of evaluation buffer, which stores any evaluation done by the
    model and prevents redoing such evaluation again in the future.
    - Calculation of payoff distribution drivers. Returns a pd.Series whose
    index is the set of sources N and whose values is the driver that will be
    used to calculate the payoffs and rewards.
        - Value-based data valuation: requires a "model" for which data will be
        used + a valuation metric to maximize (e.g. accuracy)
            * Individual value
            * LOO value
            * Shapley value (according to different algorithms):
                * Raw exact SV
                * (Truncated) Monte Carlo
                * (Truncated) Random Sampling
                * (Truncated) Structured Sampling
        - Other more simple yet useful methods to calculate the value of data:
            * Metric-based data valuation (e.g. number of registers, data 
            volume...) - provides the driver to calculate payoffs as a metric
            that is directly calculated from the different datasets.                                    
    
    - Payoff / reward distribution according to a certain driver
    
@author: santi
Adapted by @MatthewSamuel, June 2025. 
Changes include integration of Kriging interpolation and data valuation via Shapley values using Truncated Structured Sampling (TSS), for master's thesis purposes.
Under supervision from Santiago Andres Azcoitia and Jorge Garcia Cabeza
"""

import datetime
import math
import random
import pandas as pd
import numpy as np
from numpy.linalg import inv
import csv
from math import sin, cos, radians, sqrt, atan2, tan
import seaborn as sn
import matplotlib.pyplot as plt
import time
import numpy as np
from numpy.linalg import inv, pinv  
import os
from tqdm import tqdm
import config



""" 
Support functions
_______________________________________________________________________________

"""

# Creates a balanced latin square of order n
# See https://medium.com/@graycoding/balanced-latin-squares-in-python-2c3aa6ec95b9




   
def latin_squares(n):

    l = [[((j//2+1 if j%2 else n-j//2) + i) % n + 1 \
          for j in range(n)] for i in range(n)]

    #if n % 2:  # Repeat reversed for odd n

    #    l += [seq[::-1] for seq in l]

    return l


""" def getRandomUniformPermutations(self, N, r):
        
        #Wrapper voor de globale getRandomUniformPermutations functie.
        
        return getRandomUniformPermutations(N, r)  # Roep de bestaande functie aan

# The following function returns r * N permutations of the tuple N. Each
# element of the tuple will appear r times in each position of the permutations
# in the output permutation set dfPermutations.
def getRandomUniformPermutations(N, r):
        
    # Creates rxN random uniform permutations using a Latin Square of size N
    L = latin_squares(len(N))
    Q = np.random.permutation(N).tolist()
    dfPermutations = pd.Series()

    i = 0
    for round in list(range(r)):
        random.shuffle(Q)
        for Li in L:
            R = []
            for j in Li:
                R = R + [Q[j-1]]
            #print(R)
            dfPermutations.at[i] = tuple(R)
            i=i+1
        
    # print(dfPermutations)
    return dfPermutations """

# The following function returns r * N random permutations of the tuple N. 
# The output of the permutation is a pandas.Series dfPermutations containing
# all the generated permutations
def getRandomPermutations(N, r):
    dfPermutations = pd.Series()
    
    i = 0
    
    # Creates rxN random permutations
    while i < r*len(N):
        Pi = np.random.permutation(N).tolist()
        dfPermutations.at[i] = Pi
        i=i+1
        
    print(dfPermutations)
    return dfPermutations


# The following function returns r * len(N) permutations of the tuple N using a Balanced Latin Square
# so that each element appears equally often in each position.
def getRandomUniformPermutations(N, r):
    """
    Creates r * len(N) permutations of the tuple N using a Balanced Latin Square
    so that each element appears equally often in each position.
    """
    # Create Balanced Latin Square
    L = latin_squares(len(N))
    # Make a mutable copy of N for shuffling
    Q = list(N)
    dfPermutations = pd.Series(dtype=object)
    idx = 0
    for _ in range(r):
        random.shuffle(Q)
        for Li in L:
            # Build a tuple according to the Latin square indices
            perm = tuple(Q[j-1] for j in Li)
            dfPermutations.at[idx] = perm
            idx += 1
    return dfPermutations


# Returns a set with all possible combinations of iterable taken r by r
def getCoalitions(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

""" 

Class DataCombination

Abstract model class including the methods that need to be defined to combine
different data sources into one specific input y to be introduced to a model
or algorithm.

"""
class vbdeDataCombination:
    
    
    # Get a combined input for the set K of sources
    def getCombinedInput(self, K): pass
    # Gets the set of sources
    def getSources(self): pass




class KrigingDataCombination(vbdeDataCombination):
    def __init__(self, input_path, missing_geohashes, sel_freq):
        """
        Initialize data loading, variogram parameters, and combine datasets.
        
        Args:
            input_path (str): Path to the input data directory.
            missing_geohashes (str): Path to file with missing geohashes.
            sel_freq (str): Selected frequency band (e.g., '1800').
        """
        # Load datasets
        self.df1 = pd.read_csv(missing_geohashes, index_col=0)
        self.df2 = pd.read_csv(input_path)
        
        # Prepare df2 (group by geohash and filter)
        self.df2 = (
            self.df2.groupby(["gps_geohash_08", "gps_ghash_lon", "gps_ghash_lat"])
            .agg(rsrp_mean=("rsrp", "mean"), events=("rsrp", "count"))
            .reset_index()
            .rename(columns={
                "gps_geohash_08": "geohash",
                "gps_ghash_lon": "longitude",
                "gps_ghash_lat": "latitude",
                "rsrp_mean": "value"
            })
        )
        self.df2 = self.df2[self.df2["events"] > 0]  # Remove geohashes without measurements
        
        # Add dummy column for cross-join
        self.df1["dummy"] = "All"
        self.df2["dummy"] = "All"
        
        # Merge datasets (cross-join)
        self.combined = pd.merge(
            self.df1, 
            self.df2, 
            how="left",
            on="dummy", 
            suffixes=("_x", "_y")
        )
        
        # Calculate distances between all combinations
        self.combined["distance"] = self.combined.apply(
            lambda row: self.distance_btw_2_coord(
                row["latitude_x"], row["latitude_y"],
                row["longitude_x"], row["longitude_y"]
            ), 
            axis=1
        )
        
        # Variogram parameters 
        self.variogram_params = {
            "LTE-800": {"R": 373, "alpha": 5.3},
            "LTE-1800": {"R": 100, "alpha": 4.6},  
            "LTE-2100": {"R": 351, "alpha": 3.6},
            "LTE-2600": {"R": 335, "alpha": 3.2}
        }
        self.sel_freq = f"LTE-{sel_freq}"
        self.R = self.variogram_params[self.sel_freq]["R"]
        self.alpha = self.variogram_params[self.sel_freq]["alpha"]
        
        # Filter combinations within variogram range
        self.combined = self.combined[self.combined["distance"] <= self.R]
        self.combined = self.add_distance_rank(self.combined)

        # Variogram parameters
        self.C0 = 5                                 # Nugget
        self.C1 = self.df2["value"].var()           # Sill
        self.uncertainty = self.df2["value"].var()  # Measurement variance

        self.geohash_to_index = {gh: idx for idx, gh in enumerate(self.df2["geohash"])}
        
        # Bouw correlatiematrix in constructor
        self.build_correlation_matrix()

    def add_distance_rank(self, df):
        """Voeg subindex toe per geohash_x gesorteerd op afstand en geohash_y"""
        # Sorteer eerst op geohash_x, dan op distance, dan op geohash_y
        df_sorted = df.sort_values(['geohash_x', 'distance', 'geohash_y'])
        
        # Maak subindex die oploopt per geohash_x groep
        df_sorted['subindex'] = df_sorted.groupby('geohash_x').cumcount() + 1
        
        return df_sorted

    def build_correlation_matrix(self):
        """Bouw de correlatiematrix tussen alle buurgeohashes."""
        # Extract unique neighbors
        neighbors = self.df2[["geohash", "latitude", "longitude"]].drop_duplicates()
        
        # Create all combinations
        cross_joined = pd.merge(neighbors, neighbors, how="cross", suffixes=("_x", "_y"))
        
        # Calculate distances
        cross_joined["distance"] = cross_joined.apply(
            lambda row: self.distance_btw_2_coord(
                row["latitude_x"], row["latitude_y"],
                row["longitude_x"], row["longitude_y"]
            ), 
            axis=1
        )
        
        # Apply semivariogramcross_joined
        cross_joined["correlation"] = cross_joined["distance"].apply(lambda d: self.sv(d))        
        # Reshape to matrix
        n = len(neighbors)
        self.correlation_matrix = cross_joined["correlation"].values.reshape((n, n))

    def sv(self, distance):
        """Semivariogram functie."""
        return self.C0 + (self.C1 * (1 - np.exp(-self.alpha * distance / self.R)))

    def get_neighbors(self, target_geohash):
        """Retourneer buren voor een target, gesorteerd op afstand + subindex."""
        target_df = self.combined[self.combined["geohash_x"] == target_geohash]
        sorted_df = target_df.sort_values("distance").reset_index(drop=True)
        sorted_df["subindex"] = range(1, len(sorted_df) + 1)  # Subindex toewijzen
        return sorted_df  # DataFrame met kolommen [geohash_y, distance, subindex, ...]

    def get_variogram_params(self):
        return {
            "C0": self.C0,
            "C1": self.C1,
            "alpha": self.alpha,
            "R": self.R,
            "uncertainty": self.uncertainty
        }

 
    def getCombinedInput(self, target_geohash, K_subindices):
        """
        Voor Shapley: Geeft een subset van buren + bijbehorende covariantie.

        Args:
            target_geohash (str): Target geohash (bijv. 'ezjqh16e').
            K_subindices (tuple): Subindexen van de te gebruiken buren (bijv. (1, 3, 5)).
        
        Returns:
            dict: {
                'target': (lon, lat),
                'neighbors': [{'geohash': ..., 'value': ..., 'distance': ...}, ...],
                'covariance_subset': np.array  # Subset van de globale matrix
            }
        """
        # Filter buren op target en subindex
        target_neighbors = self.combined[self.combined['geohash_x'] == target_geohash]
        selected = target_neighbors[target_neighbors['subindex'].isin(K_subindices)].sort_values('subindex')
        
        # Haal indices voor covariantiesubset
        neighbor_geohashes = selected['geohash_y'].tolist()
        indices = [self.geohash_to_index[gh] for gh in neighbor_geohashes]  # Vereist geohash_to_index dict
        
        return {
            'target': (selected['longitude_x'].iloc[0], selected['latitude_x'].iloc[0]),
            'neighbors': selected[['geohash_y', 'value', 'distance','events']].to_dict('records'),
            'covariance_subset': self.correlation_matrix[np.ix_(indices, indices)]
        }
    
    def distance_btw_2_coord(self, lat1, lat2, lon1, lon2):
        R = 6373.0 * 1000  # Earth radius in meters
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c

    def getGlobalCovariance(self):
        """Retourneert de volledige covariantiematrix voor alle bekende geohashes."""
        return self.correlation_matrix

    def getSources(self, target_geohash):
        """
        Retourneert alle subindexen van de target geohash en print ze mooi uit.
        
        Args:
            target_geohash (str): De geohash waarvoor subindexen gezocht worden.
        
        Returns:
            list: Lijst van subindexen (int) gesorteerd op afstand.
        """
        # Filter de gecombineerde data voor de target geohash
        target_data = self.combined[self.combined['geohash_x'] == target_geohash]
        
        if target_data.empty:
            print(f"Geen data gevonden voor geohash: {target_geohash}")
            return []
        
        # Haal de subindexen op en sorteer ze
        subindices = sorted(target_data['subindex'].tolist())
        
        #print(f"Subindices for geohash '{target_geohash}': {', '.join(map(str, subindices))}")
        return subindices

    def data(self):
        """Final combined dataset for kriging."""
        return self.combined
    
""" 

Class Model

Abstract model class including the methods that need to be defined to calculate
the output from the inputs defined by the DataCombination module.

"""
class vbdeModel:
    def getModelOutput(self, x): pass

class KrigingModel(vbdeModel):
    def __init__(self, combiner, target_geohash, use_shapley_mode=False):
        """
        Initialize the Kriging model with combiner.
        
        Args:
            combiner (KrigingDataCombination): Initialized data combiner
        """
        self.combiner = combiner
        self.target_gh = target_geohash
        self.use_shapley_mode = use_shapley_mode
        self.selected_subindices = None  # Alleen relevant voor Shapley
       
        params = combiner.get_variogram_params()
        self.uncertainty = params["uncertainty"]
        self.C0 = params["C0"]
        self.C1 = params["C1"]
        self.alpha = params["alpha"]
        self.R = params["R"]

        if not self.use_shapley_mode:
            self.neighbors_df = combiner.get_neighbors(target_geohash)
            neighbor_geohashes = self.neighbors_df["geohash_y"].tolist()
    
            # Gebruik geohash_to_index voor snelle lookups
            self.indices = [combiner.geohash_to_index[gh] for gh in neighbor_geohashes]
    
            # NP.IX_ IS CRUCIAAL HIER
            self.C = combiner.correlation_matrix[np.ix_(self.indices, self.indices)]
            self.D = np.array([self.sv(d) for d in self.neighbors_df["distance"]])
        else:
            # Shapley-modus: Uitgestelde initialisatie
            # ---------------------------------------
            self.neighbors_df = None  # Wordt ingesteld tijdens getModelOutput()
            self.C = None
            self.D = None


    def getModelOutput(self, x=None):
        try:
            if self.use_shapley_mode:
                # 1) Haal de subset op (x bevat wat getCombinedInput(...) teruggeeft)
                neighbor_info = x["neighbors"]  # list of dicts: {geohash, value, distance, events}
                neighbor_values = [d["value"] for d in neighbor_info]
                neighbor_events = [d["events"] for d in neighbor_info]
                C = x["covariance_subset"]
                D = np.array([self.sv(d["distance"]) for d in neighbor_info])

                # 2) Los kriging op met lokale variabelen
                weights, mu = self.solve_kriging_weights(C, D, neighbor_events)
                prediction = np.dot(weights, neighbor_values)
                variance = np.dot(weights, D) + mu

                return {
                    "geohash": self.target_gh,
                    "prediction": prediction,
                    "variance": variance
                }

            else:
                # Niet-shapley modus
                neighbor_values = self.neighbors_df["value"].values
                events = self.neighbors_df["events"].values
                C = self.C
                D = self.D
                weights, mu = self.solve_kriging_weights(C, D, events)
                prediction = np.dot(weights, neighbor_values)
                variance = np.dot(weights, D) + mu

                return {
                    "geohash": self.target_gh,
                    "prediction": prediction,
                    "variance": variance
                }
            
        except Exception as e:
            print(f"Kriging error: {str(e)}")
            raise

    def solve_kriging_weights(self, C, D, events):
        n = len(D)
        events = np.array(events, dtype=float)

        # voorkom deling door 0: vervang events=0 door 1? of raise?
        events[events == 0] = 1.0

        diag_correction = np.diag(self.uncertainty / events - self.C0)
        C_corrected = C - diag_correction

        C_ext = np.zeros((n+1, n+1))
        C_ext[:n, :n] = C_corrected
        C_ext[-1, :n] = 1.0
        C_ext[:n, -1] = 1.0

        D_ext = np.zeros(n+1)
        D_ext[:n] = D
        D_ext[-1] = 1.0

       
        try:
            solution = np.linalg.inv(C_ext) @ D_ext
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(C_ext) @ D_ext

        weights = solution[:n]
        mu = solution[-1]
        return weights, mu

    def sv(self, distance):
        
        return self.C0 + (self.C1 * (1 - np.exp(-self.alpha * distance / self.R)))

""" 

Class ValueFunction

Abstract model class including the methods that need to be defined to evaluate
the output from a model. Typically, it will be a similarity metric that maps


"""
class vbdeValueFunction:
    """  Attributes: 
            defaultValue, which is the value which will be used in case
        of any failure either in the model fitting or in the evaluation of the
        result from the model.
            yTest, which is the object against which the class compares the
        output of the model
    """
    
    
    def __init__ (self, i_yTest, i_defaultValue = 0):
        self.yTest = i_yTest
        self.defaultValue = i_defaultValue
       
    def evaluateOutput(self, y1): pass

class KrigingValueFunction(vbdeValueFunction):
    def __init__(self, ground_truth_path, i_defaultValue=0):
        """
        Initialize the evaluation function.
        
        Args:
            ground_truth_path (str): Path to the ground truth data.
        """
        super().__init__(i_yTest=None, i_defaultValue=i_defaultValue)
        self.ground_truth = pd.read_csv(ground_truth_path)
        self.ground_truth = self.ground_truth.rename(columns={
            "gps_geohash_08": "geohash",
            "mean_dbm": "value"
        })
        self.defaultValue = i_defaultValue
        
    def evaluateOutput(self, y_pred):
        #print("DEBUG y_pred before merge:", y_pred)
        # Zet de dict om naar DataFrame met één rij
        df_pred = pd.DataFrame([y_pred])  

        merged = pd.merge(df_pred, self.ground_truth, on="geohash")
        #print("DEBUG merged shape:", merged.shape)
        if merged.empty:
            print("!!! Merged is empty => defaultValue = 0")
            return self.defaultValue

        rmse = np.sqrt(((merged["value"] - merged["prediction"]) ** 2).mean())
        return -rmse

"""
    Abstract class vbdeFramework
"""
class vbdeFramework:
    # Global variables
    # Combiner, DataCombination instance used to combine data sources
    # Model, vbModel instance used to get the results
    # ValueFunction, vbValueFunction used to get the value y(s) brings to the model v(s)
    
    
    # yhat
    
    # Dataframe that stores the results of the model for each coalition of inputs.
    # S. It stores in each column s the set of outputs yhat(s) from the model 
    # when it is fed by the combination of inputs s
    #  The index will be the index of yTest. We do not force it to make it as
    # general as possible.
    
    # y
    
    # Dataframe that stores the combination of inputs for each coalition s.
    # It stores in each column s the set of inputs y(s) from the model 
    # for a combination of inputs s.
    #  The index will be the index of yTest. We do not force it to make it as
    # general as possible.
    
    #vS
    
    #  Pandas Series that stores the v(S) for a set of inputs S. vS[S] provides
    # the value of the output for a combination of inputs S.

    # queueS    

    #  Pandas Series that will store the list of tuples to be distributed among
    # workers.

    # To instantiate a Data Evaluation Framework it is necessary to provide:
    #   - a method to combine inputs
    #   - a model to work on
    #   - an evaluation method to get the value of outputs from the model
    
    def __init__(self, iCombiner, iModel, iValueFunction, i_vSbuffer = True):
       
        self.Combiner = iCombiner
        self.Model = iModel
        self.ValueFunction = iValueFunction
        self.isVSbuffered = i_vSbuffer

        self.yhat = pd.DataFrame(index = pd.Index([], dtype = object, name = 'yhat'))
        self.y = pd.DataFrame(index = pd.Index([], dtype = object, name = 'y'))
        self.vS = pd.Series(index = pd.Index([], dtype = object, name = 's'), name = 'v(S)')
        self.queueS = []
        
    # All the classes that inherit from vbdeFramework must define a "default"
    # constructor to use distributed processing
    # def __init__(self): pass
        
    """
        getValue(s)
        The function returns v(S) if S was already evaluated an N/A in case not
        s is assumed to be a tuple of sources whose value will be retrieved
        from the buffer.
        If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferValue(self, s):
        if self.isVSbuffered: 
            result = self.vS[str(s)] 
        else: 
            result = self.evaluateTuple(s)
        return result
    
    
    """
        getBufferYhat(s)
        The function returns yhat(S) if S was already evaluated an N/A in case not
        s is assumed to be a tuple of sources whose value will be retrieved
        from the buffer.
        If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferYhat(self, s):
        return self.yhat[str(s)]
    
    
    """
        getBufferY(s)
        The function returns v(S) if S was already evaluated an N/A in case not
        s is assumed to be a tuple of sources whose value will be retrieved
        from the buffer.
        If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferY(self, s):
        return self.y[str(s)]
    
    """
        Clear buffers of the framework.
    """
    def clearBuffer(self):
        self.yhat = pd.DataFrame(index = pd.Index([], dtype = object, name = 'yhat'))
        self.y = pd.DataFrame(index = pd.Index([], dtype = object, name = 'y'))
        self.vS = pd.Series(index = pd.Index([], dtype = object, name = 's'), name = 'v(S)')
        
    """
        getSources()
        The function returns the set of sources the model works with. It takes
        them from the Combiner
    """

    
    
    """
        The function executes the model and evaluates the output.
        It stores the results in the buffers of the framework for future uses:
            y, yhat and vS
        In case of any error, it returns the defaultValue provided by the 
        metric.
    """
    def evaluateTuple(self, K):
        # If K is the empty coalition, return default baseline value (0)
        if len(K) == 0:
            val_fn = self.ValueFunction.result() if hasattr(self.ValueFunction, 'result') else self.ValueFunction
            if self.isVSbuffered:
                self.vS.at[str(K)] = val_fn.defaultValue
            return val_fn.defaultValue
        # Unwrap Dask futures to actual objects if needed
        combiner = self.Combiner.result()       if hasattr(self.Combiner, 'result')       else self.Combiner
        model    = self.Model.result()          if hasattr(self.Model, 'result')          else self.Model
        val_fn   = self.ValueFunction.result()  if hasattr(self.ValueFunction, 'result')  else self.ValueFunction
        if self.isCalculatedYet(K):
            return self.getBufferValue(K)
        else:
            try:
                # Haal shapley-subset op (dictionary met neighbors, covariances, ...)
                yInput  = combiner.getCombinedInput(model.target_gh, K)
                yOutput = model.getModelOutput(yInput)
                v       = val_fn.evaluateOutput(yOutput)

                if math.isnan(v):
                    if self.isVSbuffered:
                        self.vS.at[str(K)] = val_fn.defaultValue
                    return val_fn.defaultValue
                else:
                    if self.isVSbuffered:
                        self.vS.at[str(K)] = v
                    return v
            except:
                if self.isVSbuffered:
                    self.vS.at[str(K)] = val_fn.defaultValue
                return val_fn.defaultValue
            
        
    """
        The following function returns a boolean telling whether there is a
        pending work to process the coalition s. s is
        assumed to be a tuple of sources whose value will be calculated.
    """
    def isCalculatedYet(self, s):
        if self.isVSbuffered:
            return str(s) in self.vS.index
        else:
            # If the buffer is not maintained, then it is never calculated
            #and the framework is forced to ask the model for the v(s)
            return False
    
    """
        addWork
         This procedure allows to include a work to the queue of works to be
        distributed
    """
    def addWork(self, s):
        if not self.isCalculatedYet(s):
            if not s in self.queueS:
                self.queueS.append(s)
                return 1
        return 0
                
    """
        executeWorks
         The following procedure schedules the existing works among nWorkers
        in config.daskClient. It executes the works to calculate the v(S) for all
        S in queue S. It distributes the work among nWorkers. It saves the schedule 
        to the worker schedule and starts the workers. Finally it waits for them 
        to finish the works and consolidate their results.
         The implementation is simple, it just distributes the workload among
        a number of nWorkers which is an input from the callee.
    """
    def executeWorks(self, daskClient, iWorkerProcess_init, nWorkers):
        # Ensure worker directory exists
        os.makedirs('worker', exist_ok=True)
        # Build schedule dataframe
        schedule = pd.DataFrame(columns=list(range(nWorkers)))
        i = 0
        jWorker = 0
        existingWorks = []
        for work in self.queueS:
            if work not in existingWorks:
                existingWorks.append(work)
                schedule.at[i, jWorker] = work
                if jWorker < nWorkers - 1:
                    jWorker += 1
                else:
                    jWorker = 0
                    i += 1
        # Write per-worker schedule files
        for worker in schedule.columns:
            schedule[worker].to_csv(f'worker/worker_{worker}_schedule.csv', index=False)
        # Submit tasks to Dask
        futures = []
        for worker in schedule.columns:
            futures.append(daskClient.submit(
                iWorkerProcess_init,
                worker,
                self.Combiner,
                self.Model,
                self.ValueFunction
            ))
        # Wait for completion
        daskClient.gather(futures)
        # Consolidate results and clear queue
        self.consolidateWorkerFiles(nWorkers)
        self.queueS = []

    def WorkerProcess_init(self, i, combiner, model, value_fn):
        # Initialize a fresh framework for this worker
        worker_fw = vbdeFramework(combiner, model, value_fn, i_vSbuffer=True)
        # Load this worker's schedule
        schedule_df = pd.read_csv(os.path.join('worker', f'worker_{i}_schedule.csv'))
        tasks = schedule_df[str(i)].dropna().tolist()
        # Count number of executed tasks
        executed = 0
        # Execute each task
        for s in tasks:
            if isinstance(s, str) and s.startswith('('):
                try:
                    worker_fw.evaluateTuple(eval(s))
                    executed += 1
                except Exception:
                    continue
        # Export partial results
        try:
            worker_fw.vS.to_csv(os.path.join('worker', f'worker_{i}_output_vS.csv'), index=True, header=True)
        except Exception:
            pass
        try:
            worker_fw.yhat.to_csv(os.path.join('worker', f'worker_{i}_output_yhat.csv'), index=True, header=True)
        except Exception:
            pass
        return executed

    """
        getIndividualValues
        
        Given a set of sources and a set of training examples and information,
        it trains the model individually with the data of every source i and 
        obtains the results and its valuation.    
    """
    def getIndividualValues(self, N):
        
        # Result is a pd.Series object including all the valuations
        result = pd.Series(index = N)
        
        for source in N:
            if not self.isCalculatedYet((source,)):
                result[source] = self.evaluateTuple((source,))
            else:
                result[source] = self.getBufferValue((source,))
        
        # returns the result, which after exitting the loops should contain all
        # the individual values.
        return result


    """
        getIndividualValuesParallel
        
        Given a set of sources and a set of training examples and information,
        it trains the model individually with the data of every source i and 
        obtains the results and its valuation. Implementation is done using
        parallel workers to calculate all the individual values.
    """
    
    def getIndividualValuesParallel(self, daskClient, iWorkerProcess_init, N):   
        # Result is a pd.Series object including all the LOO values
        result = pd.Series(index = N)
                  
        # Add works
        for source in N:
            self.addWork((source,))
        
        nWorkers = len(config.daskClient.cluster.workers)
        self.executeWorks(daskClient, iWorkerProcess_init, nWorkers)
            
        # Calculates LOOi (vS should be updated)
        for source in N:
            vi = self.getBufferValue((source,))
                    
            result.at[source] = vi
        
        # returns the result, which after exitting the loop should contain all
        # the individual values.
        return result
    
    
    """
        getLOOValues
        
        Given a set of sources and a set of training examples and information,
        it obtains the LOOi = v(N) - v(N- {i}). The implementation uses a
        single thread.
        
        The function assumes that v(S) is independent of the order of S
    """
    def getLOOValues(self, N):  
        N = tuple(sorted(N))    
        
        # Result is a pd.Series object including all the LOO values
        result = pd.Series(index = N)
                  
        # Calculates vN
        if self.isCalculatedYet(N):
            vN = self.getBufferValue(N)
        else:
            vN = self.evaluateTuple(N)
        
        for source in N:
            N_i = tuple(x for x in N if not x == source)
            
            # Calculates vN_i
            if self.isCalculatedYet(N_i):
                vN_i = self.getBufferValue(N)
            else:
                vN_i = self.evaluateTuple(N_i)
            
            result[source] = vN - vN_i
        
        # returns the result, which after exitting the loop should contain all
        # the individual values.
        return result


    """
        getLOOValues
        
        Given a set of sources and a set of training examples and information,
        it obtains the LOOi = v(N) - v(N- {i}). The implementation distributes 
        the workload among a set of workers and retrieves the results from
        them at the end of the executions.
        
        The function assumes that v(S) is independent of the order of S
    """
    def getLOOValuesParallel(self, daskClient, iWorkerProcess_init, N):
        # Result is a pd.Series object including all the LOO values
        result = pd.Series(index = N)
                  
        # Add works
        self.addWork(tuple(sorted(N)))
        for source in N:
            N_i = tuple(x for x in N if not x == source)
            self.addWork(tuple(sorted(N_i)))
        
        nWorkers = len(config.daskClient.cluster.workers)
        self.executeWorks(daskClient, iWorkerProcess_init, nWorkers)
            
        # Calculates LOOi (vS should be updated)
        vN = self.getBufferValue(N)
        for source in N:
            N_i = tuple(x for x in N if not x == source)
            vN_i = self.getBufferValue(N_i)
                    
            result.at[source] = vN - vN_i
        
        # returns the result, which after exitting the loop should contain all
        # the individual values.
        return result


    """
    ______________________________________________________________________________
    
    IMPORT / EXPORT functions allow for storing and retrieving from disk the
    main buffers used in the Framework model:
        vS
        y
        yhat
        
     Another relevant function allows to consolidate the partial results from
    workers into the existing memory.
        
    
    importVSFile()
    The following function imports an existing vS file 
    """
    def importVSFile(self, name):
        vStemp = pd.read_csv(name)
        vStemp.columns = ['s', 'v(S)']
        vStemp = vStemp.set_index('s')
        # Exclude empty (NaN) values
        new_vS = vStemp['v(S)'].dropna()

        # Build list of non-empty series to concatenate
        parts = []
        if not self.vS.empty:
            parts.append(self.vS)
        if not new_vS.empty:
            parts.append(new_vS)

        # Update only if there is at least one non-empty series
        if parts:
            # If there's just one, no need to concat
            self.vS = parts[0] if len(parts) == 1 else pd.concat(parts)
    
    
    
    """
    exportVSFile()
    The following function exports a vS file to the 
    """
    def exportVSFile(self, name):
        self.vS.to_csv(name, index = True, header = True)
    
    """
    importYFile()
    The following function imports an existing vS file 
    """
    def importYFile(self, name):        
        ytemp = pd.read_csv(name)
        ytempcols = ytemp.columns.values
        ytempcols[0] = 'y'
        ytemp.columns = ytempcols
        ytemp = ytemp.set_index('y')
        for col in ytemp.columns:
            self.y[col] = ytemp[col]
    
    
    
    """
    exportYFile()
    The following function exports a vS file to the 
    """
    def exportYFile(self, name):
        self.y.to_csv(name, index = True, header = True)
    
    
    """
    importYhatFile()
    The following function imports an existing vS file 
    """
    def importYhatFile(self, name):
        global yhat
        
        yhattemp = pd.read_csv(name)
        yhattempcols = yhattemp.columns.values
        yhattempcols[0] = 'yhat'
        yhattemp.columns = yhattempcols
        yhattemp = yhattemp.set_index('yhat')
        for col in yhattemp.columns:
            self.yhat[col] = yhattemp[col]
    
    
    
    """
    exportYhatFile()
    The following function exports a vS file to the 
    """
    def exportYhatFile(self, name):
        self.yhat.to_csv(name, index = True, header = True)
    
    
    
    """
    consolidateWorkersResults()
    Retrieves the results from workers output files and uploads it to the
    existing framework buffers: vS, y and yhat.
    """
    def consolidateWorkerFiles(self, nWorkers):
        # Consolidate each worker's partial results into the main buffers
        for i in range(nWorkers):
            # Import vS results
            self.importVSFile(os.path.join('worker', f'worker_{i}_output_vS.csv'))
            # Import yhat results
            self.importYhatFile(os.path.join('worker', f'worker_{i}_output_yhat.csv'))
            # (Optional) Import y results if used
            # self.importYFile(os.path.join('worker', f'worker_{i}_output_y.csv'))
    
    
    """
    processWorker(i)
    
     The following code will start a worker j which will perform all
    calculations stated in 'worker j schedule.csv', that contains all the
    tuples that the worker needs to resolve, and export all the results to
    the following files:
        * 'worker j output vS.csv', two columns: tuple and v(S)
        * 'worker j output yhat.csv', which has one index column + one
        column for each of the tuples s analyzed including the results of the
        model after using the combined input of the sources in s.
        * 'worker j output y.csv', which has one index column + one
        column for each of the tuples s analyzed including the combined input
        of the sources in s.
    
     The format of the input file is a csv with only one column which
    specifies the list of tuples that the worker must process.
    
     The process assumes that the vbdeFramework has been previously initialized
    def processWorker(i):
        try:
            # Retrieve the schedule of tasks to perform:
            taskImport = (pd.read_csv('worker '+str(i)+' schedule.csv'))[str(i)]
            
            # Perform the tasks
            for s in taskImport.values:
                if str(s)[0] == '(':
                    evaluateTuple(eval(s)) # Automatically updates vS and yhat
            
            # Exports partial results
            vS.to_csv('worker '+str(i)+' output vS.csv', 
                      index = True, header = True)
            #y.to_csv('worker '+str(i)+' output y.csv', 
            #          index = True, header = True)
            yhat.to_csv('worker '+str(i)+' output yhat.csv', 
                      index = True, header = True)
            
            return(1)
        except:
           return(0)
    """
    
    """
    ------ Truncated  Structured Sampling SV approximation  -------
    
        Calculates an approximation to Shapley Value for coalitions of players 
        of a set n based on a structured sampling (TSS) algorithm described as:
    
            Initializes SV(i) = 0.0
            P = set of N*r random uniform permutations of n=(p1, ..., pN)
            t = 0
            for each P
                t = t + 1
                vS = 0
                for each j in P
                    if vS <= Truncation value
                        vSuj = v(P[0:j])
                    else
                        vSuj = vS
                    SV(Pj) = t-1/t * SV(i) + 1/t * (vSuj - vS)
                    vS = vSuj
                        
               
        We define the following set of parameters for the algorithm:
            * r: rounds of permutations to evaluate. r >= 1
            * TruncationValue: If this value is reached, it is supposed that 
            the remaining elements in the coalition have zero marginal value
            
        Apart from the TSS algorithm parameters, the function requires the 
        following inputs:
        * District c for which to calculate SV
        * Set of players n
        * Demand functions in the observation period for each player (yO)
        * Demand functions in the control period for each player (yC)
        * Temporary series of already calculated vS to speed up calculation 
            -> it will be returned back by the function
        * Temporary series of yPredicted functions 
            -> it will be returned back by the function
        
        The function assumes that v(S) is independent of the order of S
        
        The function returns the following data:
        * dfPermutations,
        * SVpred, 
        * Number of executions done
    """

    def getRandomUniformPermutations(self, N, r):
            """
            Genereer uniforme permutaties met Latin Squares.
            """
            L = latin_squares(len(N))
            Q = np.random.permutation(N).tolist()
            dfPermutations = pd.Series()

            i = 0
            for _ in range(r):
                random.shuffle(Q)
                for Li in L:
                    R = [Q[j-1] for j in Li]
                    dfPermutations.at[i] = tuple(R)
                    i += 1
            return dfPermutations


    # Single thearded version of the function
    # The function assumes that v(S) is independent of the order of S
    # Takes as inputs
    def getSV_TSS(self, N, r, TruncationValue,debug=False):
        """
        Bereken Shapley-waarden met Truncated Structured Sampling.
        
        Args:
            N (list): Lijst van spelers (geohashes).
            r (int): Aantal permutatierondes.
            TruncationValue (float): Afkapwaarde voor marginale bijdragen.
        
        Returns:
            SVpred (pd.Series): Shapley-waarden per speler.
            dfPermutations (pd.Series): Gebruikte permutaties.
            nExecs (int): Aantal uitgevoerde berekeningen.
        """
        # Initialiseer Shapley-waarden
        SVpred = pd.Series([0.0] * len(N), index=N, name="SV TSS")
        nExecs = 0
        
        dfPermutations = self.getRandomUniformPermutations(N, r)
        
        for i, P in dfPermutations.items():
            vSeval = TruncationValue - 0.000001
            for j, player in enumerate(P):
                coalition = tuple(sorted(P[:j+1]))
                
                if vSeval < TruncationValue:
                    vSui = self.evaluateTuple(coalition)
                    nExecs += 1
                    if debug:
                         print(f"v({coalition}) = {vSui}")  # Debug
                else:
                    vSui = vSeval
                
                SVpred[player] = (i / (i + 1)) * SVpred[player] + (vSui - vSeval) / (i + 1)
                vSeval = vSui
        
        return SVpred, dfPermutations, nExecs
        
    # Parallel execution of getTSS. It works in two steps:
    #  1: Calculation of the necessary v(S). It takes into consideration the
    # truncation value that is passed as an input to the function. 
    #  2: Approximation to SV - which is a simple calculation once all the v(S) are
    # in the buffer.
    # The function assumes that v(S) is independent of the order of S
    # You have to pass as an input the district c for which this is computed
    def getTSS_parallel(self, daskClient, iWorkerProcess_init, N, r, TruncationValue):
        # prints out some information about the execution
        #print("TSS parallel calculation for "+str(len(N))+" companies in district "+str(c))
            
        dfPermutations = getRandomUniformPermutations(N, r)
        
        return self.processPermutations_parallel(daskClient, \
                                            iWorkerProcess_init, N, r, \
                                            TruncationValue, dfPermutations)
    
    
    #  Parallel processing of a set of permutations for sampling algorithms. It
    # can work both with TSS and TRS algorithms.
    #  It works in two steps:
    #  1: Calculation of the necessary v(S). It takes into consideration the
    # truncation value that is passed as an input to the function. It works by
    # processing coalitions up to a certain position i, and calculates v(S) if 
    # and only if v(Si-1) < Truncation_value.
    #  2: Approximation to SV - which is a simple calculation once all the v(S)
    # are in the buffer.
    # The function assumes that v(S) is independent of the order of S
    def processPermutations_parallel(self, daskClient, iWorkerProcess_init, N,\
                                        r, TruncationValue, dfPermutations):
            # Counter of the number of executions done
            nExecs = 0
            # Series which stores the last value for permutation i, initialized just below TruncationValue
            epsilon = 1e-6
            init_val = TruncationValue - epsilon
            vSeval = pd.Series([init_val] * (r * len(N)))
            # Initializes Existing Works
            existingWorks = True
            # Chunk size: group this many positions per Dask task
            batch_size = 1
            pos = 0
            # Initialize progress bar over chunks
            total_chunks = (len(N) + batch_size - 1) // batch_size
            pbar = tqdm(total=total_chunks, desc="Shapley progress", unit="chunk")
            # Track chunk timings for ETA
            durations = []
            completed = 0
            # Process positions in chunks of size batch_size
            while pos < len(N):
                chunk_start = time.time()
                existingWorks = False
                # 1st phase: schedule all works for this chunk of positions
                for j in range(pos, min(pos + batch_size, len(N))):
                    for i in range(r * len(N)):
                        Sui = tuple(sorted(dfPermutations.at[i][0:j+1]))
                        if vSeval[i] <= TruncationValue:
                            if self.isCalculatedYet(Sui):
                                _ = self.getBufferValue(Sui)
                            else:
                                nExecs += self.addWork(Sui)
                                existingWorks = True
                # Execute chunk if needed
                if existingWorks:
                    nWorkers = len(daskClient.cluster.workers)
                    self.executeWorks(daskClient, iWorkerProcess_init, nWorkers)
                    # 2nd phase: update vSeval for this chunk
                    for j in range(pos, min(pos + batch_size, len(N))):
                        for i in range(r * len(N)):
                            Sui = tuple(sorted(dfPermutations.at[i][0:j+1]))
                            if vSeval[i] <= TruncationValue:
                                vSeval.at[i] = self.getBufferValue(Sui)
                # Compute actual duration and update custom ETA
                duration = time.time() - chunk_start
                durations.append(duration)
                completed += 1
                avg = sum(durations) / len(durations)
                remaining = total_chunks - completed
                eta_seconds = int(avg * remaining)
                eta_str = str(datetime.timedelta(seconds=eta_seconds))
                pbar.set_postfix({"avg": f"{avg:.1f}s", "ETA": eta_str})
                pos += batch_size
                pbar.update(1)
            pbar.close()
            # The former loops ensure that all the required works are executed
            # We will now proceed to calculate the SV approximation
            t = 0
            i = 0
            
            # Initializes the output: pd.Series containing the Shapley values
            SVpred = pd.Series([0.0]*len(N), N, name = "SV TSS")
            
            while i<r*len(N):
                P = dfPermutations.at[i]
                t = t + 1
                # processes permutation i - DEBUG
                # print("Processing permutation "+str(i)+" - "+str(P))
                # Initialize vSeval just below the truncation threshold for this permutation
                epsilon = 1e-6
                vSeval = TruncationValue - epsilon
                j = 0
                while j < len(P):
                    Sui = tuple(sorted(P[0:j+1]))

                    # Check if vS has reached the truncation value
                    # Only process v(Sui) if v(S) is below the truncation value
                    if vSeval <= TruncationValue:
                        # This is just a search in the v(S) database now
                        vSui = self.getBufferValue(Sui)
                    else:
                        vSui = vSeval
                    
                    #Updates temporary SV of player i
                    SVpred[P[j]] = t/(t+1) * SVpred[P[j]] + (vSui-vSeval)/(t+1)
                    
                    vSeval = vSui
                    
                    j += 1    # Next position in the permutations
                
                #debug
                #print("SVpred after "+str(t)+" executions = \n" + str(SVpred))
                
                i += 1   # Next permutation          
                
                
            # Returning results and information about the execution
            #  SVpred - the calculated SV by using those sample permutations
            #  dfPermutations - set of permutations processed
            #  nExecs - number of executions performed
            return SVpred, dfPermutations, nExecs
        

        
    
    """
    ------ Truncated Random Sampling SV approximation  -------
    
        Calculates an approximation to Shapley Value for coalitions of players 
        of a set n based on a random sampling (TRS) algorithm described as:
    
            Initializes SV(i) = 0.0
            P = set of N*r random permutations of n=(p1, ..., pN)
            t = 0
            for each P
                t = t + 1
                vS = 0
                for each j in P
                    if vS <= Truncation value
                        vSuj = v(P[0:j])
                    else
                        vSuj = vS
                    SV(Pj) = t-1/t * SV(i) + 1/t * (vSuj - vS)
                    vS = vSuj
                        
               
        We define the following set of parameters for the algorithm:
            * r: rounds of permutations to evaluate. r >= 1
            * TruncationValue: If this value is reached, it is supposed that 
            the remaining elements in the coalition have zero marginal value
            
        Apart from the TSS algorithm parameters, the function requires the 
        following inputs:
        * District c for which to calculate SV
        * Set of players n
        * Demand functions in the observation period for each player (yO)
        * Demand functions in the control period for each player (yC)
        * Temporary series of already calculated vS to speed up calculation 
            -> it will be returned back by the function
        * Temporary series of yPredicted functions 
            -> it will be returned back by the function
        
        The function assumes that v(S) is independent of the order of S
        
        The function returns the following data:
        * dfPermutations,
        * SVpred, 
        * Number of executions done
    """
    # Single thearded version of the function
    # The function assumes that v(S) is independent of the order of S
    # Takes as inputs
    def getSV_TRS(self, N, r, TruncationValue):
        
        # Initializes the output: pd.Series containing the Shapley values
        SVpred = pd.Series([0.0]*len(N), N, name = "SV TSS")
        
        # Dataframe that stores the information about executions, namely:
        #  1. dfPermutations : the permutations processed
        #  2. nExecs: number of executions until the threshold is reached
        #  3. dfvSVector: vPj
        #  4. dfSVectors: Temporary SV vector
        #  5. dfRideVectors: Temporary Ride Vectors for each permutation
        nExecs = 0
        
        # prints out some information about the execution
        print("SV calculation for "+str(len(N))+" companies: \n")
            
        dfPermutations = getRandomPermutations(N, r)
        
        # Structured sampling loop - evaluating SV for each player in each permutation
        # The improvement allows to save in v(S) calculations, since we evaluate the
        # uniform shuffles we created before
        t = 0
        i = 0
        while i<r*len(N):
            P = dfPermutations.at[i]
            t = t+1
            # processes permutation i
            #print("Processing permutation "+str(i)+" - "+str(P))
            vSeval = 0
            j = 0
            while j < len(P):
                Sui = tuple(sorted(P[0:j+1]))
    
                # Check if vS has reached the truncation value
                # Only process v(Sui) if v(S) is below the truncation value
                if vSeval <= TruncationValue:
                    if self.isCalculatedYet(Sui):
                        #Tries to get vSPred from memory
                        vSui = self.getBufferValue(Sui)
                    else:
                        vSui = self.evaluateTuple(Sui)
                        nExecs = nExecs+1
                else:
                    # If vSui is above the truncation value, it assignd vSui = 0 and makes
                    # no prediction
                    vSui = vSeval
                
                #Updates temporary SV of player i
                SVpred[P[j]] = t/(t+1) * SVpred[P[j]] + (vSui-vSeval)/(t+1)
                
                vSeval = vSui
                
                j = j + 1   
            
            #debug
            #print("SVpred after "+str(t)+" executions = \n" + str(SVpred))
            
            i = i + 1
            
            
        # Returning information about the execution
        return dfPermutations, nExecs, SVpred
    
    """
    ---------  Truncated Random Structured Sampling SV approximation  ----------
    
        Calculates an approximation to Shapley Value for coalitions of players of a 
        set n based on a truncated random sampling algorithm described as:
    
            Initializes SV(i) = 0.0
            P = set of N*r random permutations of n=(p1, ..., pN)
            t = 0
            for each P
                t = t + 1
                vS = 0
                for each j in P
                    if vS <= Truncation value
                        vSuj = v(P[0:j])
                    else
                        vSuj = vS
                    SV(Pj) = t-1/t * SV(i) + 1/t * (vSuj - vS)
                    vS = vSuj
               
        We define the following set of parameters for the algorithm:
            * r: rounds of permutations to evaluate. r >= 1
            * TruncationValue: If this value is reached, it is supposed that the 
            remaining elements in the coalition have zero marginal value
        Apart from the MC algorithm parameters, the function requires the following 
        inputs:
        * Set of players n
        * Demand functions in the observation period for each player (yO)
        * Demand functions in the control period for each player (yC)
        * Temporary series of already calculated vS to speed up calculation 
            -> it will be returned back by the function
        * Temporary series of yPredicted functions 
            -> it will be returned back by the function
        
        The function assumes that v(S) is independent of the order of S
        
        The function returns the following data:
        * dfPermutations,
        * SVpred,
    """
    
    def getTRS_parallel(self, daskClient, iWorkerProcess_init, N, r, TruncationValue):
        # prints out some information about the execution
        print("RSS parallel calculation for "+str(len(N))+" companies: \n")
            
        dfPermutations = getRandomPermutations(N, r)
        
        return self.processPermutations_parallel(daskClient, iWorkerProcess_init, N, r, TruncationValue, dfPermutations)
    
       
    """
        Raw Shapley Value Calculation is done in two stages:
            1) Calculates v(S) for all possible combinations of elements in N
            This is done in parallel using dask.distributed
            2) Calculates SV for every element in N as per the SV formula
        
        The function requires the following inputs:
        * Set of players n
        * Demand functions in the observation period for each player (yO)
        * Demand functions in the control period for each player (yC)
        * Params for adjusting the model
        
        The function returns a pd.Series including all SV calculated
        
        The function assumes that v(S) is independent of the order of elements in S
    
    """
    
    def calculateSV_parallel(self, daskClient, iWorkerProcess_init, N):
        # prints out some information about the execution
        print("Raw parallel SV calculation for the following sources: \n")
        print(str(N))
        
        i = 1
        while i <= len(N):
            for coalition in getCoalitions(N, i):
                # Adds the work to the queue
                self.addWork(coalition)
            
            # End for
            i = i + 1
            
        # Executes all the works in the queue, which should be all the possible
        #coalitions of players. The function also uploads the results to y,
        #yhat and vS
        self.executeWorks(daskClient, iWorkerProcess_init, 8)
        
        """
        ------------------SHAPLEY VALUE CALCULATION----------------------
        """
        
        # Sets up SVPred vector
        SVpred = pd.Series([0.0]*len(N), index = N, name = "SV")
        
        #  We will check the number of permutations covered to make sure we are  
        # doing all right
        nPermutations = 0
        nCoalitions = 0
        
        # SV series - key = [1..N]
        # SV Pred compares the predicted curve with the real curve in the predicted 
        #period for S
        
        SVpred = pd.Series()
        
        # We calculate SV for all sources in N
        for source in N:
            print ("Calculating Shapley value for source %s" %(source))
            
            # Calculates subset S = n - {company}
            S = list(N).copy()
            S.remove(source)
            S = tuple(S)
            
            # Calculates the individual 
            print ("Calculating SV(S) for individual contributions")
            nPermutationsi = math.factorial(len(S))
            vi = self.getBufferValue((source,))
            SVpred.at[source] = vi * nPermutationsi
            
            #print ('Individual contribution to SVPred: %f = %i * %f' \
            #       %(vi * nPermutationsi, nPermutationsi, vi))
            nCoalitions += 1
            nPermutations += nPermutationsi
            
            
            i = 1
            while i <= len(S):
                print ("Calculating SV(S) for coalitions of level %i" %(i))
                nPermutationsi = math.factorial(i)*math.factorial(len(S)-i)
                for coalition in getCoalitions(S, i):
                    #print('Calculating SV for coalition %s - temporary SVpred: %f'\
                    #      %(str(coalition), SVpred[source]))
                    # Calculates marginal value and multiplies by the number of
                    #permutations of elements in S that deliver the same marginal 
                    #value
                    Sui = tuple(sorted(coalition + (source,)))
                    vSui = self.getBufferValue(Sui)
                    vCoalition = self.getBufferValue(coalition)
                    SVpred.at[source] = SVpred[source] + nPermutationsi * \
                        (vSui-vCoalition)
                    #print('Marginal contribution: %s - %f = %i * (%f - %f)' \
                    #      %( str(coalition),  nPermutationsi * \
                    #        (vSui-vCoalition), nPermutationsi, vSui, vCoalition))
                    #if (vSui-vCoalition) < 0:
                    #    print("WARNING: negative marginal contribution")
                    # Updates the number of permutations considered
                    nPermutations += nPermutationsi
                    nCoalitions += 1
                i += 1
                    
            SVpred[source] = SVpred[source] / math.factorial(len(N))
        
        
        return SVpred
    
    # Evaluates all combinations of set N and stores the results in vS
    def calculateAllVS(self, N):
        i = 1
        while i <= len(N):
            print ("Calculating v(S) for coalitions of size " + str(i))
            for coalition in getCoalitions(N, i):
                self.evaluateTuple(coalition)
                #print("v("+ str(coalition) + ") = " + str(v))
            # End for
            i = i + 1

    # Single thread version of the function
    def calculateSV(self, N):
        # prints out some information about the execution
        print("Raw SV calculation for the following sources: \n")
        print(str(N))
        
        self.calculateAllVS(N)
            
        
        """
        ------------------SHAPLEY VALUE CALCULATION----------------------
        """
        
        # Sets up SVPred vector
        SVpred = pd.Series([0.0]*len(N), index = list(range(len(N))), name = "SV")
        
        #  We will check the number of permutations covered to make sure we are  
        # doing all right
        nPermutations = 0
        nCoalitions = 0
        
        # SV series - key = [1..N]
        # SV Pred compares the predicted curve with the real curve in the predicted 
        #period for S
        
        SVpred = pd.Series()
        
        # We calculate SV for all sources in N
        for source in N:
            print ("Calculating Shapley value for source %s" %(source))
            
            # Calculates subset S = n - {company}
            S = list(N).copy()
            S.remove(source)
            S = tuple(S)
            
            # Calculates the individual 
            #print ("Calculating SV(S) for individual contributions")
            nPermutationsi = math.factorial(len(S))
            vi = self.getBufferValue((source,))
            SVpred.at[source] = vi * nPermutationsi
            
            #print ('Individual contribution to SVPred: %f = %i * %f' \
            #       %(vi * nPermutationsi, nPermutationsi, vi))
            nCoalitions += 1
            nPermutations += nPermutationsi
            
            
            i = 1
            while i <= len(S):
                print ("Calculating SV(S) for coalitions of level %i" %(i))
                nPermutationsi = math.factorial(i)*math.factorial(len(S)-i)
                for coalition in getCoalitions(S, i):
                    #print('Coalition %s - temporary SVpred: %f'\
                    #      %(str(coalition), SVpred[source]))
                    # Calculates marginal value and multiplies by the number of
                    #permutations of elements in S that deliver the same marginal 
                    #value
                    Sui = tuple(sorted(coalition + (source,)))
                    vSui = self.getBufferValue(Sui)
                    vCoalition = self.getBufferValue(coalition)
                    SVpred.at[source] = SVpred[source] + nPermutationsi * \
                        (vSui-vCoalition)
                    #print('Marginal contribution: %s - %f = %i * (%f - %f)' \
                    #      %( str(coalition),  nPermutationsi * \
                    #        (vSui-vCoalition), nPermutationsi, vSui, vCoalition))
                    #if (vSui-vCoalition) < 0:
                    #    print("WARNING: negative marginal contribution")
                    # Updates the number of permutations considered
                    nPermutations += nPermutationsi
                    nCoalitions += 1
                i += 1
                    
            SVpred[source] = SVpred[source] / math.factorial(len(N))
        
        
        return SVpred
    
    """
        Calculates the value of a set of sources S, by returning the maximum
        value of any combination of such sources.
    """
    def calculateMaxValue(self, N):
        # prints out some information about the execution
        print("Calculation of the maximum value for the following sources: \n")
        print(str(N))
        
        maxValue = self.ValueFunction.defaultValue
        
        i = 1
        while i <= len(N):
            print ("Calculating v(S) for coalitions of size " + str(i))
            for coalition in getCoalitions(N, i):
                value = self.evaluateTuple(coalition)
                if value > maxValue:
                    maxValue = value
            # End for
            i = i + 1
            
        return maxValue
    

    """
    --------------  Truncated Monte Carlo SV approximation  -----------------------

    Calculates an approximation to Shapley Value for coalitions of players of a 
    set n based on Truncated Monte Carlo algorithm described as:

    Initializes SV(i) = 0.0
        t = 0
        While not (changes in SV are over a threshold amd t>tmin) and t < maxT
            t = t + 1
            P = random permutation of n= (p1, ..., pN)
            v(Pi) = 0
            For all i in P
                Pi = {p1 ... pi}
                Calculate v(Pi) and update v(Pi) in v Series until 
                    v(Pi) > TruncationValue. After that assume that 
                    v(Pj) = 0 / j>i
                SV(i) = t-1/t * SV(i) + 1/t * (v(Pi)-v(Pi-1))
                Update convergence condition if (v(Pi)-v(Pi-1)) > threshold
            next i
    
    We will use as convergence conditions the following:
        * A minimum number of permutations evaluated: tmin
        * A maximum number of permutations evaluated: tmax
        * A maximum threshold variation in SVs: SV_var_threshold
        
    We define the following set of parameters to 
        * Absolute or relative threshold comparison (True/False)
        
    Apart from the MC algorithm parameters, the function requires the following
    inputs:
    * Set of players N for which to calculate the SV
        
    The function returns SVpred the approximation to SV calculated as a
    result of the TMC
    """
    def getSV_TMC(self, N, tmin, tmax, SV_var_threshold, 
                          relative_covergence_condition, TruncationValue):
        # prints out some information about the execution
        print("SV calculation using TMC for " + str(len(N)) + " companies")
        
        
        SVpred = pd.Series([0.0]*len(N), N, name = "SV Monte Carlo")
        t = 0
       
        
        # Dataframe that stores the information about executions, namely:
        #  1. dfPermutations : the permutations processed
        #  2. nExecs: number of executions until the threshold is reached
        #  3. dfvSVector: vPj
        #  4. dfSVectors: Temporary SV vector
        #  5. dfRideVectors: Temporary Ride Vectors for each permutation
        dfPermutations = pd.DataFrame()
        nExecs = 0
        
        
        # Variable to control if there is any SV variation above SV_var_threshold
        SV_variation = True
        
        # Truncated Monte Carlo loop
        while (t <= tmax) and not((t > tmin) and not(SV_variation)):
            # Reinits SV_variation
            SV_variation = False
            t = t + 1
            P = np.random.permutation(N).tolist()
                    
            print("Processing permutation "+str(t)+": " + str(P))
            # vP[i] Stores the value of the coalition Pi = {p1 ... pi} 
            #vP[0] = 0
            vP = [0]*(len(N)+1)
            j = 1
            vSeval = 0
            
            # WIll come True if there is a registered variation of SV_var_threshold
            # Variation be considered relative or absolute depending on the 
            #parameter relative_covergence_condition
            SV_variation = False
            while j <= len(P):
                           
                Pj = P[0:j]
                
                # Check if vS has reached the truncation value
                if vSeval <= TruncationValue:
                    # if not, it is calculated
                    if self.isCalculatedYet(Pj):
                        #Tries to get vSPred from memory
                        vP[j] = self.getBufferValue(Pj)
                    else:
                        vP[j] = self.evaluateTuple(Pj)
                        nExecs = nExecs+1
                    vSeval = vP[j]
                else:
                    # Above the truncation value it is assumed that the value
                    #of all users is 0
                    vP[j] = vP[j-1]
                        
                           
                # Updates SV of P[j-1]
                SV_old = SVpred[P[j-1]]
                SVpred[P[j-1]] = (t-1)/t * SVpred[P[j-1]] + (vP[j]-vP[j-1])/t
                
                # Convergence function might be absolute or relative depending on 
                #the paramater relative_covergence_condition
                # If any of the SV suffers a variation, then
                if relative_covergence_condition:
                    if (SV_old == 0) or (abs((SVpred[P[j-1]] - SV_old)/SV_old) \
                        > SV_var_threshold):
                        SV_variation = True
                else:
                    if (SV_old == 0) or (abs((SVpred[P[j-1]] - SV_old)) \
                        > SV_var_threshold):
                        SV_variation = True
                
                #DEBUG
                #print("SV of %s updated from %f to %f. Var: %f. Cond: %s" \
                #      %(str(P[j-1]), SV_old, SVpred[P[j-1]],  \
                #      abs((SVpred[P[j-1]] - SV_old)/SV_old), str(SV_variation))) 
                
                # Updates j
                j += 1
            
         
            # Stores information about the execution
            dfPermutations[t] = P
        
        #Printing some debug information
        print("SV threshold achieved after processing " + str(t) + " permutations")
        print(str(nExecs) + " new model trainning and executions were needed")
        print(str(dfPermutations))
        
        # Returning information about the execution
        return SVpred

class NAException(Exception):
    pass

class vbdeFrameworkSubaditive (vbdeFramework):
    """ Constructor
    Calls the constructor of vbdeFramework without buffering
    """
    def __init__(self, iCombiner, iModel, iValueFunction):
        super().__init__(iCombiner, iModel, iValueFunction, i_vSbuffer=False)
    
    """
    getValue(s)
    The function returns v(S) if S was already evaluated an N/A in case not
    s is assumed to be a tuple of sources whose value will be retrieved
    from the buffer.
    If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferValue(self, s):
        raise NAException("getBufferValue not available for Subaditive frameworks")
        
    
    """
        getBufferYhat(s)
        The function returns yhat(S) if S was already evaluated an N/A in case not
        s is assumed to be a tuple of sources whose value will be retrieved
        from the buffer.
        If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferYhat(self, s):
        raise NAException("getBufferYhat not available for Subaditive frameworks")
    
    
    """
        getBufferY(s)
        The function returns v(S) if S was already evaluated an N/A in case not
        s is assumed to be a tuple of sources whose value will be retrieved
        from the buffer.
        If s was not processed yet, it returns an exception (index out of bouds)
    """
    def getBufferY(self, s):
        raise NAException("getBufferY not available for Subaditive frameworks")
    
    """
        Clear buffers of the framework.
    """
    def clearBuffer(self):
        raise NAException("getBufferY not available for Subaditive frameworks")
    
    
    """
        The function executes the model and evaluates the output.
        It stores the results in the buffers of the framework for future uses:
            y, yhat and vS
        In case of any error, it returns the defaultValue provided by the 
        metric.
    """
    def evaluateTuple(self, K):  

        try:
            yInput = self.Combiner.getCombinedInput(K)
            #self.y[str(K)] = yInput
            yOutput = self.Model.getModelOutput(yInput)
            #self.yhat[str(K)] = yOutput
            v = self.ValueFunction.evaluateOutput(yOutput)
            if math.isnan(v):
                return self.ValueFunction.defaultValue
            else:
                return v
        except:
            return self.ValueFunction.defaultValue
        
        
    """
        The following function returns a boolean telling whether there is a
        pending work to process the coalition s. s is
        assumed to be a tuple of sources whose value will be calculated.
    """
    def isCalculatedYet(self, s):
        return False
    
    """
        addWork
         This procedure allows to include a work to the queue of works to be
        distributed
    """
    def addWork(self, s):
        raise NAException("addWork not available for Subaditive frameworks")
                
    """
        executeWorks
         The following procedure schedules the existing works among nWorkers
        in config.daskClient. It executes the works to calculate the v(S) for all
        S in queue S. It distributes the work among nWorkers. It saves the schedule 
        to the worker schedule and starts the workers. Finally it waits for them 
        to finish the works and consolidate their results.
         The implementation is simple, it just distributes the workload among
        a number of nWorkers which is an input from the callee.
    """
    def executeWorks(self, daskClient, iWorkerProcess_init, nWorkers):
        raise NAException("executeWorks not available for Subaditive frameworks")
        

    """
        getIndividualValues (vi)
        
        Given a set of sources and a set of training examples and information,
        it trains the model individually with the data of every source i and 
        obtains the results and its valuation.    
    """
    def getIndividualValues(self, N):
        
        # Result is a pd.Series object including all the valuations
        result = pd.Series()
        
        i = 0
        for source in N:
            result.at[i] = self.evaluateTuple([source])
            i += 1
            
        # returns the result, which after exitting the loops should contain all
        # the individual values.
        return result


    """
        getIndividualValuesParallel
        
        Given a set of sources and a set of training examples and information,
        it trains the model individually with the data of every source i and 
        obtains the results and its valuation. Implementation is done using
        parallel workers to calculate all the individual values.
    """
    
    def getIndividualValuesParallel(self, daskClient, iWorkerProcess_init, N):   
        return self.getIndividualValues(N)
    
    
    """
        getLOOValues
        
        Given a set of sources and a set of training examples and information,
        it obtains the LOOi = v(N) - v(N- {i}). The implementation uses a
        single thread. In subaditive frameworks LOOi = vi
        
        The function assumes that v(S) is independent of the order of S
    """
    def getLOOValues(self, N):  
        return self.getIndividualValues(N)


    """
        getLOOValues
        
        Given a set of sources and a set of training examples and information,
        it obtains the LOOi = v(N) - v(N- {i}). The implementation distributes 
        the workload among a set of workers and retrieves the results from
        them at the end of the executions.
        
        The function assumes that v(S) is independent of the order of S
    """
    def getLOOValuesParallel(self, daskClient, iWorkerProcess_init, N):
        return self.getIndividualValues(N)

    """
    ___________________________________________________________________________
    IMPORT / EXPORT functions allow for storing and retrieving from disk the
    main buffers used in the Framework model:
        vS
        y
        yhat
        
     They are not allowed for this type of vbdeFrameworks
        
    
    importVSFile()
    The following function imports an existing vS file 
    """
    def importVSFile(self, name):
        raise NAException("importVSFile not available for Subaditive frameworks")
    
    
    
    """
    exportVSFile()
    The following function exports a vS file to the 
    """
    def exportVSFile(self, name):
        raise NAException("exportVSFile not available for Subaditive frameworks")
    
    """
    importYFile()
    The following function imports an existing vS file 
    """
    def importYFile(self, name):        
        raise NAException("importYFile not available for Subaditive frameworks")
    
    
    
    """
    exportYFile()
    The following function exports a vS file to the 
    """
    def exportYFile(self, name):
        raise NAException("exportYFile not available for Subaditive frameworks")
    
    
    """
    importYhatFile()
    The following function imports an existing vS file 
    """
    def importYhatFile(self, name):
        raise NAException("importYhatFile not available for Subaditive frameworks")
    
    """
    exportYhatFile()
    The following function exports a vS file to the 
    """
    def exportYhatFile(self, name):
        raise NAException("exportYhatFile not available for Subaditive frameworks")
    
    
    
    """
    consolidateWorkersResults()
    Retrieves the results from workers output files and uploads it to the
    existing framework buffers: vS, y and yhat.
    """
    def consolidateWorkerFiles(self, nWorkers):
        raise NAException("consolidateWorkerFiles not available for Subaditive frameworks")
    
    
    """
    ------ Truncated  Structured Sampling SV approximation  -------
    
        
    """
    # Single thearded version of the function
    # The function assumes that v(S) is independent of the order of S
    # Takes as inputs
    def getSV_TSS(self, N, r, TruncationValue):
            
        # Returning information about the execution
        return N, len(N), self.getIndividualValues(N)
    
    # Parallel execution of getTSS. It works in two steps:
    #  1: Calculation of the necessary v(S). It takes into consideration the
    # truncation value that is passed as an input to the function. 
    #  2: Approximation to SV - which is a simple calculation once all the v(S) are
    # in the buffer.
    # The function assumes that v(S) is independent of the order of S
    # You have to pass as an input the district c for which this is computed
    def getTSS_parallel(self, daskClient, iWorkerProcess_init, N, r, TruncationValue):
            
        # Returning information about the execution
        return N, len(N), self.getIndividualValues(N)
    
    #  Parallel processing of a set of permutations for sampling algorithms. It
    # can work both with TSS and TRS algorithms.
    #  It works in two steps:
    #  1: Calculation of the necessary v(S). It takes into consideration the
    # truncation value that is passed as an input to the function. It works by
    # processing coalitions up to a certain position i, and calculates v(S) if 
    # and only if v(Si-1) < Truncation_value.
    #  2: Approximation to SV - which is a simple calculation once all the v(S)
    # are in the buffer.
    # The function assumes that v(S) is independent of the order of S
    def processPermutations_parallel(self, daskClient, iWorkerProcess_init, N,\
                                     r, TruncationValue, dfPermutations):
        raise NAException("processPermutations_parallel not available for Subaditive frameworks")
        
    
    """
    ------ Truncated Random Sampling SV approximation  -------
    
        Returns individual values in subaditive frameworks
    """
    # Single thearded version of the function
    # The function assumes that v(S) is independent of the order of S
    # Takes as inputs
    def getSV_TRS(self, N, r, TruncationValue):
        
        # Returning information about the execution
        return N, len(N), self.getIndividualValues(N)
    
    """
    ---------  Truncated Random Structured Sampling SV approximation  ----------
    
        Returns individual values in subaditive frameworks
    """
    
    def getTRS_parallel(self, daskClient, iWorkerProcess_init, N, r, TruncationValue):
        # Returning information about the execution
        return N, len(N), self.getIndividualValues(N)
    
       
    """
        Raw Shapley Value Calculation is done in two stages:
            1) Calculates v(S) for all possible combinations of elements in N
            This is done in parallel using dask.distributed
            2) Calculates SV for every element in N as per the SV formula
        
        The function requires the following inputs:
        * Set of players n
        * Demand functions in the observation period for each player (yO)
        * Demand functions in the control period for each player (yC)
        * Params for adjusting the model
        
        The function returns a pd.Series including all SV calculated
        
        The function assumes that v(S) is independent of the order of elements in S
    
    """
    
    def calculateSV_parallel(self, daskClient, iWorkerProcess_init, N):
        # Returning information about the execution
        return self.getIndividualValues(N)
    
    # Evaluates all combinations of set N and stores the results in vS
    def calculateAllVS(self, N):
        raise NAException("calculateAllVS not available for Subaditive frameworks")


    # Single thread version of the function
    def calculateSV(self, N):
        return self.getIndividualValues(N)
    
    """
        Calculates the value of a set of sources S, by returning the maximum
        value of any combination of such sources.
    """
    def calculateMaxValue(self, N):
        raise NAException("calculateMaxValue not available for Subaditive frameworks")
    

    """
    --------------  Truncated Monte Carlo SV approximation  -----------------------

    Calculates an approximation to Shapley Value for coalitions of players of a 
    set n based on Truncated Monte Carlo algorithm described as:

    returns the individual values in subaditive vbdeFrameworks
    """
    def getSV_TMC(self, N, tmin, tmax, SV_var_threshold, 
                          relative_covergence_condition, TruncationValue):
        return self.getIndividualValues(N)
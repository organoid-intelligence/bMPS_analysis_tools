import os
import time
import numpy as np
import matplotlib.pyplot as plt
import math
import torch
from joblib import Parallel, delayed
from sklearn.manifold import Isomap # for Isomap dimensionality reduction
from scipy.stats import entropy
from sklearn.decomposition import PCA
import math
from sklearn.cluster import DBSCAN
import networkx as nx
from pyinform.transferentropy import *
from sklearn.decomposition import FastICA
import cv2 #only needed for videos
import json
import bct
from scipy.spatial import distance_matrix
from scipy.spatial.kdtree import distance_matrix
from nilearn.connectome import ConnectivityMeasure

"""
Spike Set Module:

This module defines the main SpikeSet class which incorporates the Functional Connectivity Metrics

Authors: Jack and Alon 
"""
def CountCalls(func):
  """This function is used to count how many times a method is called in the SpikeSet class so that multiple figures produced by the same method can
  be assigned to distinct folders. It may skip or not reset numbers which is a bug that I am trying to fix, but it should not overwrite figures."""
  def wrapper(*args, **kwargs):
      wrapper.num_calls += 1
      return func(*args, **kwargs)
  wrapper.num_calls = 0
  return wrapper


#@title SpikeSet
@CountCalls
class SpikeSet:
  def __init__(self, data, params):
    """ Initializes data structure.
    Arguments:
      data: pandas dataframe of electrophysology recording 
      params: dictionary specify arguments to the SpikeSet class
        convolution_size: size of convolution filter, when this equals convolution_step it is binning
        convolution_step: how many time steps to move convolution filter as it slides
        min_electrode_frequency: the minimum frequency in Hertz for an electrode to be analyzed, if None all electrodes in data are analyzed
        sampling_freq: the sampling frequency of electrophysiology setup in Hertz
        only_binary: convert binning values to binary values, True is the default
        permute_columns: randomly permute columns of each row, default is False
    Returns:
      None"""
    start                        = time.time() # for debugging, to identify portions of the code that take extra long
    self.save_request_number     = 0 # for indexing saved files
    self.params                  = params.copy()
    torch.manual_seed(self.params["seed"])
    self.data                    = data.copy()
    self.convolution_size        = self.params["convolution_size"]
    self.convolution_step        = self.params["convolution_step"]
    self.min_electrode_frequency = self.params["min_electrode_frequency"]
    self.sampling_freq           = self.params["sampling_freq"]
    self.only_binary             = self.params["only_binary"]
    self.permute_columns         = self.params["permute_columns"]
    # removing electrodes suspected to be noise or weird
    if self.min_electrode_frequency is not None:
      self.time_steps_total = self.data["frame"].max()
      freq                  = self.sampling_freq * self.data["electrode"].value_counts() / self.time_steps_total
      electrodes_delete     = freq[freq < self.min_electrode_frequency].keys()
      self.data             = self.data[~self.data["electrode"].isin(electrodes_delete)]
      stop                  = time.time()
      self.params["Removing Low Frequency Electrodes Time"] = stop - start

    start      = time.time()
    self.x_min = min(self.data["x"])
    self.x_max = max(self.data["x"])
    self.y_min = min(self.data["y"])
    self.y_max = max(self.data["y"])

    """put data into raster format in which each row represents the patterns of
       spikes at a given time and each column represents a different electrode"""
    
    self.electrode_count  = self.data["electrode"].nunique() # get number of electrodes
    self.channel_count    = self.data["channel"].nunique() # get number of electrodes
    self.time_steps_total = self.data["frame"].max() # total time steps of experiment, roughly (since might be no electrodes active later)
    self.time_steps_count = self.data["frame"].nunique()
    self.spike_format     = np.zeros((self.time_steps_count, self.electrode_count))
    # generate vector for all unique times
    self.time_vector      = np.sort(self.data.copy().drop_duplicates(subset=['frame'])["frame"].to_numpy())
    # generate vector for all unique electrodes
    self.data             = self.data.dropna(subset=["electrode"])
    self.electrode_vector = np.sort(self.data.copy().drop_duplicates(subset=['electrode'])["electrode"].to_numpy())
    self.channel_vector   = np.sort(self.data.copy().drop_duplicates(subset=['channel'])["channel"].to_numpy())
    if math.isnan(self.electrode_vector[-1]):
      self.electrode_vector = self.electrode_vector[:-1]
    self.electrode_dict   = dict(zip(self.electrode_vector, np.arange(self.electrode_count))) 
    self.time_dict        =  dict(zip(self.time_vector, np.arange(self.time_steps_count)))
    stop  = time.time()
    # iterate through time_vector and then populate spike format
    start = time.time()
    for index, row in self.data.iterrows():
      electrode_active = row["electrode"]
      electrode_index  = self.electrode_dict[electrode_active] # dict
      frame            = row["frame"]
      time_index       = self.time_dict[frame]
      self.spike_format[time_index][electrode_index] = 1

    self.spike_prob_vector = np.sum(self.spike_format, axis = 0) / self.time_steps_total
    stop = time.time()
    self.params["Population Spike Format Time"] = stop - start
    
    # permute columns if self.permute_columns is True
    if self.permute_columns:
        for i in range(len(self.spike_format)):
            self.spike_format[i] = np.random.permutation(self.spike_format[i])
    # generate convolution
    start = time.time()
    self.spike_format_convolved = self.SpikeSet_Convolve(self.convolution_size, self.convolution_step)
    if self.only_binary:
      self.spike_format_convolved[self.spike_format_convolved > 0] = 1
    stop = time.time()
    self.params["Bin Time"] = stop - start
    # print("BIN:", stop - start)

    # get position of each channel
    channel_positions = []
    for channel, index in zip(self.channel_vector, range(self.channel_count)):
      row = self.data.loc[self.data['channel'] == channel].iloc[0]
      x   = row["x"]
      y   = self.y_max - row["y"]
      channel_positions.append([x, y])
    self.channel_positions = np.array(channel_positions)
    
    # get position of each electrode
    electrode_positions = []
    for electrode, index in zip(self.electrode_vector, range(self.electrode_count)):
      row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
      x   = row["x"]
      y   = row["y"]
      electrode_positions.append([x, y])
    self.electrode_positions = np.array(electrode_positions)   
    
    # write to log file
    if self.params["organoid_name"] is not None:
      self.params["results_path"] = self.params["results_path"] + self.params["organoid_name"] + "/"# + date_time + '/'
    else:
      self.params["results_path"] = self.params["results_path"] + "/"# + date_time + '/'
    isExist = os.path.exists(self.params["results_path"])
    if not isExist:
        os.makedirs(self.params["results_path"])
    log_file = self.params["results_path"] + "log.txt"
    json.dump(self.params, open(log_file,'w'))
    

  
  def SpikeSet_GetParams(self):
    """Returns the main parameters of the SpikeSet object.
    Arguments:
        None
    Returns: 
        params: main parameters of SpikeSet object"""

    return self.params

  def SpikeSet_Convolve(self, convolution_size, convolution_step):
    """Convolve the spike counts over time. This is tricky since consecutive rows of self.spike_format do not
    necessarily represent consecutive frames. If self.spike_format[i] refers to frame t, and no electrode fired
    in frame t + 1, then self.spike_format[i+1] will refer to some frame t' such that t' > t + 1. The frame t' 
    is the first frame in which an electrode fired after frame t. IMPORTANT: This is only intended for binning, 
    meaning convolution_size == convolution_step. The code could be changed to generalize to more convolutions.
    
    Arguments:
      convolution_size: the size of each bin
      convolution_step: the stride the convolution kernel uses
    Returns:
      spike_format_binned: the result of the convolution"""
    # filter = np.ones((convolution_size, 1)) / convolution_size # to perform convolution
    # time_vector_upper = np.zeros_like(self.time_vector)
    modified_size = convolution_step * int((self.time_steps_count - convolution_size) / convolution_step) + convolution_size + 1
    # time_vector_adjusted = self.time_vector[:modified_size]
    # spike_format_convolved = np.zeros((self.electrode_count, int(((self.time_steps_count - convolution_size) / convolution_step) + 1)))
    # spike_format_transposed = np.transpose(self.spike_format[:modified_size])
    self.convolution_size = convolution_size
    self.convolution_step = convolution_step
#     print("Number of steps:", len(time_vector_adjusted) - convolution_size)
#     print("Spike Format Convolved:", np.shape(spike_format_convolved))

    binned_steps = int(((self.time_steps_total) / convolution_step) + 1) # number of time steps after binning
#     print("binned steps:", binned_steps)
    spike_format_binned = np.zeros((binned_steps, self.electrode_count))
    t = 0
    for index in range(binned_steps):
      index_time_lower = np.where(self.time_vector >= t)
      index_time_upper = np.where(self.time_vector >= t + convolution_size)
      if len(index_time_upper[0]) == 0:
        index_time_upper = [[self.time_steps_count]]
      index_time_lower = index_time_lower[0][0]
      index_time_upper = index_time_upper[0][0]
      if index_time_lower == index_time_upper: # no spikes during interval
        t+=convolution_step
        #print("Same")
        #print("index time lower:", index_time_lower, "index time upper:", index_time_upper)
        continue
      #print("index time lower:", index_time_lower, "index time upper:", index_time_upper)
      #spike_format_binned[int(t /convolution_size)] = np.mean(self.spike_format[index_time_lower:index_time_upper], axis = 0)
      spike_format_binned[index] = np.mean(self.spike_format[index_time_lower:index_time_upper], axis = 0)
      t+=convolution_step
      #print("T:", t)

    # print("binning done")
    return spike_format_binned
                          

  def SpikeSet_Generate_PCA_ICA(self, num_components = 3, use_binned = True, data_type = "electrodes", reduction_type="PCA", use_dist_matrix=False):
    """Perform PCA using time steps as samples and electrodes as features or 
    electrodes as samples and time steps as features. Creates instance variable 
    containing the result of the PCA transform.
    
    Arguments:
      num_components: number of PCA components to keep
      use_binned: boolean indicating whether to use the original or binned spike data
      data_type: "electrodes" or "global activity" indicating which is the sample
      use_dist_matrix: boolean indicating whether PCA should be performed on spike data or distance matrix
    Returns:
      None """

    spike_format = self.spike_format
    if use_binned:
      spike_format = self.spike_format_convolved
    if data_type == "electrodes":
      spike_format = np.transpose(spike_format)
    if use_dist_matrix:
      spike_format = self.dist_matrix
    if reduction_type == "PCA":
      pca = PCA(n_components=num_components, svd_solver='full')
      self.spike_format_pca = pca.fit_transform(spike_format)
    if reduction_type == "ICA":
      ica = FastICA(n_components=num_components, random_state=0, whiten='unit-variance')
      self.spike_format_ica = ica.fit_transform(spike_format)

  def SpikeSet_Plot_PCA_ICA(self, reduction_type="PCA", superimpose = False, slicewise=False):
    """Plot the results of the PCA as a 2D or 3D plot in a manner consistent with
    the spatial locations of the electrodes.

    Arguments:
      superimpose: boolean indicating whether to make a 3D plot in which Z axis is a different component
      slicewise: boolean indicating whether to create 2D plots in which X,Y axises are electrode spatial positions
      and color is the intensity of specific component
    Returns:
      None"""
    if reduction_type == "PCA":
      reduced_dim = self.spike_format_pca
    if reduction_type == "ICA":
      reduced_dim = self.spike_
    num_components = np.shape(self.spike_format_pca)[1]
    if superimpose == False:
      plt.scatter(reduced_dim[:,0], reduced_dim[:,1])
      plt.xlabel("Component 1")
      plt.ylabel("Component 2")
      plt.show()
    if superimpose == True: # 3D plots
      x = []
      y = []
      for i in range(self.electrode_count):
        row = self.data.loc[self.data['electrode'] == self.electrode_vector[i]].iloc[0]
        x.append(row["x"])
        y.append(self.y_max - row["y"])

      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      for i in range(num_components):
        ax.scatter(x, y,i * np.ones(len(x)), c = reduced_dim[:,i], cmap='viridis')
        # plt.colorbar()
      ax.set_xlabel("Spatial X Dimension")
      ax.set_ylabel("Spatial Y Dimenson")
      ax.set_zlabel("Component")
      ax.set_title("PCA of Electrode Activity")
      plt.show()
      if slicewise: # 2D plots
        for i in range(num_components):
          plt.scatter(x, y, c = reduced_dim[:,i], cmap='viridis')
          plt.xlabel("Spatial X Dimension")
          plt.ylabel("Spatial Y Dimension")
          plt.title("PCA of Electrode Activity For Component " + str(i))
          plt.colorbar()
          plt.show()

  def SpikeSet_GenerateDistanceMatrix(self, use_binned=True, data_type="electrodes", distance_metric = "cosine", remove_outliers=True, outlier_percentile=95, timesteps = None):
    """Generate a matrix in which each entry is the distance/strength between two samples. 
    A sample can either be the activity of each electrode over time or the activity 
    of a population of electrodes at different times. Set instance variable to hold distance matrix.
    
    Arguments:
      use_binned: boolean specifying whether to use binned data
      data_type: "electrodes" or "global activity" indicating which is the sample
      distance_metric: ["cosine", "mutual_information", "transfer_entropy"] indicating how to measure distance/weight
      remove_outliers: boolean indicating whether to remove entries of the matrix above a specific percentile
      outlier percentile: percentile above which entries are considered outliers
      timesteps: specific timesteps on which the distance matrix should be computed (used for making connectivity movie)
    Returns:
      None
    """
    
    self.distance_metric = distance_metric
    spike_format = self.spike_format_convolved
    if use_binned == False:
      spike_format = self.spike_format
    if data_type == "electrodes":
      spike_format = np.transpose(spike_format)
    if timesteps is not None: # specific slice to use
      spike_format = spike_format[:,timesteps[0]:timesteps[1]]
#     print("Spike Format:", np.shape(spike_format))
    spike_format[spike_format > 0] = 1
    dims = np.shape(spike_format)
    dist_matrix = np.zeros((dims[0], dims[0]))
    # different distance metrics
    if self.distance_metric == "cosine":
      dist_matrix = np.matmul(spike_format, np.transpose(spike_format))
      norm = np.reshape(np.linalg.norm(spike_format, axis = 1), (-1,1))
#       print(np.shape(norm))
      norm = np.matmul(norm, np.transpose(norm))
#       print(np.shape(norm))
      dist_matrix = 1 - dist_matrix / norm
      dist_matrix[dist_matrix < 0] = 0
      dist_matrix = np.nan_to_num(dist_matrix, nan=1) # to get rid of NAN
      
      magnitude_vector = np.sum(dist_matrix, axis=1)
#       print("Distance_Matrix:", np.shape(dist_matrix))
      non_outlier_indexes = magnitude_vector < np.percentile(magnitude_vector, outlier_percentile)
      
    if self.distance_metric == "mutual_information":
      # I(X,Y) = H(X) + H(Y) - H(X, Y)
      binary_weights = np.reshape(np.power(2, np.arange(2)),(-1,1))
      probs = np.sum(spike_format, axis = 1) / dims[1] 
      complement_probs = 1 - probs
      entropy_vector_marginal = np.reshape(entropy(np.stack((probs, complement_probs)), base=2), (-1,1))
      print("entropy vector marginal:", np.shape(entropy_vector_marginal))
      print("Dims:", dims)
      entropy_matrix_marginal = np.matmul(entropy_vector_marginal, np.ones((1, dims[0])))
      frequency_matrix = np.zeros((dims[0], dims[0], 4))
      frequency_matrix[:,:,0] = np.matmul(spike_format, np.transpose(spike_format))
      frequency_matrix[:,:,1] = np.matmul(spike_format, np.transpose(1 -  spike_format))
      frequency_matrix[:,:,2] = np.matmul(1 - spike_format, np.transpose(spike_format))
      frequency_matrix[:,:,3] = np.matmul(1 - spike_format, np.transpose(1 - spike_format))
      frequency_matrix = frequency_matrix / dims[1] # to make it a probability it should be 1, had been 0
      entropy_matrix_joint = entropy(frequency_matrix, base=2, axis = 2)
      print("entropy joint:", np.shape(entropy_matrix_joint))
      dist_matrix = entropy_matrix_marginal + np.transpose(entropy_matrix_marginal) - entropy_matrix_joint
      dist_matrix[dist_matrix < 0] = 0
      magnitude_vector = np.sum(dist_matrix, axis=1)
      print("Distance_Matrix:", np.shape(dist_matrix))
      non_outlier_indexes = magnitude_vector > np.percentile(magnitude_vector, 100 - outlier_percentile)
    if distance_metric == "transfer_entropy":
      dist_matrix = np.zeros((dims[0], dims[0]))
      for i in range(dims[0]):
        if i % 50 == 0:
          print(i)
        for j in range(dims[0]):
          dist_matrix[i][j] = transfer_entropy(spike_format[i], spike_format[j], k = 2)
      magnitude_vector = np.sum(dist_matrix, axis=1)
      print("Distance_Matrix:", np.shape(dist_matrix))
      non_outlier_indexes = magnitude_vector > np.percentile(magnitude_vector, 100 - outlier_percentile)
    
    #Alternate Measures - Alon 08/06/23
    if distance_metric == 'partial_correlation':
      dist_matrix = np.zeros((dims[0], dims[0]))
      correlation_measure = ConnectivityMeasure(kind='partial correlation')
      dist_matrix = correlation_measure.fit_transform([spike_format.T])[0]
      magnitude_vector = np.sum(dist_matrix, axis=1)
#       print(dist_matrix.shape)
    
    if distance_metric == 'correlation':
      print('Starting Correlation')
      #Try Pinguin alternate:
      X=spike_format.T
      n_vars = spike_format.T.shape[1]

      def parallel_corrcoef(X, i, j):
          return np.corrcoef(X[:, i], X[:, j])[0, 1]

      edges = np.zeros((n_vars, n_vars))
      # edges = np.corrcoef(X,X)
      #   # Parallel computation
      edges = Parallel(n_jobs=-1)(delayed(parallel_corrcoef)(X, i, j) for i in range(n_vars) for j in range(n_vars))
      edges = np.array(edges).reshape((n_vars, n_vars))
      dist_matrix = np.zeros((dims[0], dims[0]))
      # edges = np.nan_to_num(np.corrcoef(spike_format.T))
      print('Corr Done')
      # print(edges.shape)

      # np.fill_diagonal(edges, 0)
      graph = nx.Graph(edges)
      # print('Removing Edges')
      graph.remove_edges_from([(n1, n2) for n1, n2, w in graph.edges(data="weight") if w < 0])
      dist_matrix=nx.adjacency_matrix(graph)
      # print('Finished Correlation Matrix')
      magnitude_vector = np.sum(dist_matrix, axis=1)
#       print(dist_matrix.shape)
    
    if distance_metric == 'precision':
      dist_matrix = np.zeros((dims[0], dims[0]))
      correlation_measure = ConnectivityMeasure(kind='precision')
      dist_matrix = correlation_measure.fit_transform([spike_format.T])[0]
      magnitude_vector = np.sum(dist_matrix, axis=1)
#       print(dist_matrix.shape)
    
    self.non_outlier_indexes = magnitude_vector >= np.percentile(magnitude_vector, 0)
    if remove_outliers: # removing noisy entries
      self.non_outlier_indexes = non_outlier_indexes
      dist_matrix = dist_matrix[self.non_outlier_indexes]
      dist_matrix = dist_matrix[:, self.non_outlier_indexes]
    self.dist_matrix = dist_matrix
    self.distance_metric = distance_metric
  

  def SpikeSet_GenerateIsomap(self, neighborhood_size = 5, dim_num = 3, use_binned = True, metric = "precomputed", data_type = "global_activity"):
    """Reduce the dimensionality of the data using isomaps. Creates instance variable of 
    isomap object.
    
    Arguments:
      neighborhood_size: used by isomap to determine geodisc distances
      dim_num: target dimensionality after isomap transform
      use_binned: boolean indicating whether to use binned data
      metric: how to compute distances, "precomputed" means that the precomputed distance matrix is used
      data_type: "electrodes" or "global activity" indicating which is the sample
    Returns:
      spike_format_isomap: data dimensionally reduced using isomap
      self.iso: isomap object
    """
    self.isomap_dict = {}
    self.isomap_dict["neighborhood_size"] = 5
    self.isomap_dict["dim_num"] = 3
    self.isomap_dict["use_binned"] = True
    self.isomap_dict["metric"] = metric
    self.isomap_dict["data_type"] = data_type
    
    spike_format = self.spike_format_convolved
    if use_binned == False:
      spike_format = self.spike_format
    if data_type == "electrodes":
      spike_format = np.transpose(spike_format)
        
    dims = np.shape(spike_format)
    if metric == "precomputed":
      spike_format = self.dist_matrix
    elif dims[0] > dims[1]: #need to precompute distances
      dist_matrix = np.zeros((dims[0], dims[0]))
      metric = "precomputed"
      for i in range(dims[0]):
        for j in range(i, dims[0]):
          dist_matrix[i][j] = 1 - np.dot(spike_format[i], spike_format[j])/max(np.linalg.norm(spike_format[i]) * np.linalg.norm(spike_format[j]),1)
          dist_matrix[j][i] = dist_matrix[i][j]
      spike_format = dist_matrix
      spike_format[spike_format < 0] = 0
      # print(np.shape(spike_format))
#       print(np.min(spike_format), np.max(spike_format))
    self.iso = Isomap(
    n_neighbors=neighborhood_size, # default=5, algorithm finds local structures based on the nearest neighbors
    n_components=dim_num, # number of dimensions
    eigen_solver='auto', # {‘auto’, ‘arpack’, ‘dense’}, default=’auto’
    tol=0, # default=0, Convergence tolerance passed to arpack or lobpcg. not used if eigen_solver == ‘dense’.
    max_iter=None, # default=None, Maximum number of iterations for the arpack solver. not used if eigen_solver == ‘dense’.
    path_method='auto', # {‘auto’, ‘FW’, ‘D’}, default=’auto’, Method to use in finding shortest path.
    neighbors_algorithm='auto', # neighbors_algorithm{‘auto’, ‘brute’, ‘kd_tree’, ‘ball_tree’}, default=’auto’
    n_jobs=-1, # n_jobsint or None, default=None, The number of parallel jobs to run. -1 means using all processors
    metric=metric, # string, or callable, default=”minkowski”
    p=2, # default=2, Parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2
    metric_params=None # default=None, Additional keyword arguments for the metric function.
    )
    ### Fit the data and transform it
    spike_format_isomap = self.iso.fit_transform(spike_format)
    return (spike_format_isomap, self.isomap_dict, self.iso)

  def SpikeSet_PlotIsomap(self, isomap, all_slices = True, slicewise = False):
    """Plot the results of the isomap as a 2D or 3D plot in a manner consistent with
    the spatial locations of the electrodes. 

    Arguments:
      isomap: data transformed using isomap, matrix
      all_slices: boolean indicating whether to make 3D plot of isomap components 
      slicewise: boolean indicating whether to create 2D plots in which X,Y axises are electrode spatial positions
      and color is the intensity of specific component
    Returns: 
      None
    """
    dim_num = np.shape(isomap)[1]
    if all_slices:
      x = []
      y = []
      for i in range(self.electrode_count):
        row = self.data.loc[self.data['electrode'] == self.electrode_vector[i]].iloc[0]
        x.append(row["x"])
        y.append(self.y_max - row["y"])
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      # Generate the values

      for i in range(dim_num):
        ax.scatter(x, y,i * np.ones(len(x)), c = isomap[:,i], cmap='viridis')
        # plt.colorbar()
      ax.set_xlabel("Spatial X Dimension")
      ax.set_ylabel("Spatial Y Dimenson")
      ax.set_zlabel("Component")
      ax.set_title("Isomap of Electrode Activity")
      plt.show()

      if slicewise:
        for i in range(dim_num):
          plt.scatter(x, y, c = isomap[:,i], cmap='viridis')
          plt.colorbar()
          plt.xlabel("Spatial X Dimension")
          plt.ylabel("Spatial Y Dimension")
          plt.title("Measure of Electrode Activity For Component " + str(i))
          plt.show()
    
    if all_slices == False and dim_num == 3: #3D plot
      # Create a 3D scatter plot
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      # Generate the values
      x_vals = isomap[:, 0:1]
      y_vals = isomap[:, 1:2]
      z_vals = isomap[:, 2:3]
      # Plot the values
      ax.scatter(x_vals, y_vals, z_vals, c = 'b', marker='o')
      ax.set_xlabel('X-axis')
      ax.set_ylabel('Y-axis')
      ax.set_zlabel('Z-axis')
      ax.set_title("Isomap")
      plt.show()
    if all_slices == False and dim_num==2: #2D plot
      plt.scatter(isomap[:,0:1], isomap[:,1:2])
      plt.xlabel("Component 1")
      plt.ylabel("Component 2")
      plt.show()
      plt.close('all')

    
  def SpikeSet_ClusterConnections(self, max_num_clusters, explained_entropy_frac):
    """This function puts each electrode into clusters using the DBSCAN algorithm
    applied to the distance matrix. The most recent call of GenerateDistanceMatrix
    will generate the distance matrix that is used in these calculations. For each
    cluster, the program tries to identify which of the other clusters have a strong 
    causal relationship with the specific cluster. This is determined by computing
    the sum of the pairwise transfer entropy between two clusters A and B. Note
    that the sum for A -> B is not necessarily the same as the sum for B-> A. The 
    transfer entropy from each cluster is then summed together and the minimum number of
    clusters needed to explain some desired fraction of this total transfer entropy 
    determines which clusters are deemed to cause another. For example imagine there
    are 4 clusters: A, B, C, D. Imagine we are interested in seeing which clusters 
    cause A. We compute the transfer entropy x -> y for each x in {B} and each 
    y in {A} to get w_A,B. We do the same for the other clusters to get w_A,C and 
    w_A,D. We then compute W_total = W_A,B + W_A,C + W_A,D. We find te smallest number
    of elements from {W_A,B, W_A,C, W_A,D} to exceed some specified fraction of W_total.
    The clusters to which these weights connect define the clusters deemed to 'cause'
    A and the associated weights.  This method seems to produce some semi-plausible results
    but it's hard to say if it's correct and sometimes the results do not seem to
    make much sense."""
    self.max_num_clusters = max_num_clusters
    color_array = ["blue", "red", "green", "orange", "yellow", "purple", "violet", "pink", "grey", "black"]
    self.labels = self.labels + 1
    values, counts = np.unique(self.labels, return_counts=True)
    indexes = np.flip(np.argsort(counts)) 
    # counts_truncated = np.zeros(max_num_clusters)
    # counts_truncated[:max_num_clusters - 1] = indexes[:max_num_clusters - 1]
    # counts_truncated[-1] = np.sum(indexes[max_num_clusters - 1:])
    # cluster_edge_counts = np.matmul(np.transpose(counts), counts)
    # generate matrix showing transfer entropy between each cluster
    causality_matrix = np.zeros((max_num_clusters, max_num_clusters))
    edge_count_matrix = np.zeros((max_num_clusters, max_num_clusters))
    self.electrodes_by_cluster = {}
    for label_x, index_x in zip(self.labels, range(self.electrode_count)):
      label_x = int(np.where(values==label_x)[0][0])
      label_x = int(np.where(indexes==label_x)[0][0])
      if label_x > max_num_clusters - 1: # if cluster is not one of the larger ones
        label_x = max_num_clusters - 1
      # electrode lists
      if label_x in self.electrodes_by_cluster.keys():
        self.electrodes_by_cluster[label_x].append(index_x)
      else:
        self.electrodes_by_cluster[label_x] = [index_x]
      for label_y, index_y in zip(self.labels, range(self.electrode_count)):
        label_y = int(np.where(values==label_y)[0][0])
        label_y = int(np.where(indexes==label_y)[0][0])
        if label_y > max_num_clusters - 1: # if cluster is not one of the larger ones
          label_y = max_num_clusters - 1
        causality_matrix[label_x][label_y] += self.dist_matrix[index_x][index_y]
        edge_count_matrix[label_x][label_y] += 1
    causality_matrix = causality_matrix / edge_count_matrix
    causality_matrix[np.diag_indices_from(causality_matrix)] = 0 # ignore the internal transfer entropy 

    # for each cluster identify the smallest set of other clusters to explain a specified fraction of the transfer entropy
    # the "post-synaptic" cluster is defined by columns
    self.causality_map = {}
    for i in range(max_num_clusters):
      causal_clusters = []
      total_entropy = np.sum(causality_matrix[i,:])
      cummulative_entropy = 0
      while (cummulative_entropy / total_entropy) < explained_entropy_frac:
        # max_index = np.argmax(causality_matrix[:,i])
        max_index = np.argmax(causality_matrix[i,:])
        causal_clusters.append((max_index, causality_matrix[i,max_index]))
        cummulative_entropy += causality_matrix[i, max_index]
        causality_matrix[i, max_index] = 0
      self.causality_map[i] = causal_clusters
    
  def SpikeSet_ConnectivityMovie(self, video_params):
    """This function creates a movie showing how the connectivity of the organoid 
    evolves over time.
    
    Arguments:
      grouping_size: how many bins to look at when computing the connectivity
      grouping step: how many bins to skip when moving from one analysis step to another analysis 
      distance_metric: how to compute the distance between electrode tiem series
    Returns:
      Nothing"""

    self.average_clustering_array = []
    grouping_size = video_params["grouping_size"]
    grouping_step = video_params["grouping_step"]
    every_n = video_params["every_n"]
    num_edges = video_params["num_edges"]
    exponential_coefficient = video_params["exponential_coefficient"]
    use_exp = video_params["use_exp"]
    normalize = video_params["normalize"]
    distance_metric = video_params["distance_metric"]


    time_steps_total = len(self.spike_format_convolved)
    index_prev = 0
    connectivity_frame = 0
    im_array = []
    # generate each frame of connectivity
    for index in range(grouping_size, time_steps_total, grouping_step): # iterate to make each frame of movie
      self.SpikeSet_GenerateDistanceMatrix(use_binned=True, 
      data_type="electrodes", distance_metric = distance_metric, remove_outliers=False, 
      outlier_percentile = 96, timesteps=[index_prev, index])
      self.SpikeSet_Generate_Graph(every_n = every_n, num_edges = num_edges, exponential_coefficient = exponential_coefficient, use_exp=use_exp, normalize=normalize)#, exponential_coefficient = 1*10**(-14))
      self.SpikeSet_Plot_Graph(connectivity_frame=connectivity_frame)
      index_prev = index
      connectivity_frame += 1

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    im = cv2.imread("connectivity_0.png")
    h,w,c = im.shape
    path = self.params["results_path"] + "Connectivity_Movie/" + str(self.save_request_number) + "/"
    isExist = os.path.exists(path)
    if not isExist:
      os.makedirs(path)
    # make small worldness plot
    clustering_coefficient_plot_file_name = path +"average_clustering.png"
    plt.plot(range(grouping_size, time_steps_total, grouping_step), self.average_clustering_array)
    plt.xlabel("Time Steps")
    plt.ylabel("Average Clustering Coefficient")
    plt.title("Average Clustering Coefficient Over Time")
    plt.savefig(clustering_coefficient_plot_file_name, dpi=300, bbox_inches='tight')
    plt.clf()

    video_name = path + "connectivity_grouping_size_" + str(grouping_size) + "_grouping_step_" + str(grouping_step) + ".mp4"
    video = cv2.VideoWriter(video_name, fourcc, 5, (w, h))

    for i in range(connectivity_frame):
      file_name = "connectivity_" + str(i) + ".png"
      img = cv2.imread(file_name)
      video.write(img)
    cv2.destroyAllWindows()
    video.release()
    log_file = path + "log.txt"
    json.dump(video_params, open(log_file,'w'))
    
    # fig, ax = plt.subplots()
    # ani = animation.ArtistAnimation(fig, im_array, interval=50, blit=True,
    #                             repeat_delay=1000)
    # ani.save("movie.mp4")
  
  # def SpikeSet_PredictActivity(self, training_steps, validation_steps, test_steps, params, early_stop=None, with_ann=True):
  #   # make neural networks
  #   for node, connecting_clusters in self.causality_map.items():
  #     input_electrodes = []
  #     for cluster_object in connecting_clusters:
  #       cluster = cluster_object[0]
  #       input_electrodes.append(self.electrodes_by_cluster[cluster])
  #     output_electrodes = np.array(self.electrodes_by_cluster[node])
  #     input_electrodes = np.array(input_electrodes[0])
  #     spike_format = self.spike_format
  #     if params["use_binned"]:
  #       spike_format = self.spike_format_convolved
  #     spike_format_train = spike_format[training_steps[0]:training_steps[1]]
  #     spike_format_validation = spike_format[validation_steps[0]:validation_steps[1]]
  #     self.spike_format_test = spike_format[test_steps[0]:test_steps[1]]
  #     # if with_ann:
  #       # run_ann = RunANN(spike_format_train, spike_format_validation, self.spike_format_test, input_electrodes, output_electrodes, params)
  #       # run_ann.train_epochs() # train neural network on data
  #       # self.causality_map[node] = (self.causality_map[node], run_ann)
  #     # else:
  #     self.causality_map[node] = (self.causality_map[node], 0)
  #     if early_stop is not None:
  #       if node == early_stop:
  #         return

  def SpikeSet_PlotRelations(self, node_size=20, arrowsize=20, exponential_coefficient=None, normalize = False):
    graph = np.zeros((self.max_num_clusters, self.max_num_clusters))
    for key, value in self.causality_map.items():
      for val in value[0]:
        graph[key][val[0]] = val[1]
    G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    print("G")
    edges,weights = zip(*nx.get_edge_attributes(G,'weight').items())
    print("edges, weights", weights)
    if exponential_coefficient is not None:
      weights =  tuple(math.e ** (exponential_coefficient * np.asarray(weights)))
    if normalize:
      weights = (weights - np.mean(weights)) / np.std(weights)
    nx.draw(G, pos = nx.spring_layout(G), node_size = node_size, edgelist = edges, width=weights, node_color = self.color_array[:G.number_of_nodes()], arrows=True, arrowsize = arrowsize)

  def SpikeSet_Generate_Graph(self, every_n = 1, exponential_coefficient = 1*10**(-10), num_edges = 30, use_exp=False, normalize=False,pos=[]):
    """Use the NetworkX library to create a graph from matrix of distances or weights.
    Makes instance variable of graph.
    
    Arguments:
      every_n: integer specifying how often to choose electrode to include in graph, used to speedup computation and improve visualization
      exponential_coefficient: used to enhance disparity between large and small weights/distances to improve visualization
      num_edges: how many edges to show
      use_exp: whether to exponentially scale the distance values to make it easier to visualize
      normalize: whether to normalize the distance matrix values for visualization purposes
    Returns:
      None
    """
    self.every_n = every_n
    self.graph_params = {}
    self.graph_params["every_n"] = self.every_n
    self.graph_params["exponential_coefficient"] = exponential_coefficient
    self.graph_params["num_edges_to_show"] = num_edges
    self.graph_params["use_exp"] = use_exp
    self.graph_params["normalize"] = normalize
    self.graph_params["distance_metric"] = self.distance_metric
    dist_matrix = self.dist_matrix[::every_n].todense()
    dist_matrix = np.transpose(np.transpose(dist_matrix)[::every_n])
    self.G_weights = dist_matrix
    self.G = nx.from_numpy_array(dist_matrix)
    self.G.remove_edges_from(nx.selfloop_edges(self.G))
    ci,q=bct.community_louvain(dist_matrix)
    ccoeff = sum(bct.algorithms.clustering_coef_wu(dist_matrix))/len(dist_matrix)
    char_path = bct.algorithms.charpath(dist_matrix)
    bc = bct.algorithms.betweenness_wei(dist_matrix)
    
    self.graph_params["q"]=q
    pcoeff=bct.participation_coef(dist_matrix,ci)
    # mz=bct.module_degree_zscore(dist_matrix,ci)
    
    # #electrode positions
    pos2 = {}
    i = 0
    for electrode, index in zip(self.electrode_vector[::self.every_n], range(0, self.electrode_count, self.every_n)):
      row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
      x = row['x']#plot_scale * 1.0 * row["x"]
      y = row['y']#plot_scale * 1.0 * (self.y_max - row["y"])
      pos2[i] = (x,y)
      i += 1
    
    self.relevant_pos=pos2
    #place pcoeff/mz in electrode position:
    val=[]
    for i in range(len(pos)):
        for j in range(len(pos2)):
            if pos[i]==pos2[j]:
                val.append(i)
    
    self.graph_params["ccoeff"]=ccoeff
    self.graph_params["char_path"]=char_path
    self.graph_params["bc"]=bc

#     set_trace()
    # pcoeff_full=np.empty((len(pos)))
#     mz_full=np.empty((len(pos)))
    # pcoeff_full[:]=np.nan
#     mz_full[:]=np.nan
    # pcoeff_full[val]=pcoeff
#     mz_full[val]=mz
    self.graph_params["pcoeff_relevant"]=pcoeff
#     self.graph_params["mz_relevant"]=mz
    # self.graph_params["pcoeff"]=pcoeff_full
# #     self.graph_params["mz"]=mz_full
    self.graph_params["nw_degree"]=nx.degree(self.G,weight='weight')
#     self.graph_params["nw_density"]=nx.density(self.G)
    self.graph_params["num_edges"]=self.G.number_of_edges()
    self.graph_params["num_nodes"]=self.G.number_of_nodes()

    if self.distance_metric == "transfer_entropy":
      graph_type = nx.MultiDiGraph
      use_exp = True
      normalize = True
    if self.distance_metric == "mutual_information":
      num_edges = num_edges * 2 
      graph_type = nx.Graph
    if self.distance_metric == "cosine":
      num_edges = num_edges * 2 
      dist_matrix = 1 - dist_matrix
      graph_type = nx.Graph
    else: #Alternate measures - Alon 08/06/23
      graph_type = nx.Graph 
    if use_exp == True:
      dist_matrix = math.e ** (exponential_coefficient * dist_matrix)
      
    vals = np.sort(dist_matrix.flatten())
    vals_diff = np.abs(vals[1:] - vals[:-1])
    noise_magnitude = 0.95 * np.min(vals_diff[np.nonzero(vals_diff)]) / 2
    dist_matrix = dist_matrix + noise_magnitude * np.random.uniform(-1,1,np.shape(dist_matrix))
    dist_matrix[np.diag_indices_from(dist_matrix)] = np.min(dist_matrix) - 100
    vals = np.sort(dist_matrix.flatten())
    threshold = vals[-1 * num_edges]
    indexes_prune = dist_matrix < threshold
    #print("indexes prune:", np.shape(indexes_prune))
    relevant = dist_matrix[dist_matrix >= (threshold)]
    if normalize:
      relevant = (relevant - np.mean(relevant)) / (np.std(relevant) + 0.0000000000001)
    dist_matrix[dist_matrix >= (threshold)] = relevant

    vals = np.sort(dist_matrix.flatten())
    self.threshold = threshold#vals[-1 * num_edges:]

    self.relevant_G_weights = dist_matrix
    self.relevant_G = nx.from_numpy_array(dist_matrix, create_using=graph_type)
    self.relevant_G.remove_edges_from(nx.selfloop_edges(self.relevant_G))

    if self.distance_metric == "cosine": # to make more "similar" electrodes have a larger weight
      dist_matrix = 1 - dist_matrix
      self.threshold = np.sort(dist_matrix.flatten())[-1 * num_edges]


  def SpikeSet_shortest_path_length(self):
    """Determine the shortest path length of the existing graph object.
    Returns:
      shortest_path_length: average shortest path length
      """
    G = nx.from_numpy_array(self.dist_matrix)
    shortest_path_length = nx.average_shortest_path_length(G, True)
    return shortest_path_length
  
  def SpikeSet_clustering_coeff(self):
    """Find the average clustering coefficient.
    Returns:
      mean_clustering_coefficient: the average clustering coefficient of the nodes of the graph"""
    
    mean_clustering_coefficient = np.mean(bct.clustering_coef_wu(self.dist_matrix))
    return mean_clustering_coefficient
  
  @CountCalls
  def SpikeSet_Plot_Graph(self, connectivity_frame = None, vae_container=None, plot_scale=1, save_individual_frame = False,pos={},full_graph=[],spec_graph=[],dataset=1):
    """Plots the computational graph in which nodes are electrodes that are in 
    the same spatial formation as they are on the mea. Edge thickness is indicative 
    of the weight/distance metric between the electrodes.

    Arguments:
      connectivity_frame: default None, used to specify the name to save the figure as if it is used to create connectivity movie
      vae_container: vae_container object trained on SpikeSet data (not used for this aspect of code)
      plot_scale: controls how big to make image
      save_individual_frame: False by default, whether to save the fame to a path based on the params arguments to the SpikeSet class
    Returns:
      None
     """
     
    pos2=self.relevant_pos
        
#     # Highlight specific Electrode positions
    if len(self.params["relevant_elecs"])>0 and dataset==2:
        self.relevant_elecs=self.params["relevant_elecs"]    
        pos_spec = {}
        i=0
        for electrode, index in zip(self.relevant_elecs, range(0, len(self.relevant_elecs), self.every_n)):
          try:
              row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
              x = row['x']#plot_scale * 1.0 * row["x"]
              y = row['y']#plot_scale * 1.0 * (self.y_max - row["y"])
              pos_spec[i] = (x,y)
              i += 1
          except:
            continue
        self.spec_pos=pos_spec
    
#     #Highlight only relevant channels:
#     self.relevant_chans=self.params["relevant_chans"]    
#     chan_idx=np.where(np.in1d(self.channel_vector,self.relevant_chans))[0]
#     relevant_xy=np.array(list(pos2.values()))[chan_idx]
#     relevant_adjMat=self.dist_matrix[chan_idx,:][:,chan_idx]
#     #Specific channels pos
#     k=0
#     pos_spec={}
#     for j in chan_idx:
#         pos_spec.update({k:tuple(relevant_xy[k])})
#         k+=1
    
    self.G_max=full_graph
#     nx.set_node_attributes(self.G_max, pos, "pos")
#     nx.set_node_attributes(self.relevant_G, pos2, "pos")
    # prune low weight edges
    edge_weights = nx.get_edge_attributes(self.relevant_G,'weight')
    self.relevant_G.remove_edges_from((e for e, w in edge_weights.items() if w < self.threshold))
    average_clustering = nx.average_clustering(self.relevant_G)
    self.graph_params["average_clustering"] = average_clustering
    # display graph
    try:
        edges,weights = zip(*nx.get_edge_attributes(self.relevant_G,'weight').items())
    except:
        edges,weights = zip(*nx.get_edge_attributes(self.G,'weight').items())

    fig=plt.figure()
    ax=plt.gca()
    #Plot electrode layout
    if len(self.G_max)>0:
        g1=nx.draw_networkx_nodes(self.G_max,pos=pos,node_size=1,alpha=0.6,node_color='xkcd:pale orange',ax=ax)
        g1.set_zorder(0)
    if self.distance_metric == "transfer_entropy":
      nx.draw(self.relevant_G, pos=pos2, node_color='r', node_size=2, edgelist=edges, width=weights, edge_color = "black", arrowsize=15)
    else:
#       try:
        #Plot Functional Graph
        g2=nx.draw_networkx_nodes(self.relevant_G,pos=pos2,node_size=2,node_color='xkcd:scarlet',ax=ax)
        g2.set_zorder(3)
        g3=nx.draw_networkx_edges(self.relevant_G, pos=pos2, edgelist=edges, width=np.array(weights)/2, edge_color = "black",ax=ax) #log of weights
        g3.set_zorder(1)
        #Plot Sam's electrodes
        if dataset==2:
            g4=nx.draw_networkx_nodes(spec_graph,pos=pos_spec,node_size=10,node_color='#0ffef9',ax=ax)  
            g4.set_zorder(2)
#       except:
#         print('This network has no edges')
    if save_individual_frame:
      static_connectivity_folder = self.params["results_path"] + "static_connectivity_folder/"
      isExist = os.path.exists(static_connectivity_folder)
      if not isExist:
          os.makedirs(static_connectivity_folder)
      image_directory = static_connectivity_folder + "Plot_Graph/" 
      isExist = os.path.exists(image_directory)
      if not isExist:
          os.makedirs(image_directory)
      image_path = image_directory + "connectivity_map"
      # plt.clf()
      fig.savefig(image_path, dpi=300, bbox_inches='tight')
      plt.close()
      plt.close('all')
      log_file = image_directory + "log.txt"
      json.dump(self.graph_params, open(log_file,'w'))
      self.save_request_number += 1

    
    #Plot pcoeff and mz subgraphs:
    
    fig2=plt.figure()
    ax2=plt.gca()
    plt.title('Participation Coefficient',fontsize='15')
#     g1=nx.draw_networkx_nodes(self.G_max,pos=pos,node_size=1,alpha=0.6,node_color='xkcd:light grey')
#     g1.set_zorder(0)
    
    ec = nx.draw_networkx_edges(self.relevant_G,width=np.array(weights)/2, pos=pos2)
    try:
        ec.set_zorder(1)
    except:
        ec = nx.draw_networkx_edges(self.G,width=np.array(weights)/2, pos=pos2)
        ec.set_zorder(1)
    cmap=plt.cm.plasma
    vmin=0;vmax=1
    
    
    nc=nx.draw_networkx_nodes(self.relevant_G, pos=pos2, node_color=self.graph_params["pcoeff_relevant"], 
                                node_size=2, cmap=cmap,vmin=0,vmax=1)
    try:
        nc.set_zorder(2)
    except:
        nc=nx.draw_networkx_nodes(self.G, pos=pos2, node_color=self.graph_params["pcoeff_relevant"], 
                                node_size=2, cmap=cmap,vmin=0,vmax=1)        
        nc.set_zorder(2)
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm,ax=ax2)
    image_path2 = image_directory + "pcoeff"
    fig2.savefig(image_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig3=plt.figure()
    ax3=plt.gca()
    plt.title('Module Z Score',fontsize='15')
#     g1=nx.draw_networkx_nodes(self.G_max,pos=pos,node_size=1,alpha=0.6,node_color='xkcd:light grey')
#     g1.set_zorder(0)

    ec2 = nx.draw_networkx_edges(self.relevant_G,width=np.array(weights)/2, pos=pos2)
    try:
        ec2.set_zorder(1)
    except:
        ec2= nx.draw_networkx_edges(self.G,width=np.array(weights)/2, pos=pos2)
        ec2.set_zorder(1)    
    cmap=plt.cm.viridis
    vmin2=-2;vmax2=4
    nc2=nx.draw_networkx_nodes(self.relevant_G, pos=pos2, node_color=self.graph_params["mz_relevant"], 
                                node_size=2, cmap=cmap,vmin=vmin2,vmax=vmax2)
    try:
        nc2.set_zorder(2)
    except:
        nc2=nx.draw_networkx_nodes(self.G, pos=pos2, node_color=self.graph_params["mz_relevant"],                                                                    node_size=2, cmap=cmap,vmin=vmin2,vmax=vmax2)

        nc2.set_zorder(2)

    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin2, vmax=vmax2))
    sm1._A = []
    
    plt.colorbar(sm1,ax=ax3)

    image_path3 = image_directory + "mz"
    fig3.savefig(image_path3, dpi=300, bbox_inches='tight')
    plt.close()

    if connectivity_frame is not None:
      self.average_clustering_array.append(average_clustering)
      file_name = "connectivity_" + str(connectivity_frame) + ".png"
      plt.savefig(file_name, dpi=300, bbox_inches='tight')
      plt.clf()
      # im = plt.show()
    elif vae_container is not None:
        results_path = vae_container.params["results_path"]
        # make subdirectory for the image
#         date_time = datetime.now()
#         date_time = date_time.strftime("%m_%d_%Y__%H_%M_%S")
        image_directory_path = results_path + "generated_connectivity/"
        isExist = os.path.exists(image_directory_path)
        if not isExist:
            os.makedirs(image_directory_path)
        # store image
        image_path = image_directory_path + "generated_connectivity"
        ax = plt.gca()
        ax.set_aspect("equal")
#         plt.savefig(image_path, dpi=300, bbox_inches='tight')
        # store log file
#         log_file = image_directory_path + "log.txt"
#         json.dump(self.graph_params, open(log_file,'w'))
        plt.close()
#     return pos2


  @CountCalls
  def SpikeSet_Cluster(self, input_data, input_data_dict, eps=1.3, min_samples=1):
    """Place electrodes into clusters using DBSCAN algorithm. Displays clusters 
    with each cluster indicated by a different node color. The electrodes are 
    displayed in the same spatial relationship as they are found on the mea. Only 
    the clusters containing the largest number of electrodes are shown.
    NOTE: The default values were determined by trial and error.

    Arguments:
      input_data: data used to generate clusters
      input_data_dict: dictionary explaining the parameters used to generate the input data (ex. if input data from isomap, it specifies isomap parameters)
      eps: maximum distance between two samples for one to be considered in the
      neighborhood of the other
      min_samples: number of samples in a neighborhood, including the point itself for the pont to be determined a core point.
    Returns:
      None
    """
    self.eps = eps
    self.min_samples = min_samples
    self.params["eps"] = self.eps
    self.params["min_samples"] = self.min_samples

    self.cluster_params = {}
    self.cluster_params["eps"] = eps
    self.cluster_params["min_samples"] = min_samples
    self.cluster_params["input_data_dict"] = input_data_dict
 


    clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(input_data) # 1.4, 6
    labels = clustering.labels_
    self.labels = labels
    values, counts = np.unique(labels, return_counts=True)
    self.color_array = ["blue", "red", "green", "orange", "yellow", "purple", "violet", "pink", "grey", "black"]
    labels = labels + 1
    values, counts = np.unique(labels, return_counts=True)
    indexes = np.flip(np.argsort(counts)) 
    print("Number of Clusters:", len(values))
    for electrode, label, index in zip(self.electrode_vector, labels, range(self.electrode_count)):
      if self.non_outlier_indexes[index]:
        row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
        x = row["x"]
        y = self.y_max - row["y"]
        label = int(np.where(values==label)[0][0])
        label = int(np.where(indexes==label)[0][0])
        if label > len(self.color_array) - 6 or label == -1: # if cluster is not one of the larger ones
          label = len(self.color_array) - 1
          continue
        plt.scatter(x, y, color = self.color_array[label])
        plt.xlim(self.x_min, self.x_max)
        plt.ylim(self.y_min, self.y_max)
    static_connectivity_folder = self.params["results_path"] + "static_connectivity_folder/"
    isExist = os.path.exists(static_connectivity_folder)
    if not isExist:
        os.makedirs(static_connectivity_folder)
    image_directory = static_connectivity_folder + "Plot_Cluster/" + str(self.SpikeSet_Cluster.num_calls) + "/"
    os.makedirs(image_directory)
    image_path = image_directory + "cluster"
    # plt.clf()
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_file = image_directory + "log.txt"
    json.dump(self.cluster_params, open(log_file,'w'))

  @CountCalls
  def SpikeSet_GraphDistances(self):
    """This function plots how the strength of connection between electrodes varies based on the physical distance between the electrodes.
    Arguments:
        None
    Returns:
        None """
    self.graph_distances_params = {}
    self.graph_distances_params["distance_metric"] = self.distance_metric
    D, B = bct.distance_wei(self.dist_matrix)
    spatial_distances = distance_matrix(self.electrode_positions, self.electrode_positions)
    # plt.scatter(spatial_distances.flatten(), D.flatten())

    static_connectivity_folder = self.params["results_path"] + "static_connectivity_folder/"
    isExist = os.path.exists(static_connectivity_folder)
    if not isExist:
        os.makedirs(static_connectivity_folder)
    image_directory = static_connectivity_folder + "Spatial/" + str(self.SpikeSet_GraphDistances.num_calls) + "/"
    os.makedirs(image_directory)
    image_path = image_directory + "connectivity_v_spatial_graph"
    # plt.clf()
    plt.scatter(spatial_distances.flatten(), D.flatten())
    plt.xlabel("Spatial Distance")
    plt.ylabel("Connectivity Distance")
    plt.title("Connectivity Distances Versus Spatial Distance")
    plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_file = image_directory + "log.txt"
    json.dump(self.graph_distances_params, open(log_file,'w'))
  
  @CountCalls
  def SpikeSet_community_louvain(self):
    """This function calls the Brain Connectivity Toolboxes community-louvain community detection method."""
    plot_params = {}
    plot_params["distance_metric"] = self.distance_metric
    c, q = bct.community_louvain(self.dist_matrix)
    plot_params["modularity_metric"] = q
    print("c:", c)
    print("q:", q)

    # make color values
    colors = []
    step = 1 / len(c)
    for i in range(len(c)):
      value = int(i * step)
      colors.append((value, value, value))
    num_communities = np.max(c)
    for electrode, c_val, index in zip(self.electrode_vector, c, range(self.electrode_count)):
      row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
      x = row["x"]
      y = self.y_max - row["y"]
      plt.scatter(x, y, color = colors[c_val - 1])
      plt.xlim(self.x_min - 0.1 * self.x_min, self.x_max + 0.1 * self.x_max)
      plt.ylim(self.y_min - 0.1 * self.y_min, self.y_max + 0.1 * self.y_max)
    static_connectivity_folder = self.params["results_path"] + "static_connectivity_folder/"
    isExist = os.path.exists(static_connectivity_folder)
    if not isExist:
        os.makedirs(static_connectivity_folder)
    image_directory = static_connectivity_folder + "CommunityLouvain/" + str(self.SpikeSet_community_louvain.num_calls) + "/"
    isExist = os.path.exists(image_directory)
    if not isExist:
        os.makedirs(image_directory)
    image_path = image_directory + "community_louvain_graph"
#     plt.savefig(image_path, dpi=300, bbox_inches='tight')
    plt.close()
#     log_file = image_directory + "log.txt"
#     json.dump(plot_params, open(log_file,'w'))
    return q

  
  def SpikeSet_PlotElectrodes(self, mask):
    """Plot all electrodes with True mask value

    Arguments:
      mask: binary vector of length equal to self.electrode_count
    Returns:
      None
    """
    for electrode, index in zip(self.electrode_vector,  range(self.electrode_count)):
      if mask[index]:
        row = self.data.loc[self.data['electrode'] == electrode].iloc[0]
        x = row["x"]
        y = self.y_max - row["y"]
        plt.scatter(x, y)
    plt.show()


  def SpikeSet_GenerateMask(self, num_masks):
    """Generates binary masks in which electrodes in a given cluster are masked on and
    electrodes in other clusters are masked off. Masks do not have any overlap between one another.
    
    Arguments:
        num_masks: number of masks such that the clusters are chosen to be masks in descending order of cluster size with the masks having 
        no overlap between one another.
    Returns:
        mask_matrix: matrix in which each row is a different mask, has as many rows as
        num_masks"""
    labels = self.labels
    values, counts = np.unique(labels, return_counts=True)
    mask_matrix = np.zeros((num_masks, len(labels)))
    for i in range(num_masks):
        max_index = np.argmax(counts)
        counts[max_index] = 0
        mask = np.copy(labels)
        mask[mask != max_index] = -1
        mask[mask > -1] = 1
        mask[mask == -1] = 0
        mask_matrix[i] = mask
    return mask_matrix

    
    
    
  def SpikeSet_GenerateRaster(self, electrodes, frame_min, frame_max):
    """Generate a raster plot of the spikes from specific electrodes between specific
    time points.
    
    Arguments:
      electrodes: array of indexes specifying electrodes whose spike data should be displayed
      frame_min: specifying the earliest frame whose spike data should be displayed
      frame_max: specifying the latest frame whose spike data should be displayed
    Returns:
      None
    """
    index_min = np.min(np.where(self.time_vector >= frame_min))
    index_max = np.min(np.where(self.time_vector >= frame_max))
    tempData = self.spike_format[np.ix_(range(index_min,index_max), electrodes)] # ignores the spots where there is no spikes
    # populate corect rows of complete data with temp data
    plotData = np.zeros((frame_max - frame_min, len(electrodes)))
    indices = self.time_vector[index_min:index_max] - self.time_vector[index_min]
    plotData[indices] = tempData
    print(np.sum(plotData))
    plotData = np.transpose(plotData)

    # Loop to plot raster for each trial
    fig, ax = plt.subplots()
    for electrode_index in range(len(electrodes)):
      spike_times = [i for i, x in enumerate(plotData[electrode_index]) if x == 1]
      ax.vlines(spike_times, electrode_index - 0.5, electrode_index + 0.5)
      ax.set_xlim([0, frame_max - frame_min])
      ax.set_xlabel('Frame')
      # specify tick marks and label label y axis
      ax.set_yticks(range(len(electrodes)),electrodes)
      ax.set_ylabel('Electrode Number')
      ax.set_title('Neuronal Spike Times') 
    



  # ####**** Functions Not Debugged Or Unfinished****####
  # def SpikeSet_GenerateGraphs_Experimental(self, neighborhood_size, use_binned = True, data_type="electrodes", distance_metric = "test"):
  #   #data_type = {"electrodes", "global_activity"}
  #   # spike_format_isomap, iso = self.SpikeSet_GenerateIsomap(neighborhood_size, 
  #   # use_binned=use_binned, data_type=data_type)
  #   # # print("ISO GENERATED")
  #   dist_matrix = self.dist_matrix
  #   if distance_metric == "covariance":
  #     cov_matrix = self.SpikeSet_GetCovariance(use_convolution = True, include_zeros = True)
  #     diagonal_vector = np.reshape(np.diagonal(cov_matrix),(-1,1))
  #     normalization_matrix = np.sqrt(np.matmul(diagonal_vector, np.transpose(diagonal_vector)))
  #     print("matrix:", np.shape(normalization_matrix))
  #     dist_matrix = -1 * abs(cov_matrix)# / normalization_matrix)
  #   node_count = np.shape(dist_matrix)[0]
  #   node_vector = np.zeros(node_count)
  #   ones_vector = np.ones(node_count)
  #   # do BFS search
  #   queue = [np.argmin(np.mean(dist_matrix, axis = 1))] # choose most "connected node to start"
  #   graph_array = []
  #   while np.sum(node_vector) < node_count: # repeat until all nodes are in a graph
  #     mean_dist = np.mean(dist_matrix, axis = 1)
  #     mean_dist[node_vector == 1] = 1000
  #     queue = [np.argmin(mean_dist)]
  #     # print("Node count:", node_count)
  #     # print("Sum of node vector:", np.sum(node_vector))
  #     # queue = [np.argmin(node_vector)]
  #     print("QUEUE:", queue)
  #     graph = []
  #     node_vector_temp = np.copy(node_vector)
  #     while len(queue) > 0:
  #       # print("Queue loop")
  #       node = queue.pop(0)
  #       node_vector[node] = 1
  #       # get the neighborhood_size closest nodes to the current node
  #       neighbors = np.argsort(dist_matrix[node]) # because the current node will be returned
  #       neighbor_count = 0
  #       for neighbor in neighbors:
  #         if node_vector_temp[neighbor] == 0 and neighbor != node:
  #           neighbor_count += 1
  #         if neighbor != node: # no self connections
  #           if node_vector[neighbor] != 1:
  #             node_vector[neighbor] = 1
  #             queue.append(neighbor)
  #             graph.append(neighbor)
  #           if neighbor_count == neighborhood_size - 1:
  #             break
  #     if len(graph) != 0:
  #       graph_array.append(graph)
  #   self.graph_array = graph_array
  #   return self.graph_array
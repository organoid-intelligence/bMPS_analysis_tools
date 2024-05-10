from SpikeSet import spikeset_module
import numpy as np
import pandas as pd
import networkx as nx 
from pathlib import Path
import matplotlib.pyplot as plt

def generate_Electrode_Array(pos):
    """
    This function loads data and preprocesses it
    
    Args:
        pos: Empty position dict of x,y electrode positions
        
    Output:
        pos: Filled position dict of x,y electrode positions
        full_graphs: List of Network Graphs 
    """
    import itertools
 
    xs=np.arange(0, 2065+17.5, 17.5) #these values are taken from all the x,y's in the data files, and extrapolated to fill the array. We were not given the full array structure, so we have to construct it.
    ys=np.arange(0, 3832.5+35, 35)
    xys = np.array(list(itertools.product(xs,ys))).astype(tuple)
    xys[:,1][xys[:,0] %35 ==0]+=17.5 #interpolation of electrodes

    i=0
    full_graph=nx.Graph()
    for xy in xys:
        pos[i]=(xy[1],xy[0])
        full_graph.add_node(i)
        i+=1
    return full_graph,pos

def run_dataset1(run,run_index,well,well_index,div,div_index,path,ld_index=-1,params=[],results=[],pos=[],full_graph=[]):
    """
    Run analysis for dataset 1. The actual graphical analysis happens in the SpikeSet module.
    """
    
    spike_set=[]
    print('\n run' + str(run) +' | well '+ str(well) + ' | div ' + str(div) +'\n')
    data=pd.read_csv(path) #load csv
    data = pd.DataFrame(data=data) #convert to dataframe
    #       data = data
    params["data_path"] = path
    params["organoid_name"] = None  #"run_" + str(run) + "well_" + str(well) + "div_" + str(div)
    params["relevant_elecs"]=[]
    spike_set = spikeset_module.SpikeSet(data, params) # create SpikeSet instance
    # make the distance matrix
    spike_set.SpikeSet_GenerateDistanceMatrix(use_binned=True, data_type="electrodes", distance_metric = "mutual_information", remove_outliers=False, outlier_percentile = 96)
    # make a connectivity graph based on all frames
    spike_set.SpikeSet_Generate_Graph(every_n = 1, exponential_coefficient = 1*10**(-10), num_edges = 300, use_exp = False, normalize = True,pos=pos) # was 200
    # visualize the graph (saved to file based on the path you defined)
    pos2=spike_set.SpikeSet_Plot_Graph(plot_scale = 1, save_individual_frame=True,pos=pos,full_graph=full_graph)
        
    results["electrode_count"][run_index, well_index, div_index] = spike_set.electrode_count

    if ld_index>-1:
        results["dist_matrix"][run_index][well_index][div_index][ld_index] = spike_set.dist_matrix
        results["bct_vals"][run_index][well_index][div_index][ld_index] = {'clustering_coeff':spike_set.SpikeSet_clustering_coeff(),'community_metric':spike_set.graph_params["q"],'degree':spike_set.graph_params["nw_degree"],
                                                           'density':spike_set.graph_params["nw_density"],'num_edges':spike_set.graph_params["num_edges"],'num_nodes':spike_set.graph_params["num_nodes"],'avg spl':spike_set.graph_params["avg_spl"],'pcoeff':spike_set.graph_params["pcoeff"],
                                                           'mz': spike_set.graph_params["mz"],'nw_pos':pos2}#,'shortpath':sp} 
    else:
        results["dist_matrix"][run_index][well_index][div_index] = spike_set.dist_matrix
        results["bct_vals"][run][well_index] = {'clustering_coeff':spike_set.SpikeSet_clustering_coeff(),'community_metric':spike_set.graph_params["q"],'degree':spike_set.graph_params["nw_degree"],
                                                           'density':spike_set.graph_params["nw_density"],'num_edges':spike_set.graph_params["num_edges"],'num_nodes':spike_set.graph_params["num_nodes"],'avg spl':spike_set.graph_params["avg_spl"],'pcoeff':spike_set.graph_params["pcoeff"],
                                                            'mz': spike_set.graph_params["mz"],'nw_pos':pos2,'elec_pos':spike_set.electrode_positions,'channel_ids':spike_set.channel_vector,'elec_ids':spike_set.electrode_vector,
                                                            'communicability':spike_set.graph_params['communicability']}#,'shortpath':sp} 
    return results


def run_dataset2(run,well,well_index,recording_num,load_path,path,params=[],results=[],pos=[],full_graph=[]):
    
    """
    Run analysis for dataset 2
    """
    
    spike_set=[]
    relevant_elecs=[[3269,9114,20516],#[10337,5467,3269,15899,15543,12136],
                    [21149,20930,11469,24964,12059],#[7611,20492,20930,14099,4328,20379,2767,24496,25991,22726,11764,20011,5624,18509],
                    [11328,3971,20825,15344,4155,21536,21354,6106,4969,22822,22304,22820,15638,14297,20598,22151,5714,9717,19936,19620,12431,22808,21243,11257,21902,11784,19549,18499,20325],
                    [15519,10191,24502,1865,6024,16393,1779,1560,6902,24000],
                    [16437,21597,25382,25165,7173,3002,2325,26101,21916,2154,25661,1234,348,4320,3438,3219],
                    [18918,2659,22955,957,23819,872,5994,26071,22392,3842,17271,2769,19228,24279,22655,21374,23333,298,6590,26041,24634,21086,13329,25858,10693,3374,24990,24166,22073,23176,5270,25762,20498,21593,23994,6926,1833,4459,4915,5956]]
    
    print('\n well '+ str(well) + ' | recording ' + str(recording_num+1) +'\n')
    data=pd.read_csv(path)
    data = pd.DataFrame(data=data)
    
    #extract relevant channels
    relevant_data=data.copy().iloc[np.where(np.in1d(data['electrode'].values,relevant_elecs[well_index]))[0]]
#     relevant_data2=data.copy().iloc[np.where(np.in1d(data['channel'].values,relevant_chans[well_index]))[0]]

    params["data_path"] = path
    params["organoid_name"] = None  #"run_" + str(run) + "well_" + str(well) + "div_" + str(div)
    params["relevant_elecs"]=relevant_elecs[well_index]

    spike_set = spikeset_module.SpikeSet(data, params) # create SpikeSet instance
    spike_set_relevant = spikeset_module.SpikeSet(relevant_data,params)
    # make the distance matrix
    spike_set_relevant.SpikeSet_GenerateDistanceMatrix(use_binned=True, data_type="electrodes", distance_metric = "mutual_information", remove_outliers=False, outlier_percentile = 96)
    spike_set.SpikeSet_GenerateDistanceMatrix(use_binned=True, data_type="electrodes", distance_metric = "mutual_information", remove_outliers=False, outlier_percentile = 96)
    # make a connectivity graph based on all frames
    spec_graph=nx.from_numpy_array(spike_set_relevant.dist_matrix)
    spike_set.SpikeSet_Generate_Graph(every_n = 1, exponential_coefficient = 1*10**(-10), num_edges = 300, use_exp = False, normalize = True,pos=pos) # was 200
    # visualize the graph (saved to file based on the path you defined)
    pos2=spike_set.SpikeSet_Plot_Graph(plot_scale = 1, save_individual_frame=True,pos=pos,full_graph=full_graph,spec_graph=spec_graph,dataset=2)
    
    results["dist_matrix"][run][well_index][recording_num] = spike_set.dist_matrix
    results["bct_vals"][run][well_index][recording_num] = {'clustering_coeff':spike_set.SpikeSet_clustering_coeff(),'community_metric':spike_set.graph_params["q"],'degree':spike_set.graph_params["nw_degree"],
                                                           'density':spike_set.graph_params["nw_density"],'num_edges':spike_set.graph_params["num_edges"],'num_nodes':spike_set.graph_params["num_nodes"],'avg spl':spike_set.graph_params["avg_spl"],'pcoeff':spike_set.graph_params["pcoeff"],
                                                            'mz': spike_set.graph_params["mz"],'nw_pos':pos2,'elec_pos':spike_set.electrode_positions,'channel_ids':spike_set.channel_vector,'elec_ids':spike_set.electrode_vector}
                                                            #'communicability':spike_set.graph_params['communicability']}#,'smallworld':spike_set.graph_params['smallworld']}#,'shortpath':sp} }#,'shortpath':sp} 
    return results
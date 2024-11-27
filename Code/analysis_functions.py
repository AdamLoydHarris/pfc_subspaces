import os, sys, pickle, time, re, csv, io, math
import numpy as np
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import Lasso
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from tqdm import tqdm

def get_sessions(mouse_recday, Data_folder):
    
    Tasks = np.load(Data_folder + "Task_data_" + mouse_recday + ".npy")
    sessions_to_try = range(len(Tasks))
    sessions = []
    for session in sessions_to_try:
        try:
            data_neurons = np.load(
                Data_folder + "Neuron_" + mouse_recday + "_" + str(session) + ".npy"
            )
            sessions.append(session)
        except:
            print("Exception: " + mouse_recday + "_" + str(session) + ".npy  not found")
        
    return sessions, Tasks

def get_cell_types(mouse_recday, Data_folder):
    
    # cell_properties_of_interest:
    # phase, state, place
    
    Phase_state_place_anchoring=np.load(Data_folder+'Phase_state_place_anchored_'+mouse_recday+'.npy')
    
    # first order
    phase=Phase_state_place_anchoring[:,0]
    state=Phase_state_place_anchoring[:,1]
    place=Phase_state_place_anchoring[:,2]
    anchored=Phase_state_place_anchoring[:,3]
    
    # second order
    phase_state=np.logical_and(phase==True,state==True)
    no_phase_no_state=np.logical_and(phase==False,state==False)
    place_phase=np.logical_and(place==True,phase==True)
    no_place_no_phase=np.logical_and(place==False,phase==False)
    place_state = np.logical_and(place==True,state==True)
    no_place_no_state =np.logical_and(place==False,state==False)
    
    
    # third order
    place_phase_state = np.logical_and(phase_state==True,place==True)
    place_phase_no_state = np.logical_and(place_phase==True,state==False)
    place_no_phase_no_state = np.logical_and(no_phase_no_state==True,place==True)
    place_no_phase_state = np.logical_and(place_state==True,phase==False)
    
    no_place_no_phase_no_state = np.logical_and(no_phase_no_state==True,place==False)
    no_place_no_phase_state = np.logical_and(no_place_no_phase==True,state==True)
    no_place_phase_no_state = np.logical_and(no_place_no_state==True,phase==True)
    no_place_phase_state = np.logical_and(phase_state==True,place==False)                                 
                                        
    cell_type_dict = {
        
        'place_phase_state': place_phase_state,
        'place_phase_no_state': place_phase_no_state,
        'place_no_phase_no_state': place_no_phase_no_state,
        'place_no_phase_state': place_no_phase_state,
        
        'no_place_no_phase_no_state': no_place_no_phase_no_state,
        'no_place_no_phase_state': no_place_no_phase_state,
        'no_place_phase_no_state': no_place_phase_no_state,
        'no_place_phase_state':no_place_phase_state
                
    }   
    
    return cell_type_dict


def unroll_listoflists(l):
    flat_list = [item for sublist in l for item in sublist]
    return(flat_list)

def smooth_circ(xx, sigma=10, axis=0):
    x_smoothed = gaussian_filter1d(np.hstack((xx, xx, xx)), sigma, axis=axis)[
        len(xx) : int(len(xx) * 2)
    ]
    return x_smoothed

def mean_neuron_session(data_neuron):
    
    return np.mean(data_neuron, 0)

def std_err(data_neuron):
    return smooth_circ(sem(data_neuron, axis=0))

def occupency_map(raw_locs, maze_mat, place_plot):
    occupency_map = maze_mat.copy()
    for i in range(1,22):
        occupency_map[place_plot[i]]=np.sum(raw_locs==i)
        
    return occupency_map

def spikes_in_place_map(raw_spikes, raw_locs, maze_mat, place_plot):
    max_ind = min(len(raw_spikes), len(raw_locs))
    spikes_map=maze_mat.copy()
    raw_locs = raw_locs[:max_ind]
    raw_spikes = raw_spikes[:max_ind]
    for i in range(1,22):
        locs_ = raw_locs==i
        spikes_ = np.sum(raw_spikes[locs_])
        spikes_map[place_plot[i]] = spikes_
        
    return(spikes_map)

def place_map(raw_spikes, raw_locs, maze_mat, place_plot):
    occupency_ = occupency_map(raw_locs, maze_mat, place_plot)
    spikes_in_place_ = spikes_in_place_map(raw_spikes, raw_locs, maze_mat, place_plot)
    place_map = np.divide(spikes_in_place_, occupency_)
    return(place_map)
    
def get_raster_arrays(raw_spikes_state):
    spike_events_ = [np.where(raw_spikes_state[i]>0)[0] for i in range(len(raw_spikes_state))]
    trial_len_ = np.asarray([[len(raw_spikes_state[i])] for i in range(len(raw_spikes_state))])
    return spike_events_, trial_len_

def get_data_for_state(Neuron_raw, Location_raw, state, trial_times):

    Neuron_state = []
    Location_state = []
    Neuron_state_pertrial = []
    state_dic = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3
    }
    
    for trial in trial_times:
        trial_inds = (trial/25).astype(int) # 25 is scaling factor between tria samling rate and neuron/locs sampling rate
        state_inds = [trial_inds[state_dic[state]], trial_inds[state_dic[state]+1]]

        neuron_ = Neuron_raw[state_inds[0]:state_inds[1]]
        locs_ = Location_raw[state_inds[0]:state_inds[1]]
        Neuron_state.extend(neuron_)
        Neuron_state_pertrial.append(neuron_)
        Location_state.extend(locs_)
        
    Neuron_state_arr = np.array(Neuron_state)
    Location_state_arr = np.array(Location_state)    
        
    return Neuron_state_arr, Location_state_arr, Neuron_state_pertrial


def get_max_time(raster, conversion=40):
    max_time=0
    for i in raster:
        if len(i)>0:
            if max(i)>max_time:
                max_time=max(i)
    bins = np.array(list(range(max_time)))
    bins_s = bins/conversion
    ind_of_max_s = 0
    max_s = 0
    for i , j in enumerate(bins_s):
        if i%conversion==0:
            ind_of_max_s  = i
            max_s = j.astype(int)
    return max_time/conversion, bins, ind_of_max_s, max_s

def make_example_cell_plot(mouse_recday, Neuron_raw, Location_raw, Trial_times, data_neurons, task, save=True):
    eventplot_linewidth_ = 0.1
    fontsize=8
    tickwidth = 1
    ticklength = 5
    eventplot_line_colour_ = 'gray'
    heatmap_colormap = 'viridis'
    global maze_mat, loc_mapping_for_plot, cell_type, cell_pdf
    states = ["A", "B", "C", "D"]
    
    fig = plt.figure(constrained_layout=True, figsize = (9,9))
    gs = GridSpec(6, 4, figure=fig)

    ax0 = fig.add_subplot(gs[0:2, 0:2])
    place_mapx= place_map(Neuron_raw[neuron,:], Location_raw, maze_mat, loc_mapping_for_plot)           
    ax0.matshow(place_mapx, cmap=heatmap_colormap)
    ax0.set(title='Aggregate place map')
    ax0.set_xticks([])
    ax0.set_yticks([])
    
    for ind, goal in enumerate(task):
        goal_locs = np.array(loc_mapping_for_plot[goal])[::-1]-0.5
        ax0.add_patch(
         patches.Rectangle(
             goal_locs,
             1,
             1,
             edgecolor='red',
             fill=False,
             lw=2
         ) )
        ax0.text(goal_locs[0]+0.3, goal_locs[1]+0.7, states[ind], fontsize=18, color='white')

    ax1 = fig.add_subplot(gs[2, 0])
    state=states[0]
    Neuron_state_arr, Location_state_arr,  _ = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    place_mapx= place_map(Neuron_state_arr, Location_state_arr, maze_mat, loc_mapping_for_plot)           
    ax1.matshow(place_mapx, cmap=heatmap_colormap)
    ax1.set(title=state)
    ax1.set_xticks([])
    ax1.set_yticks([])

    goal_locs = np.array(loc_mapping_for_plot[task[0]])[::-1]-0.5
    ax1.add_patch(
         patches.Rectangle(
             goal_locs,
             1,
             1,
             edgecolor='red',
             fill=False,
             lw=2
         ) )


    ax2 = fig.add_subplot(gs[2, 1])
    state=states[1]
    Neuron_state_arr, Location_state_arr,  _ = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    place_mapx= place_map(Neuron_state_arr, Location_state_arr, maze_mat, loc_mapping_for_plot)           
    ax2.matshow(place_mapx, cmap=heatmap_colormap)
    ax2.titlesize=20
    ax2.set(title=state)
    ax2.set_xticks([])
    ax2.set_yticks([])
    goal_locs = np.array(loc_mapping_for_plot[task[1]])[::-1]-0.5
    ax2.add_patch(
         patches.Rectangle(
             goal_locs,
             1,
             1,
             edgecolor='red',
             fill=False,
             lw=2
         ) )

    ax3 = fig.add_subplot(gs[2, 2])
    state=states[2]
    Neuron_state_arr, Location_state_arr,  _ = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    place_mapx= place_map(Neuron_state_arr, Location_state_arr, maze_mat, loc_mapping_for_plot)           
    ax3.matshow(place_mapx, cmap=heatmap_colormap)
    ax3.set(title=state)
    ax3.set_xticks([])
    ax3.set_yticks([])
    goal_locs = np.array(loc_mapping_for_plot[task[2]])[::-1]-0.5
    ax3.add_patch(
         patches.Rectangle(
             goal_locs,
             1,
             1,
             edgecolor='red',
             fill=False,
             lw=2
         ) )

    ax4 = fig.add_subplot(gs[2, 3])
    state=states[3]
    Neuron_state_arr, Location_state_arr,  _ = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    place_mapx= place_map(Neuron_state_arr, Location_state_arr, maze_mat, loc_mapping_for_plot)           
    ax4.matshow(place_mapx, cmap=heatmap_colormap)
    ax4.set(title=state)
    ax4.set_xticks([])
    ax4.set_yticks([])
    goal_locs = np.array(loc_mapping_for_plot[task[3]])[::-1]-0.5
    ax4.add_patch(
         patches.Rectangle(
             goal_locs,
             1,
             1,
             edgecolor='red',
             fill=False,
             lw=2
         ) )

    ax5 = fig.add_subplot(gs[3, 0])
    state=states[0]
    _, _, neuron_state_alltrials = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    spikes_raster, reward_raster = get_raster_arrays(neuron_state_alltrials)
    ax5.eventplot(spikes_raster, color=eventplot_line_colour_, linewidth=eventplot_linewidth_)
    ax5.eventplot(reward_raster, color='red', linewidth=2)
    ax5.set_yticks([0, data_neurons[neuron].shape[0]])
    
    _, _, ind_of_max_s, max_s = get_max_time(spikes_raster)

    ax5.set_yticks([0, data_neurons[neuron].shape[0]])
    ax5.set_xticks([0, ind_of_max_s])
    ax5.set_xticklabels([0, max_s])
    ax5.xaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax5.yaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax5.axvline(0, color='k')
    ax5.set_ylabel('Trial', fontsize=fontsize)
    ax5.set_xlabel('Time (seconds)', fontsize=fontsize)

    ax6 = fig.add_subplot(gs[3, 1])
    state=states[1]
    _, _, neuron_state_alltrials = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    spikes_raster, reward_raster = get_raster_arrays(neuron_state_alltrials)
    ax6.eventplot(spikes_raster, color=eventplot_line_colour_, linewidth=eventplot_linewidth_)
    ax6.eventplot(reward_raster, color='red', linewidth=2)
    ax6.set_yticks([0, data_neurons[neuron].shape[0]])
    _, _, ind_of_max_s, max_s = get_max_time(spikes_raster)

    ax6.set_yticks([0, data_neurons[neuron].shape[0]])
    ax6.set_xticks([0, ind_of_max_s])
    ax6.set_xticklabels([0, max_s])
    ax6.xaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax6.yaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax6.axvline(0, color='k')
    ax6.tick_params(labelsize=fontsize)
    ax6.set_ylabel('Trial', fontsize=fontsize)
    ax6.set_xlabel('Time (seconds)', fontsize=fontsize)

    ax7 = fig.add_subplot(gs[3, 2])
    state=states[2]
    _, _, neuron_state_alltrials = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    spikes_raster, reward_raster = get_raster_arrays(neuron_state_alltrials)
    ax7.eventplot(spikes_raster, color=eventplot_line_colour_, linewidth=eventplot_linewidth_)
    ax7.eventplot(reward_raster, color='red', linewidth=2)
    ax7.axvline(0, color='k')
    _, _, ind_of_max_s, max_s = get_max_time(spikes_raster)

    ax7.set_yticks([0, data_neurons[neuron].shape[0]])
    ax7.set_xticks([0, ind_of_max_s])
    ax7.set_xticklabels([0, max_s])
    
    ax7.tick_params(labelsize=fontsize)
    ax7.set_yticks([0, data_neurons[neuron].shape[0]])
    ax7.xaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax7.yaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax7.set_ylabel('Trial', fontsize=fontsize)
    ax7.set_xlabel('Time (seconds)', fontsize=fontsize)

    ax8 = fig.add_subplot(gs[3, 3])
    state=states[3]
    _, _, neuron_state_alltrials = get_data_for_state(Neuron_raw[neuron,:], Location_raw, state, Trial_times)
    spikes_raster, reward_raster = get_raster_arrays(neuron_state_alltrials)
    ax8.eventplot(spikes_raster, color=eventplot_line_colour_, linewidth=eventplot_linewidth_)
    ax8.eventplot(reward_raster, color='red', linewidth=2)
    ax8.axvline(0, color='k')
    ax8.tick_params(labelsize=fontsize)
    _, _, ind_of_max_s, max_s = get_max_time(spikes_raster)

    ax8.set_yticks([0, data_neurons[neuron].shape[0]])
    ax8.set_xticks([0, ind_of_max_s])
    ax8.set_xticklabels([0, max_s])
    
    ax8.set_yticks([0, data_neurons[neuron].shape[0]])
    ax8.set_ylabel('Trial', fontsize=fontsize)
    ax8.set_xlabel('Time (seconds)', fontsize=fontsize)
    ax8.xaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax8.yaxis.set_tick_params(width=tickwidth, length=ticklength)
    
    ax9 = fig.add_subplot(gs[:2, 2:], projection='polar')
    theta = np.radians(np.array(list(range(360))))
    err = std_err(data_neurons[neuron])
    y = smooth_circ(mean_neuron_session(data_neurons[neuron]))
    ax9.plot(theta, y, color='black')
    ax9.fill_between(theta, y-err, y+err)
    ax9.set_theta_zero_location("N")
    ax9.set_theta_direction(-1)
    ax9.tick_params(labelsize=fontsize)
    ax9.set_yticklabels([])
    ax9.set(title='Task tuning')
    lines, labels = plt.thetagrids((0, 90, 180, 270), ("A", "B", "C", "D"))

    ax10 = fig.add_subplot(gs[4:, :])
    ax10.matshow(data_neurons[neuron], origin='lower', cmap=heatmap_colormap)
    ax10.set_ylabel('Trial', fontsize=fontsize)
    ax10.set_xlabel('Task angle', fontsize=fontsize)
    ax10.set_yticks([0, data_neurons[neuron].shape[0]])
    ax10.set_xticks([0, 90, 180, 270, 360])
    ax10.xaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax10.yaxis.set_tick_params(width=tickwidth, length=ticklength)
    ax10.tick_params(top=False, labelbottom=True, labeltop=False, labelsize=fontsize)
    for angle in np.arange(4)*90:
        ax10.axvline(angle,color='red',ls='dashed')
    fig.suptitle(f'neuron {neuron}, {mouse_recday}, session {session+1}', fontsize=15) 
    
    if save==True:
        cell_pdf.savefig(fig)
    
    plt.close() 

def rdm(mat):
    rdm = np.zeros((mat.shape[1],mat.shape[1]))
    for i in range(mat.shape[1]):
        for j in range(mat.shape[1]):
            rdm[i,j] = pearsonr(mat[:,i], mat[:,j])[0]
    return rdm

def remove_consecutive_repeats(lst):
    return [lst[i] for i in range(len(lst)) if i == 0 or lst[i] != lst[i - 1]]

def get_route(locs):
    locs_nodes = [i for i in locs if i<=9]
    locs_nodes_norepeats = remove_consecutive_repeats(locs_nodes)
    return locs_nodes_norepeats 

def get_specific_route_state_activity(route, data_neurons, state):
    state_ind_dic = {
        'A': [0, 90],
        'B': [90, 180],
        'C': [180, 270],
        'D': [270, 360]
    }
    
    data_neurons_state = data_neurons[:, route, :]
    mean_data_neurons_state = np.zeros((data_neurons_state.shape[0], data_neurons_state.shape[2]))

    for neuron in range(mean_data_neurons_state.shape[0]):
        mean_data_neurons_state[neuron, :] = smooth_circ(np.mean(data_neurons_state[neuron], axis=0))
    
    return mean_data_neurons_state[:,state_ind_dic[state][0]:state_ind_dic[state][1]]

def normalise_rows(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

def interpolate_tuples(tuple1, tuple2, n):
    if len(tuple1) != len(tuple2):
        raise ValueError("Tuples must have the same length")

    result = []
    for i in range(len(tuple1)):
        start_value = tuple1[i]
        end_value = tuple2[i]
        step_size = (end_value - start_value) / (n + 1)
        interpolated_values = [start_value + step_size * j for j in range(1, n + 1)]
        interpolated_values = [start_value] + interpolated_values + [end_value]
        result.append(interpolated_values)

    return result

def calculate_euclidean_distances(data):
    # Calculate the squared differences between consecutive timepoints
    squared_differences = np.diff(data, axis=0) ** 2
    
    # Sum the squared differences along the columns (x and y coordinates)
    sum_squared_differences = np.sum(squared_differences, axis=1)
    
    # Take the square root to obtain the Euclidean distances
    euclidean_distances = np.sqrt(sum_squared_differences)
    
    return euclidean_distances

def jump_outlier_detect(data, threshold=5):
    z_scores = np.abs(stats.zscore(data, axis=0))
    outlier_indices = [True if j > threshold else False for i, j in enumerate(z_scores)]
    return outlier_indices


def normalise_xy_for_plot(coords, bounds = [0,4]):
    x = coords[:,0]
    y = coords[:,1]
    
    x = x-min(x) + bounds[0]
    y = y-min(y) + bounds[0]
    
    x = (x/max(x)) * bounds[1]
    y = (y/max(y)) * bounds[1]
    
    x = (x*1.28) - 0.45
    y= (y*1.28) - 0.45
    
    return(x, y)




def rotate_coordinates(coordinates, angle_deg):
    radians = np.radians(angle_deg)
    rotation_matrix = np.array([[np.cos(radians), -np.sin(radians)],
                                [np.sin(radians), np.cos(radians)]])
    
    rotated_coordinates = np.dot(coordinates, rotation_matrix)
    return rotated_coordinates


def get_mindistance_mat():
    x=(0,1,2)
    Task_grid=np.asarray(list(product(x, x)))

    mapping_pyth={2:2,5:3,8:4}

    distance_mat_raw=distance_matrix(Task_grid, Task_grid)
    len_matrix=len(distance_mat_raw)
    distance_mat=np.zeros((len_matrix,len_matrix))
    for ii in range(len_matrix):
        for jj in range(len_matrix):
            if (distance_mat_raw[ii,jj]).is_integer()==False:
                hyp=int((distance_mat_raw[ii,jj])**2)
                distance_mat[ii,jj]=mapping_pyth[hyp]
            else:
                distance_mat[ii,jj]=distance_mat_raw[ii,jj]
    mindistance_mat=distance_mat.astype(int)
    return mindistance_mat


def string_route_to_int_list(string_route):
    route = json.loads(string_route)
    list_int_route = [int(item) for item in route]
    return list_int_route


def get_route_dic(locations, state):
     
    state_ind_dic = {
        'A': [0, 90],
        'B': [90, 180],
        'C': [180, 270],
        'D': [270, 360]
    }
    
    state_inds = state_ind_dic[state]
    
    routes_dic = {}
    for i, _ in enumerate(locations):
        route = get_route(locations[i][state_inds[0]:state_inds[1]])
        if str(route) not in routes_dic:
            routes_dic[str(route)] = [i]
        else:
            routes_dic[str(route)].append(i)
    return routes_dic


def most_freq_correct_route(routes_dic, start, goal):
    mindistance_mat = get_mindistance_mat()
    route_list = []
    num_trials_list = []
    for route in routes_dic.keys():
        route_int = string_route_to_int_list(route)
        if route_int[0]==start and route_int[-1]==goal:
            shortest_route_length=mindistance_mat[start-1, goal-1]
            
            if len(route_int)==int(shortest_route_length)+1:
                num_trials_list.append(len(routes_dic[route]))
                route_list.append(route)
    trial_counts = np.array(num_trials_list)
    route_ind = np.argmax(trial_counts)
    
    route_choice = route_list[route_ind]
    route_trials = routes_dic[route_choice]
    
    return [route_choice, route_trials]

def start_goal_pairs_for_task(task):
    start_goal_pairs = []
    for i in range(len(task)):
        start_goal_pairs.append([task[i], task[(i+1)%len(task)]])
    return start_goal_pairs

def clean_and_smoooth_coords(coords_, threshold=5):
    coords_ = ndimage.uniform_filter1d(coords_, 10, axis=0)
    pairwise_dists = calculate_euclidean_distances(coords_)
    outliers = jump_outlier_detect(pairwise_dists, threshold)

    coords_=coords_[1:]
    coords_[outliers,:] = np.nan
    
    return coords_


def plot_2d_raster_trajectories(neurons_chosen_from_plot, 
                                Neuron_raw, 
                                trials_times_trajectory_dic,
                                Trial_times,
                                coords_clean, 
                                colors_state, 
                                trial_num,
                                common_shortest=True, 
                                save_name=None):
    
    colormap = plt.cm.gist_rainbow
    n = len(neurons_chosen_from_plot)
    color_values = np.linspace(0, 1, n)
    colors_neurons = [colormap(val) for val in color_values]

    state_ints = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3
    }
    neurons_subset_array = Neuron_raw[neurons_chosen_from_plot, :]
    neurons_subset_array = (neurons_subset_array > 0).astype(int)

    fig, ax = plt.subplots(figsize=(10,10))
    plt.xlim(0, 1100)
    plt.ylim(0, 1100)

    for i, s in enumerate(states_for_traj_plot):
        
        if common_shortest == True:
            trial_times = (trials_times_trajectory_dic[s]/25).astype(int)
        else:
            trial_times = (Trial_times[:,state_ints[s]:state_ints[s]+2]/25).astype(int)
    
        linewidth = 0.1
        if trial_num!='all':
            trial_times = [trial_times[trial_num,:]]
            linewidth = 1
        
        for t in trial_times:
            
            ax.plot(coords_clean[t[0]:t[1],0], 
                    coords_clean[t[0]:t[1],1],
                    color=colors_state[i], linewidth=linewidth)
            
            if trial_num!='all':
                ax.scatter(coords_clean[t[0]:t[1],0][0], coords_clean[t[0]:t[1],1][0], 
                          color = 'gray', s=100, zorder=4, marker='s')
                ax.scatter(coords_clean[t[0]:t[1],0][-1], coords_clean[t[0]:t[1],1][-1], 
                          color = 'gray', s=300, zorder=4, marker='*')
    
    for n, label in enumerate(neurons_chosen_from_plot):
        neuron_x_combined = []
        neuron_y_combined = []
        for i, s in enumerate(states_for_traj_plot):
            
            if common_shortest == True:
                trial_times = (trials_times_trajectory_dic[s]/25).astype(int)
            else:
                trial_times = (Trial_times[:,state_ints[s]:state_ints[s]+2]/25).astype(int)
    
            
            
            if trial_num!='all':
                trial_times = [trial_times[trial_num,:]]
            
            for t in trial_times:
                
                x_ = coords_clean[t[0]:t[1],0]
                y_ = coords_clean[t[0]:t[1],1]

                spikes_trial_state = neurons_subset_array[n,t[0]:t[1]]

                neuron_x_combined.extend(x_[spikes_trial_state.astype(bool)])

                neuron_y_combined.extend(y_[spikes_trial_state.astype(bool)])

        if trial_num=='all':
            scatter_size = 5
        else:
            scatter_size=50
        ax.scatter(neuron_x_combined, neuron_y_combined, zorder=10-n, color = colors_neurons[n], s=scatter_size,
                  label=label)
        
    ax.legend()
    if save_name:
        
        plt.savefig(save_name)
        
def peak_comparison_plot(neuron_peaks_state_dic,
                        states_for_plot):
    
    fig, ax=plt.subplots(figsize=(10,10))
    ax.scatter(neuron_peaks_state_dic[states_for_plot[0]], 
               neuron_peaks_state_dic[states_for_plot[1]], 
               c=neuron_peaks_state_dic[states_for_plot[0]])

    for i, txt in enumerate(neuron_peaks_state_dic[states_for_plot[0]]):
        ax.annotate(str(i), (neuron_peaks_state_dic[states_for_plot[0]][i],
                            neuron_peaks_state_dic[states_for_plot[1]][i]))
    ax.plot(range(90), range(90))
    plt.xlabel('State x peak')
    plt.ylabel('State y peak')
    
def get_whole_run(trial, Trial_times, cleaned_coords_):
    Trial_times = (Trial_times/25).astype(int)
    single_trial_times_ = Trial_times[trial, :]

    start = single_trial_times_[0]
    end = single_trial_times_[-1]
    coords_for_run = cleaned_coords_[start:end,:] 
    
    coords_for_rew = []
    
    for i,_ in enumerate(single_trial_times_[:-1]):

        coords_for_rew.append(cleaned_coords_[single_trial_times_[i]])
    
    return coords_for_run, coords_for_rew


def get_whole_trial_raster(neuron, 
                           Neuron_raw, 
                           Trial_times,
                           trial,
                           coords_clean):
    
    Trial_times = Trial_times = (Trial_times/25).astype(int)
    single_trial_times_ = Trial_times[trial, :]
    
    neurons_subset_array = Neuron_raw[neuron, :]
    neurons_subset_array = (neurons_subset_array > 0).astype(int)
    
    spikes_x = []
    spikes_y = []
    
    x_ = coords_clean[single_trial_times_[0]:single_trial_times_[-1],0]
    y_ = coords_clean[single_trial_times_[0]:single_trial_times_[-1],1]

    spikes_trial_state = neurons_subset_array[single_trial_times_[0]:single_trial_times_[-1]].astype(bool)

    spikes_x = x_[spikes_trial_state]

    spikes_y = y_[spikes_trial_state]

    return(spikes_x, spikes_y)

    
    
if __name__ == "__main__":
    mindistance_mat = get_mindistance_mat()
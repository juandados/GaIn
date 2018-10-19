from __future__ import print_function
import csv
import os
import glob
from tempfile import NamedTemporaryFile
from io import StringIO
import pickle
from random import shuffle

import numpy as np
import scipy
import constants as const
from collections import defaultdict

from transformation import transform, all_transformed, create_timestamps_for_future


def get_data_from_file(path_to_file, to_normalize=True, normalizing_function=None, float_type=float):
    """Read HuGaDB data from file and get numpy matrix
        Parameters:
            path_to_file: string
                path to HuGaDB file.
            is_normalize: boolean 
                if True programm will use norm_function for data transformation
                By deafult :Normalization means deviding data by int16
                (for gyro and accs) and shifting and deviding 
                by 128 for EMG
            norm_function: function 
                input: numpy-array(X, 39),  return numpy-array(X, 39)
        Return: 2d-array 
            Data in numpy format 
    """

    data = np.genfromtxt(path_to_file, delimiter='\t', skip_header=4).astype(float_type)

    # find outliers
    if np.where(data > 32768.0, True, False).any():
        print("remove {0} rows with file ({1}) because of outliers".format(
                np.where(data >
                         32768.0)[0], path_to_file))
        np.delete(data, np.where(data > 32768.0)[0], 0)

    #find NaN data
    if np.isnan(data).any():
        print("In {0} file in {2} positions".format(path_to_file, 
                                                    np.where(np.isnan(data))))

    #normalize data
    if to_normalize:
        if not normalizing_function:
            # for accs and gyros devide by int16
            for i in range(data.shape[1]-3):
                data[:, i] /= 32768.0

            # for EMG devide by byte/2 and shift it by byte/2 to normalize it at zero
            data[:, -2] = (data[:, -2] - 127.0)/128.0
            data[:, -3] = (data[:, -3] - 127.0)/128.0
        else: 
            data = normalizing_function(data)
    return data

def get_data_from_files(files, to_normalize=True, normalizing_function=None):
    """
    Reading data from files and creating one big matrix.
    Parametrs:
        files: list 
            List of the path to data file
        is_normalize: boolean 
                Normalization means deviding data by int16
                (for gyro and accs) and shifting and deviding 
                by 128 for EMG
    Return:
        Data from all files in one matrix
    """
    data = get_data_from_file(files[0], to_normalize, normalizing_function)
    for i in range(1, len(files)):
        data = np.row_stack((data, get_data_from_file(files[i], to_normalize, normalizing_function) ) )
    return data


def get_activity_set_from_file(filename):
    """
    Getting set of activities from file.
    Parametrs:
        filename - name of HuGaDB file with data.
    Retrun:
        set of string-states
    """
    sets_of_act = set()
    with open(filename) as fl:
        for line in fl:
            if "#Activity\t" in line:
                for a in line.split('\t')[1].split(' '):
                    sets_of_act.add(a)
                break
            if not "#" in line:
                raise RuntimeError("There is no  sets in file header")
    if '\n' in sets_of_act:
        sets_of_act.remove('\n')
    if '\r\n' in sets_of_act:
        sets_of_act.remove('\r\n')
    return sets_of_act


def get_activity_set_from_files(files):
    """
    Getting set of activities from all files.
    Parametrs:
        filename - array-like 
            list of names of HuGaDB file with data.
    Retrun:
        set of string-states from all files
    """
    states = set();
    for f in files:
        states |= get_activity_set_from_file(f)
    return states

    
def get_files(path="..\\..\\Data\\", partisipant_list=None, naming_list=["various"], 
                activity_list=["walking", "running", "going_up", "going_down", 
                                "sitting", "sitting_down", "standing_up", "standing", 
                                "up_by_elevator", "down_by_elevator",
                                ]):
    """
    Get list of HuGaDB files based on constitution.
    Parametrs:
        path: string
            path to folder with HuGaDB files
        partisipants_list: numeric list
            list of partisipants, data from which we want to get.
            If None use 18 partisipants
        naming_list: string list
            list of words which must be in file name.
            If None use all HuGaDB names.
        activities_list: string list
            list of activities that we want to get. Fot example:
            activities_list = ["walking", "sitting"], file only with 
            "walking" activity will be included, but file with
            "sitting_down" and "sitting" won't be included.
            If None use all HuGaDB activities
    Return: string list
        List with path to files. This path contains path to folder
    """
    if partisipant_list is None:
        partisipant_list = [i for i in range(1, 19)]


    if naming_list is None or naming_list == []:
        naming_list = ["walking", "running", "going_up", "going_down", "sitting",
                        "sitting_down", "standing_up", "standing", "bicycling", 
                        "up_by_elevator", "down_by_elevator", "sitting_in_car", "various"
                        ]
    if activity_list is None:
        activity_list = ["walking", "running", "going_up", "going_down", "sitting",
                            "sitting_down", "standing_up", "standing", "bicycling", 
                            "up_by_elevator", "down_by_elevator", "sitting_in_car" 
                            ]

    activities_set = set(activity_list)
    tmp_files = []
    #find files for partisipants"
    for partis in partisipant_list:
        tmp_files += glob.glob(path + "*_{0:02}_*".format(partis))
    files = []
    for file_act_name in naming_list:
        for file_ in tmp_files:
            #we want be sure that file contains only requirement activities
            #summing is requred because sitting may be find in sitting_down
            if (file_act_name + "_0" in file_ or file_act_name + "_1" in file_) and \
                    get_activity_set_from_file(file_).issubset(activities_set):
                files.append(file_)
    return files


def get_less_files(path="..\\..\\Data\\", partisipant_list=None, naming_list=["various"], 
                activity_list=["walking", "running", "going_up", "going_down", 
                                "sitting", "sitting_down", "standing_up", "standing", 
                                "up_by_elevator", "down_by_elevator",
                                ]):

    if partisipant_list is None:
        partisipant_list = [i for i in range(1, 19)]


    if naming_list is None or naming_list == []:
        naming_list = ["walking", "running", "going_up", "going_down", "sitting",
                        "sitting_down", "standing_up", "standing", "bicycling", 
                        "up_by_elevator", "down_by_elevator", "sitting_in_car", "various"
                        ]
    if activity_list is None:
        activity_list = ["walking", "running", "going_up", "going_down", "sitting",
                            "sitting_down", "standing_up", "standing", "bicycling", 
                            "up_by_elevator", "down_by_elevator", "sitting_in_car" 
                            ]

    activities_set = set(activity_list)
    tmp_files = []
    #find files for partisipants"
    for id in partisipant_list:
        tmp_files += glob.glob(path + "*_{0:02}_*".format(id))
    
    files = []
    for file_act_name in naming_list:
        for file_ in tmp_files:
            #we want be sure that file contains only requirement activities
            #summing is requred because sitting may be find in sitting_down
            if (file_act_name + "_0" in file_ or file_act_name + "_1" in file_) and \
                    get_activity_set_from_file(file_).issubset(activities_set):
                files.append(file_)
                
    print(files[0])            
    less_files = []
    
    for id in partisipant_list:
        template = "_{0:02}_".format(id)
        test_set = set()
        for f in files:
            if template in f:
                acts = get_activity_set_from_file(f)
                if (test_set | acts) != test_set:
                    less_files.append(f)
                    test_set = test_set | acts
    return less_files


def create_and_save_data_in_list(saved_file_name = "data_in_list.pkl", 
                                 path="..\\..\\Data\\", 
                                 participant_list=None, naming_list=["various"], 
                                 activity_list=["walking", "running", "going_up", "going_down", 
                                                 "sitting", "sitting_down", "standing_up", 
                                                 "standing", "up_by_elevator", "down_by_elevator",
                                                 ], 
                                 to_normalize=True, normalizing_function=None, remove_lifting=True, 
                                 get_files=get_files, float_type=float
                                 ):
    """
    Read data from text files and save it in file.
    Data saved in lists: (data_list, label_list, activity_list)
    Parameters:
        saved_file_name: string
            name of the file with data
        path: string
            path to the directory with HuGaDB textfiles
        partisipants_list: numeric list
            list of partisipants, data from which we want to get.
            If None use 18 partisipants
        naming_list: string list
            list of words which must be in file name.
            If None use all HuGaDB names.
        activity_list: string list
            list of activities that we want to get. Fot example:
            activity_list = ["walking", "sitting"], file only with 
            "walking" activity will be included, but file with
            "sitting_down" and "sitting" won't be included.
            If None use all HuGaDB activities
        to_normalize: boolean 
            normalization means deviding data by int16
            (for gyro and accs) and shifting and deviding 
            by 128 for EMG
        remove_lifting: boolean
            set down_by_elevator and up_by_elevator labels as stading
    """
    if participant_list is None:
        participant_list = [i for i in range(1, 19)]
   
    # list of paritsipants data/labels
    input_data_lists = []
    output_data_lists = []
    label_lists = []

    # activity states
    states = set()
    for id in participant_list:
        print("Saving data for {0} participant".format(id))
        files = get_files(path, [id], naming_list, activity_list) 
        states |= get_activity_set_from_files(files)
        print(files)
        # concrete partisipant data/labels
        partisipant_input_data_list = []
        partisipant_output_data_list = []
        partisipant_label_list = []
        for f in files:
            in_tmp_d, out_tmp_d, tmp_l = separate_data_and_activities(get_data_from_file(f, to_normalize, normalizing_function,                                                               float_type=float_type))
            if remove_lifting:
                tmp_l[tmp_l == const.UP_BY_ELEVATOR_ID] = const.STANDING_ID
                tmp_l[tmp_l == const.DOWN_BY_ELEVATOR_ID] = const.STANDING_ID
            partisipant_input_data_list.append(in_tmp_d)
            partisipant_output_data_list.append(out_tmp_d)
            partisipant_label_list.append(tmp_l)

        #save data from partisipant
        input_data_lists.append(partisipant_input_data_list)
        output_data_lists.append(partisipant_output_data_list)
        label_lists.append(partisipant_label_list)

    # I do this to have same order for activities
    list_of_states = []
    for i in range(len(const.ALL_ACTIVITIES)):
        if const.ALL_ACTIVITIES[i] in states:
            list_of_states.append(const.ALL_ACTIVITIES[i])

    if remove_lifting:
        if const.UP_BY_ELEVATOR_STRING in list_of_states:
            list_of_states.remove(const.UP_BY_ELEVATOR_STRING)
        if const.DOWN_BY_ELEVATOR_STRING in list_of_states:
            list_of_states.remove(const.DOWN_BY_ELEVATOR_STRING)

    with open(saved_file_name, 'wb') as f:
        tup = (input_data_lists, output_data_lists, label_lists, list_of_states)
        #second protocol is used for using with Python 2.7
        pickle.dump(tup, f, protocol=2) 
        
        
def separate_data_and_activities(data):
    """
    Get input, output data and activity column of the matrix and 
    remove this (activity) column from matrix
    Parameters:
        data - 2d array
            HuGaDB raw data
    Return:
        (data from thigh include EMG, data from rest leg, last colomn of raw data (acitivities))
    """
    data = np.array(data)                           # tranfrom list to array
    states = data[:, len(data[0])-1]                # get activity column
    data = scipy.delete(data, len(data[0])-1, 1)    # delete activity from matrix
    return (data[:, const.input_features_indexes], data[:, const.output_features_indexes], states)


def create_data_and_labels_from_list(input_data_list, output_data_list, label_list):
    """
    Concatinate data matrix in list in one big matrix. 
    Concatinate labels in list into one list
    Parameters:
        input_data_list: list of numpy array
            List of HuGaDB thigh data in numpy array
        output_data_list: list of numpy array
            List of HuGaDB data (exclude data from thigh) in numpy array
        label_list: list of numpy array
            List of activities labels in numpy array
    Return:
        (concatinated data, concatinated labels)
    """
    input_data = input_data_list[0]
    output_data = output_data_list[0]
    labels = label_list[0]
    for i in range(1, len(input_data_list)):
        input_data   = np.concatenate((input_data,  input_data_list[i]))
        output_data   = np.concatenate((output_data,  output_data_list[i]))
        labels = np.concatenate((labels, label_list[i]))
    return input_data, output_data, labels

def test_same_length(input_data, output_data, labels, text, addition_text=""):
    """
    Testing are labels and data have save lenght.
    If they have different legth excaption raised
    Parameters:
        input data: numpy-array
            HuGaDB data from thigh
        output data: numpy-array
            HuGaDB data from rest leg
        labels: list
            HuGaDB labels
        text: string
            string equal to 'Learning' or 'Testing'
        addition_text: string
            index of element in test data in list
    """
    if len(input_data) != len(labels):
        raise RuntimeError("{0} input data len ({1}) != \
                            {0} labels len ({2}). {3}".format(text, len(input_data), 
                                                              len(labels), addition_text
                                                              )
                           ) 
    if len(output_data) != len(labels):
        raise RuntimeError("{0} output data len ({1}) != \
                            {0} labels len ({2}). {3}".format(text, len(output_data), 
                                                              len(labels), addition_text
                                                              )
                           )

def create_for_one_participant(X_lists, Y_lists, label_lists, list_of_states): 
    X_learn_list = []
    Y_learn_list = []
    learning_labels_list = []

    X_test_list = []
    Y_test_list = [] 
    testing_labels_list = []

    test_set = set()
    for i in range(len(X_lists)):
        # if we already have this in 
        if (test_set | set(label_lists[i])) == test_set:
            X_learn_list.append(X_lists[i])
            Y_learn_list.append(Y_lists[i])
            learning_labels_list.append(label_lists[i])
        else:
            test_set = test_set | set(label_lists[i]) 
            X_test_list.append(X_lists[i])
            Y_test_list.append(Y_lists[i])
            testing_labels_list.append(label_lists[i])

    X_learn, Y_learn, learning_labels = create_data_and_labels_from_list(X_learn_list, Y_learn_list, learning_labels_list)    

    X_test, Y_test, testing_labels = create_data_and_labels_from_list(X_test_list, Y_test_list, testing_labels_list) 
    
    learning_labels = np.array(learning_labels)
    testing_labels = np.array(testing_labels)
    
    return X_learn, Y_learn, learning_labels, X_test, Y_test, testing_labels

def create_for_one_participant_from_list(id, X_lists, Y_lists, label_lists, walk_list_ids): 
    
    test_X, test_Y, test_labels = create_data_and_labels_from_list(X_lists[id-1], Y_lists[id-1], label_lists[id-1])

    tmp_X = []
    tmp_Y = []
    tmp_labels = []
    for i in walk_list_ids:
        if i != id:
            tmp_X.extend(X_lists[i-1])
            tmp_Y.extend(Y_lists[i-1])
            tmp_labels.extend(label_lists[i-1])

            learning_X, learning_Y, learning_labels = create_data_and_labels_from_list(tmp_X, tmp_Y, tmp_labels)
    
    return test_X, test_Y, test_labels, learning_X, learning_Y, learning_labels  
        
        
def create_dict_from_several_id(X_id_lists, Y_id_lists, label_id_lists, walk_list_ids):
    test_X_dict = {}
    test_Y_dict = {}
    test_labels_dict = {}

    learning_X_dict = {}
    learning_Y_dict = {}
    learning_labels_dict = {}
    
    for id in walk_list_ids:
        if len(X_id_lists[id-1]) != 0:
            test_X_dict[id], test_Y_dict[id], test_labels_dict[id] = \
            create_data_and_labels_from_list(X_id_lists[id - 1], Y_id_lists[id - 1], label_id_lists[id - 1])
            tmp_X = []
            tmp_Y = []
            tmp_labels = []
            for i in range(1, 19):
                if i != id:
                    tmp_X.extend(X_id_lists[i-1])
                    tmp_Y.extend(Y_id_lists[i-1])
                    tmp_labels.extend(label_id_lists[i-1])
                    print('{0}, '.format(i), end='')
                    
            learning_X_dict[id], learning_Y_dict[id], learning_labels_dict[id] = \
                 create_data_and_labels_from_list(tmp_X, tmp_Y, tmp_labels)
            print()
    return test_X_dict, test_Y_dict, test_labels_dict, learning_X_dict, learning_Y_dict, learning_labels_dict    


def load_acc_data_for_one(data_file="various_7_01.pkl"):
    with open(data_file, 'rb') as f:
        X_lists, Y_lists, label_lists, list_of_states = pickle.load(f)

    X_lists, Y_lists, label_lists, list_of_states = X_lists[0], Y_lists[0], label_lists[0], list_of_states[0] 

    X_learn, Y_learn, learning_labels, X_test, Y_test, testing_labels = \
        create_for_one_participant(X_lists, Y_lists, label_lists, list_of_states)

    X_learn = X_learn[:, [0, 1, 2,  6, 7, 8,  12, 13]]
    Y_learn = Y_learn[:, [0, 1, 2,  6, 7, 8]]
    X_test = X_test[:, [0, 1, 2,  6, 7, 8,  12, 13]]
    Y_test = Y_test[:, [0, 1, 2,  6, 7, 8]]
    
    return X_learn, Y_learn, learning_labels, X_test, Y_test, testing_labels 


def create_transformed_for_data_for_all(X_id_lists, Y_id_lists, label_id_lists, 
                                        X_features, Y_features, walk_list_ids,
                                        data_transformation, 
                                        template="data\\RNN_all_acc_fourier_10_{0}_10.pkl", 
                                        **kwarg
                                       ):
    transformed_X = defaultdict(list)
    transformed_Y = defaultdict(list)
    transformed_labels = defaultdict(list)
    
    for id in walk_list_ids: # for each part
        print("preparing data for id: ", id)
        id = id - 1
        for i in range(len(X_id_lists[id])): # for each file 
            transformed_X[id].append(None)
            transformed_Y[id].append(None)
            transformed_labels[id].append(None)
            
            transformed_X[id][i], transformed_Y[id][i],  transformed_labels[id][i] = \
                data_transformation(X=X_id_lists[id][i][:, X_features], 
                                    Y=Y_id_lists[id][i][:, Y_features], 
                                    labels=label_id_lists[id][i], 
                                    **kwarg
                                   )   

    for id in walk_list_ids:   
        print("creating data for id ", id)
        test_X, test_Y, test_labels, learning_X, learning_Y, learning_labels = \
            create_for_one_participant_from_list(id, transformed_X, transformed_Y, transformed_labels, walk_list_ids) 

        with open(template.format(id), 'wb') as f:
            tup = test_X, test_Y, test_labels, learning_X, learning_Y, learning_labels
            pickle.dump(tup, f) 

def load_data_for_all(id, template="data\\std_emg_reccurent_all_{0}_10.pkl"):
     with open(template.format(id), 'rb') as f:
        return pickle.load(f)
    
    
def create_data(walk_list_ids, transformation_list, template, N_std, N_RNN, 
                to_dict=False, to_RNN=True, datafile="SUSDSS_all_01.pkl", 
                X_features = [12, 13], features_extraction=None,
                labels_func=None, test_to_list=False, is_verbose=True):
    
    with open(datafile, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        X_id_lists, Y_id_lists, label_id_lists, list_id_of_states = u.load()

    if is_verbose: print(walk_list_ids)
    for id in walk_list_ids: 
        id = id - 1
        for i in range(len(X_id_lists[id])): 
            if features_extraction is not None:
                X_id_lists[id][i] = features_extraction(X_id_lists[id][i])
            else:
                X_id_lists[id][i] = X_id_lists[id][i][:, X_features]
            
            Y_id_lists[id][i] = np.copy(X_id_lists[id][i][N_std:])

            if labels_func is not None:
                label_id_lists[id][i] = labels_func(label_id_lists[id][i][N_std:])

            if transformation_list is not None and len(transformation_list) != 0:
                X_id_lists[id][i] = all_transformed(X_id_lists[id][i], transformations=transformation_list, N=N_std) 
                Y_id_lists[id][i] = np.hstack((Y_id_lists[id][i], X_id_lists[id][i]))
                
            if to_RNN:
                X_id_lists[id][i], Y_id_lists[id][i], label_id_lists[id][i] = \
                    create_timestamps_for_future(X_id_lists[id][i],
                                                 Y_id_lists[id][i], 
                                                 n=N_RNN, 
                                                 labels=label_id_lists[id][i]
                                                )
    data_dict = {}
    if test_to_list:
        if is_verbose: print("****************************\ntest in list!\n****************************\n")
    for id in walk_list_ids:
        test_X, test_Y, test_labels, learning_X, learning_Y, learning_labels = \
            create_for_one_participant_from_list(id, X_id_lists, Y_id_lists, label_id_lists, walk_list_ids) 
        if is_verbose: print(id, test_X.shape, test_Y.shape, test_labels.shape, 
                             learning_X.shape, test_labels.shape, learning_labels.shape, end='\r')
        
        if test_to_list:
            tup = X_id_lists[id-1], Y_id_lists[id-1], label_id_lists[id-1], learning_X, test_labels, learning_labels   
    
        else:
            tup = test_X, test_Y, test_labels, learning_X, test_labels, learning_labels
                
        if to_dict:   
            data_dict[id] = tup
        else:
            with open(template.format(id), 'wb') as f:
                pickle.dump(tup, f) 
    
    if to_dict:
        return data_dict
from utility import merge_data
import json
import os
import shutil

def create_dictionary(path_daughter: str):
    keys_init = ['cryostat', "run", 'measurement', "channel", "number_file", 'window_point', "mean_pulse_min", 'mean_pulse_max',
            'mean_pulse_sample', 'baseline_sample', 'sigma_baseline', 'inc_sigma_baseline', 'threshold',
            'max_search_jitter', 'delay', 'path_heat_channel', 'filename']
    try :
        dictionary = load_dict(path_daughter+'dictionary.json')
        keys = [key for key in keys_init if key not in dictionary.keys()]
    except FileNotFoundError:
        dictionary = {'path': path_daughter}
        keys = keys_init
    print(dictionary)
    bool_merge = 'path_heat_channel' not in dictionary.keys()
    for key in keys:
        value = input(key + ': ')
        if key == 'sigma_baseline':
            print(5*value)
        try:
            dictionary[key] = float(value)
        except ValueError:
            dictionary[key] = value
    if bool_merge and dictionary["number_file"] > 1 :
        merge_data.merge_files(dictionary['path'], dictionary['path_heat_channel'], dictionary['filename'])
    with open(path_daughter + 'dictionary.json', 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)


def load_dict(dict_path: str) -> dict:
    with open(dict_path, 'r') as json_file:
        dictionary = json.load(json_file)
    return dictionary


def update_dict(dict_path: str, new_dict: dict):
    with open(dict_path, 'r') as json_file:
        dictionary = json.load(json_file)
    dictionary.update(new_dict)
    with open(dict_path, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)


def get_values(dictionary: dict, key_list: list) -> list:
    return [dictionary[key] for key in key_list]


def get_keys_starting_with(dictionary, sequence):
    return [key for key in dictionary.keys() if key.startswith(sequence)]


def list_files_in_directories(directory_path):
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for dirnames in dirnames:
            path = os.path.join(dirpath, dirnames)
            print(path)
            A = path.split("/")
            A[-2] = "meas3"
            B = '/'.join(A)+'/'
            list_files = os.listdir(path)
            for file in list_files:
                shutil.move(os.path.join(path, file), B)


if __name__ == "__main__":
    path_daughter = '/Users/mp274748/Documents/data_arg/RUN97/measu3'
    list_files_in_directories(path_daughter)


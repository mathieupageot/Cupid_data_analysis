import os
import json
import numpy as np
import dictionary_handler


def decode_binary(file_path):
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    return np.frombuffer(binary_data, dtype=np.uint32)


def merge_binary(path, filename_start):
    list_file = os.listdir(path)
    with open(path+filename_start+'.bin' , "wb") as merged_file:
        for filename in list_file:
            if (filename_start in filename) and ('.bin' == filename[-4:]):
                print(path+filename)
                with open(path+filename, 'rb') as file:
                    merged_file.write(file.read())


def merge_files(path_daughter, path_mother, filename_new, extension='.ntp',type='heat'):
    with open(path_daughter + filename_new, "w") as merged_file :
        list_file = np.sort(os.listdir(path_mother))
        dictionary = {}

        i = 0
        for filename in list_file:
            if filename[-len(extension):] == extension and len(filename)==len('000015_20230721T115252_003_000.bin.ntp'):
                with open(path_mother+filename, "r") as file:
                    merged_file.write(file.read())
                with open(path_mother + filename, "r") as file:
                    line_count = sum(1 for _ in file)
                name = "len_{}_{}".format(type,i)
                dictionary.update({name: line_count})
                print(filename)
                i+=1
        dictionary_handler.update_dict(path_daughter+'dictionary.json', dictionary)

def lim_bin_files(path,type,i):
    return os.path.getsize(path_daughter+'000015_20230609T235012_001_000.bin')//4







if __name__ == '__main__':
    import numpy as np
    path_daughter = '/Users/mp274748/Documents/data_arg/RUN97/meas4/channel8/'
    merge_files(path_daughter, path_daughter,'000015_20230721T115252_008.bin.ntp',extension=".bin.ntp",type='heat')
    #merge_binary(path_daughter,"000015_20230609T235012")

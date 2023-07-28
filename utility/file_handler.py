import numpy as np



def save_array_with_label(labels,array,path):
    with open(path, 'w') as file:
        # Write strings on the first line
        strings_line = "\t".join(labels)
        file.write(strings_line + "\n")
        # Write numbers on the rest of the lines
        np.savetxt(file, array, fmt='%.4e', delimiter='\t')

def load_array_with_label(path):
    return(np.genfromtxt(path, delimiter='\t')[1:])
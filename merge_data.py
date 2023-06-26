path =  '/Users/mp274748/Documents/data_arg/RUN96/Measurement6/LMO18/'
pathbis = '/Users/mp274748/Documents/data_arg/RUN96/Measurement6/chan1bis/'
name1 = '000015_20230609T235012_001_00'
name2 = '.bin.ntp'

merged_file = open(path + "000015_20230609T235012_001_bis.ntp", "w")

for i in range(4):
    with open(pathbis+name1+str(i)+name2, "r") as file:
        merged_file.write(file.read())

merged_file.close()
PATH_REF = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level3/reference/'
PATH_TEST = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level3/test/'
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

DEMENSION = 3

def load_all_data(path):
    data = []
    file_list = os.listdir(path)
    file_list.sort()
    for file in file_list:
        print(file_list)
        print(path+file)
        df = pd.read_csv(path + file, delimiter='\t', header=None)
        df = df.iloc[:, :-1]
        
        print(df)

        data.append(df)
    return data

def main():

    # Load all data
    ref_data_list = load_all_data(PATH_REF)
    test_data_list = load_all_data(PATH_TEST)
    

    # ax = plt.subplot(111, projection='3d')
    ax = plt.axes(projection='3d')

    for data in ref_data_list:
        ax.plot(data[0], data[1], data[2], '-')
    
    for data in test_data_list:
        ax.plot(data[0], data[1], data[2], '--', alpha=0.4)
    
    plt.show()

if  __name__ == '__main__':
    main()
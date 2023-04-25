PATH_REF = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level2/reference/'
PATH_TEST = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level2/test/'
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
        df = pd.read_csv(path + file, delimiter='\t', header=None)
        df = df.iloc[:, :-1]

        data.append(df)
    return data
def DynamicTimeWarping(s, t):
    D = np.zeros((len(s), len(t)))

    for i in range(len(s)):
        for j in range(len(t)):
            cost = abs(s[i] - t[j])
            if i == 0 and j == 0:
                D[i][j] = cost
            elif i == 0 and j > 0:
                D[i][j] = D[i][j-1] + cost
            elif i > 0 and j == 0:
                D[i][j] = D[i-1][j] + cost
            else:
                D[i][j] = min(D[i-1][j], D[i][j-1], D[i-1][j-1]) + cost

    return D[len(s)-1][len(t)-1]

def main():

    # Load all data
    ref_data_list = load_all_data(PATH_REF)
    test_data_list = load_all_data(PATH_TEST)
    
    dim = len(ref_data_list[0].columns)
    

    # ax = plt.subplot(111, projection='3d')
    ax = plt.axes(projection='3d')
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'w']

    for i in range(len(ref_data_list)):
        data = ref_data_list[i]
        ax.plot(data[0], data[1], data[2], '-', color = colors[i])
    
    for data in test_data_list:
        ref1_cost = 0
        ref2_cost = 0
        for i in range(dim):
            ref1_cost += DynamicTimeWarping(data[i], ref_data_list[0][i])
            ref2_cost += DynamicTimeWarping(data[i], ref_data_list[1][i])

        if ref1_cost < ref2_cost:
            ax.plot(data[0], data[1], data[2], '--', alpha=0.4, color = colors[0])
        else:
            ax.plot(data[0], data[1], data[2], '--', alpha=0.4, color = colors[1])

    plt.show()

if  __name__ == '__main__':
    main()
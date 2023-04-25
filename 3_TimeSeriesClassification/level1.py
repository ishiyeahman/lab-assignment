PATH_REF = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level1/reference/'
PATH_TEST = '/home/rish/projects/beginner/3_TimeSeriesClassification/dataset/level1/test/'
import os
import numpy as np
import matplotlib.pyplot as plt


def load_all_data(path):
    data = []
    file_list = os.listdir(path)
    file_list.sort()
    for file in file_list:
        d = np.loadtxt(path + file, delimiter='\t')
        data.append(d)
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
    colors = ['red', 'blue', 'green', 'yellow', 'black', 'pink', 'purple', 'orange']
    plt.plot(ref_data_list[0], '-', color=colors[0], label='reference:1')
    plt.plot(ref_data_list[1], '-', color=colors[1], label='reference:2')


    for data in test_data_list:
        
        for i in range(len(data)):
            ref1_cost = DynamicTimeWarping(ref_data_list[0], data)
            ref2_cost = DynamicTimeWarping(ref_data_list[1], data)

        if ref1_cost < ref2_cost:
            plt.plot(data, '--', color=colors[0], alpha=0.4)
        else:
            plt.plot(data, '--', color=colors[1], alpha=0.4)
    
    plt.legend()
    plt.savefig('level1.pdf')
    # plt.show()

if  __name__ == '__main__':
    main()
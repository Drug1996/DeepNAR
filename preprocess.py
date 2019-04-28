import sys
import os
import numpy as np
import pickle

ACTION_TYPE = {'none': 0, 'standing': 1, 'turning': 2}
CORRECTNESS = {'right': 0, 'wrong': 1}


def preprocess(path, savename='dataset.pkl'):
    MAXT = 0
    MINT = 1000
    os.chdir('.\\' + path)
    filelist = os.listdir()
    dataset = []

    for item in filelist:
        if not item.split('_')[0].startswith('continuous'):
            data_sample = []
            data_matrix = []
            datafile = open(item)
            datafile.readline()  # omit the first line
            data = datafile.readlines()
            MAXT = max(MAXT, len(data))
            MINT = min(MINT, len(data))
            # extract the matrix from data
            for line in data:
                line = line.split()
                line = line[1:]
                line = list(map(eval, line))
                data_matrix.append(line)
            data_matrix = np.array(data_matrix)
            data_sample.append(data_matrix)
            datafile.close()
            # extract label from name of the file
            data_sample.append(ACTION_TYPE[item.split('_')[0]])
            data_sample.append(CORRECTNESS[item.split('_')[1]])
            # save one sample into dataset
            dataset.append(data_sample)

    print('The number of data file', len(dataset))
    print('Max length is: ', MAXT)
    print('Min length is: ', MINT)
    os.chdir('.\\..')
    output = open(savename, 'wb')
    pickle.dump(dataset, output)
    output.close()


if __name__ == '__main__':
    for path in sys.argv[1:]:
        preprocess(path, 'dataset_' + path + '.pkl')
    print('Finished')

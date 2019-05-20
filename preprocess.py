import os
import numpy as np
import pickle

DATASET_NAME = ['lin', 'zhong']


def preprocess(path, savename='dataset.pkl'):
    max_t = 0
    min_t = 10000
    os.chdir(path)
    filelist = os.listdir()
    dataset = {'right_standing': [],
               'wrong_standing': [],
               'right_turning': [],
               'wrong_turning': []}

    print('Start to generate ' + savename)
    for item in filelist:
        if not item.split('_')[0].startswith('continuous'):
            data_matrix = []
            datafile = open(item)
            datafile.readline()  # omit the first line
            data = datafile.readlines()
            print(item, 'length:', len(data))
            max_t = max(max_t, len(data))
            min_t = min(min_t, len(data))
            # extract the matrix from data
            for line in data:
                line = line.split()
                line = line[1:]  # remove number column
                line = list(map(eval, line))
                data_matrix.append(line)
            datafile.close()
            data_matrix = np.array(data_matrix)
            # extract label from name of the file
            dataset[item.split('_')[1] + '_' + item.split('_')[0]].append(data_matrix)

    dataset['length_range'] = [min_t, max_t]
    for action, sample in zip(dataset.keys(), dataset.values()):
        print(action, len(sample))
    print('Max length is: ', max_t, end=', ')
    print('Min length is: ', min_t)
    os.chdir('..')
    output = open(savename, 'wb')
    pickle.dump(dataset, output)
    output.close()


def create_dataset(sub_dataset_list=['dataset_lin.pkl', 'dataset_zhong.pkl']):
    print('Start to create dataset.pkl')
    dataset = {}
    for name in sub_dataset_list:
        pkl_file = open(name, 'rb')
        sub_dataset = pickle.load(pkl_file)
        dataset[name.split('.')[0].split('_')[1]] = sub_dataset
        pkl_file.close()

    output = open('dataset.pkl', 'wb')
    pickle.dump(dataset, output)
    output.close()


if __name__ == '__main__':
    sub_dataset_list = []
    for path in DATASET_NAME:
        savename = 'dataset_' + path + '.pkl'
        preprocess(path, savename)
        sub_dataset_list.append(savename)
    create_dataset(sub_dataset_list)
    print('Finished')

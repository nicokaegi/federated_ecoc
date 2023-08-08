import glob
import os
import numpy as np
import pandas as pd
import copy


'''
this is just a that I make on the file to create quick, and dirty scripts to file a one of need every, now and then
'''
def load_client_lables(source_dir):

    starter_dir = os.getcwd()
    os.chdir(source_dir)
    target_files = glob.glob('*non_ecoc_lables*')

    clients = {}
    for item in target_files:
        clients[item] = np.loadtxt(item, delimiter=',')

    clients = {key : value for key , value in sorted(clients.items(), key=lambda item : len(np.unique(item[1])), reverse=True)}

    os.chdir(starter_dir)
    return clients


def get_test_subset_indexes(test_lables, client_lables):

    client_classes = np.unique(client_lables)
    subset = []
    pos = 0
    for item in test_lables:
        if item in client_classes:
            subset.append(pos)
        pos += 1
    return subset


def main():

    source_dir = '../landmark_proccesed_data'
    stop_point = 51


    clients = load_client_lables(source_dir)

    client_values = list(clients.items())[1:stop_point]

    set = client_values[0][1]
    lengths = len(client_values[0][1])
    clients_list = []
    for value in client_values:
        clients_list.append(value[0].split('_')[0] + '.npz')
        set = np.union1d(set, value[1])
        lengths += len(value[1])

    classes = np.loadtxt('../landmark_proccesed_data/landmark_test_non_ecoc_lables.txt')
    print(classes.shape)
    class_counts = np.unique(classes, return_counts=True)[1]
    print(len(np.unique(set)))

    test_set = np.load('../landmark_test.npz')

    test_images = test_set[test_set.files[0]]
    test_lables = test_set[test_set.files[1]]
    print(test_lables.shape)
    starter_dir = os.getcwd()
    os.chdir(source_dir)
    pos = 0
    client_list = []
    client_test_sizes = []

    client_stat_dicitonary = {}

    for client in client_values:
        indexes = get_test_subset_indexes(classes, client[1])
        npz_name = client[0].split('_')[0] + ".npz"
        print(npz_name)
        client_npz = np.load(npz_name)
        client_list.append(client_list)

        client_test_images = test_images[indexes]
        client_test_lables = test_lables[indexes]

        client_stat_dicitonary[npz_name] = [client_test_images.shape[0], round(client_test_images.shape[0]/len(classes), 2), len(client[1]), len(np.unique(client[1]))]
        #np.savez(npz_name, client_npz[client_npz.files[0]], client_npz[client_npz.files[1]], client_test_images, client_test_lables)

        pos += 1

    os.chdir(starter_dir)


    out_frame = pd.DataFrame.from_dict(client_stat_dicitonary, orient='index')

    out_frame.to_csv('client_50_stat_file', sep=',')

    #print(clients_list)
    #np.savetxt('clients_list_50.txt',clients_list, delimiter=',', fmt="%s")
    print(client_test_sizes)
    #np.savetxt('clients_list_50_shapes.txt',client_test_sizes, delimiter=',', fmt="%s")
    #print(len(np.unique(set)))
    print(lengths)

if __name__ == '__main__':
    main()

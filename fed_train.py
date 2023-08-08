import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torch.multiprocessing import Pool
import torch.multiprocessing as mp

import sys
import copy
import os
import json
import time
import random
from pathlib import Path


'''
quick, and dirty attempt at a parrall training

Process_list = []
p = None
pos = 0
for client in clients_list:
    Process_list.append([client, train_image_list[pos], train_lable_list[pos], batch, epochs, client_data_list[pos], _round, run_name])
    if len(Process_list) == workers:
        p = Pool(len(Process_list))
        p.map(training_helper, Process_list)
        Process_list = []
    pos += 1
if len(Process_list) != 0:
    p = Pool(len(Process_list))
    p.map(training_helper, Process_list)
    Process_list = []
'''

class Main_Model(nn.Module):

    device = None

    def __init__(self, device_type):
        self.device = torch.device(device_type)
        super(Main_Model,self).__init__()
        self.eye = torchvision.models.mobilenet_v2(pretrained=True)
        self.fc = nn.Linear(1000, 64)
        self.output = nn.Linear(64, 1)
        self.to(self.device)

    def forward(self, x):
        y = self.eye(x)
        y = self.fc(y)
        y = self.output(y)
        return y


    def fit(self,X, y, batch_size, epochs):

        self.train(True)
        train_set = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        train_set = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        pos = 0

        total_loss = 0

        while( pos < epochs):
            start_time = time.time()
            running_loss = 0
            for image_batch, label_batch in iter(train_set):
                image_batch, label_batch = image_batch.to(self.device).to(torch.float32), label_batch.to(self.device)
                optimizer.zero_grad()
                outputs = self(image_batch)
                loss = loss_fn(torch.flatten(outputs),label_batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().item()

            total_loss += running_loss
            epoch_time = (time.time() - start_time)
            print("-{}/{}".format(pos+1,epochs),"train time:", epoch_time,"loss total:", running_loss)

            pos += 1


        print('done')
        return total_loss

    def predict(self,X):
      X = torch.tensor(X, dtype=torch.float32, device=self.device)
      with torch.no_grad():
          output = self(X)
          return torch.sigmoid(output).round()

    def eval(self,X, y, batch_size):
        self.train(False)
        test_set = torch.utils.data.TensorDataset(torch.tensor(X),torch.tensor(y))
        #test_set = HDF5Dataset(url)
        test_set = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
        right = 0

        for image_batch, label_batch in iter(test_set):
            image_batch, label_batch = image_batch.to(self.device, torch.float32), label_batch.to(self.device)
            output_batch = self.predict(image_batch)
            right += (torch.flatten(output_batch) == label_batch).sum().item()

        after = (right/y.shape[0])
        print("after: ", after)

        return after

def training_helper(active, count, client, train_data, test_data, _round, run_name, client_name, epochs, batch):

    loss = client.fit(train_data[0], train_data[1], batch, epochs)
    with open('losses/{}_loss_{}_{}.txt'.format(run_name, _round, client_name),'w' ) as output_file:
        output_file.write(str(loss))

    client_perfomance = client.eval(test_data[0], test_data[1], 500)
    with open('losses/{}_client_{}_{}.txt'.format(run_name, _round, client_name),'w' ) as output_file:
        output_file.write(str(client_perfomance))
    active[count] = 0

def train_clients(clients_list, train_data, test_data, _round, run_name, client_data_list, epochs=10, batch=32, workers=1,devices=[]):

    mgr = mp.Manager()
    active = mgr.list()

    for item in range(workers):
        active.append(0)

    proccess_list = []
    pos = 0
    while(pos < len(clients_list)):
        count = 0
        for slot in active:
            if pos == len(clients_list):
                break
            if slot == 0:
                #instaniate a new proccess at position count
                print("alocating", pos)
                arguments = (active, count, clients_list[pos], train_data[pos], test_data[pos], _round, run_name, client_data_list[pos], epochs, batch)
                p = mp.Process(target=training_helper, args=arguments)
                proccess_list.append(p)
                active[count] = 1
                p.start()
                pos += 1
            count += 1
        time.sleep(.5)

    for p in proccess_list:
        if p.is_alive():
            p.join()


def base_line(input_tuple):

    print(input_tuple[5])
    loss = input_tuple[0].fit(input_tuple[1], input_tuple[2], input_tuple[3], input_tuple[4])
    with open('losses/{}_loss_{}.txt'.format(run_name, input_tuple[5]),'w' ) as output_file:
        output_file.write(str(loss))
    round_accuracy =  input_tuple[0].eval(input_tuple[6], input_tuple[7], 32)
    with open('test_accuracy/{}_test_{}.txt'.format(run_name, input_tuple[5]),'w') as output_file:
        output_file.write(str(round_accuracy))
    print('test_done')

def train_base_line(clients_list,train_image_list, train_lable_list, test_images, test_lables, epochs=400, batch=32, workers=1):

    Process_list = []
    p = Pool(workers)
    pos = 0
    for client in clients_list:
        Process_list.append((client, train_image_list[pos], train_lable_list[pos], batch, epochs, client_data_list[pos], test_images, test_lables))
        if len(Process_list) == workers:
            p = Pool(len(Process_list))
            p.map(training_helper, Process_list)
            Process_list = []
        pos += 1
    if len(Process_list) != 0:
        p = Pool(len(Process_list))
        p.map(training_helper, Process_list)
        Process_list = []

def aggergate_clients(target_model, clients_list, orginal_clients, lr=0.01):

    tempoarary_device =  torch.device('cpu')
    output_device = torch.device('cuda:0')
    state_dict = target_model.state_dict()
    for layer in state_dict:
        delta = (orginal_clients[0].state_dict()[layer].to(tempoarary_device) - clients_list[0].state_dict()[layer].to(tempoarary_device) )
        for pos in range(1,len(clients_list)):
            client_state = clients_list[pos].state_dict()
            orginal_state = orginal_clients[pos].state_dict()
            delta = delta + (orginal_state[layer].to(tempoarary_device) - client_state[layer].to(tempoarary_device))
        delta = delta/len(clients_list)
        state_dict[layer] = state_dict[layer].to(tempoarary_device) - lr*delta
        state_dict[layer].to(output_device)
    target_model.load_state_dict(state_dict)

def set_up_clients(model_constructor, number_of_clients, devices):

    client_list = []
    client_list.append(model_constructor(devices[0]))
    state_dict = client_list[0].state_dict()
    pos = 1
    for item in range(1,number_of_clients):
        client_list.append(model_constructor(devices[item % len(devices)]))
        client_list[1].load_state_dict(state_dict)
    return client_list

def copy_weights(target_model, client_list):

    new_state_dict = target_model.state_dict()
    for client in client_list:
        client.load_state_dict(new_state_dict)

def main():

    run_name = 'mobilenet_v2_50'
    column = 62
    rounds = 80
    number_of_clients = 49
    global client_data_list #this global beacuse I relized that when recording the losses that I shold probaly use the client names, right after everything else was written.
    client_data_list = np.loadtxt('../landmark_proccesed_data/clients_list_50.txt', dtype=str)

    #'cuda:7','cuda:6','cuda:5','cuda:4', 'cuda:3','cuda:2','cuda:1' len(client_data_list)

    #load, and prep the data
    fed_test_data = np.load('../landmark_test.npz')
    fed_test_images = fed_test_data[fed_test_data.files[0]].swapaxes(1,3)
    fed_test_lables = fed_test_data[fed_test_data.files[1]][:, column]

    train_data = []
    test_data = []

    pos = 0
    for client in client_data_list:
        data = np.load('../landmark_proccesed_data/' + client)
        print(pos, client, data.files)
        train_data.append((data[data.files[0]].swapaxes(1,3), data[data.files[1]][:, column]))
        test_data.append((data[data.files[2]].swapaxes(1,3), data[data.files[3]][:, column]))

        pos += 1

    #the federated alogrithm

    #create the intial model
    target_model = Main_Model('cuda:0')
    #set up each client
    clients_list = set_up_clients(Main_Model, number_of_clients , ['cuda:7','cuda:6','cuda:5','cuda:4', 'cuda:3','cuda:2','cuda:1', 'cuda:0'])


    for _round in range(rounds):
        print(_round)
        #save a pre-training copy of the clients for the later averaging
        orginal_clients = copy.deepcopy(clients_list)
        #train the clients stored in client list
        train_clients(clients_list, train_data, test_data, _round, run_name, client_data_list, workers=8, epochs=10, batch=32)
        print('averaging')
        #average the delta of the clients weights from before and after training, and the newly created weights in the test model
        aggergate_clients(target_model, clients_list, orginal_clients, lr=1)
        round_accuracy = target_model.eval(fed_test_images, fed_test_lables, 32)
        with open('test_accuracy/{}_test_{}.txt'.format(run_name,_round),'w') as output_file:
            output_file.write(str(round_accuracy))
        copy_weights(target_model, clients_list)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()

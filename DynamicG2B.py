# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dgl
import dgl.nn.pytorch.conv as gnn_conv

import time
import random

from sklearn import metrics

try:
    import google.colab

    IN_COLAB = True
except:
    IN_COLAB = False


def Average(lst):
    return sum(lst) / len(lst)

def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di

def one_hot(l, classnum=1):  # classnum fix some special case
    one_hot_l = np.zeros((len(l), max(l.max() + 1, classnum)))
    for i in range(len(l)):
        one_hot_l[i][l[i]] = 1
    return one_hot_l


class MyGNN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(MyGNN, self).__init__()

        self.nhid = nhid
        self.nclass = nclass
        self.num_layers = 2
        self.dropout = dropout

        self.gc1 = gnn_conv.SAGEConv(nfeat, nhid, aggregator_type='mean')
        self.gc2 = gnn_conv.SAGEConv(nhid, nhid, aggregator_type='mean')

        self.gc3 = gnn_conv.GATConv(nhid, nhid, num_heads=1)

        self.LS_end = nn.LSTM(input_size=nhid, hidden_size=nclass, num_layers=2, batch_first=True, bidirectional=True)

        self.linear22 = nn.Linear(nhid * 2, nclass)

    def forward(self, x, adj):
        out = []
        now_adj = adj[:, 0, :].clone()

        time_wise_attention = []

        for i in range(1, adj.shape[1]):
            adj2 = dgl.from_networkx(nx.Graph(adj[:, i, :].numpy()))
            adj2 = dgl.add_self_loop(adj2)
            h0 = torch.zeros(self.num_layers * 2, x[:, i, :].size(0), self.nhid)  # 2 for bidirection
            c0 = torch.zeros(self.num_layers * 2, x[:, i, :].size(0), self.nhid)

            one_out = F.relu(self.gc1(adj2, x[:, i, :]))

            one_out = F.dropout(one_out, self.dropout, training=self.training)

            one_out = self.gc2(adj2, one_out)
            one_out = F.dropout(one_out, self.dropout, training=self.training)
            one_out, yy = self.gc3(adj2, one_out, get_attention=True)
            one_out = one_out.reshape(one_out.shape[0], one_out.shape[2])

            out += [one_out]

        out = torch.stack(out, 1)

        out = self.LS_end(out, (h0, c0))[0][:, -1, :]

        out = self.linear22(out)

        return F.log_softmax(out, dim=1)


def train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, model_type, file_name):
    t = time.time()
    model.train()
    optimizer.zero_grad()

    output = model(features, adj)


    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    pred_labels = torch.argmax(output, axis=1)
    acc_train = metrics.accuracy_score(pred_labels[idx_train].cpu().detach().numpy(),
                                       labels[idx_train].cpu().detach().numpy())

    loss_train.backward(retain_graph=True)
    optimizer.step()

    # validation
    model.eval()

    output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = metrics.accuracy_score(pred_labels[idx_val].cpu().detach().numpy(),
                                     labels[idx_val].cpu().detach().numpy())

    performance_file = open(file_name + "_performance.txt", "a+")

    print('Epoch: {:02d}'.format(epoch + 1),
          'loss_train: {:.3f}'.format(loss_train.item()),
          'acc_train: {:.3f}'.format(acc_train.item()),
          'loss_val: {:.3f}'.format(loss_val.item()),
          'acc_val: {:.3f}'.format(acc_val.item()),
          'time: {:.3f}s'.format(time.time() - t))

    performance_file.write('Epoch: {:02d}'.format(epoch + 1) + '; ' +
                           'loss_train: {:.3f}'.format(loss_train.item()) + '; ' +
                           'acc_train: {:.3f}'.format(acc_train.item()) + '; ' +
                           'loss_val: {:.3f}'.format(loss_val.item()) + '; ' +
                           'acc_val: {:.3f}'.format(acc_val.item()) + '; ' +
                           'time: {:.3f}s'.format(time.time() - t) + '\n')
    performance_file.close()

    return acc_val


def test(model, features, adj, labels, idx_test):
    model.eval()
    # print("\nNow in Testing: \n")
    output = model(features, adj)
    pred_labels = torch.argmax(output, axis=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = metrics.accuracy_score(labels[idx_test].cpu().detach().numpy(),
                                      pred_labels[idx_test].cpu().detach().numpy())
    f1_test = metrics.f1_score(labels[idx_test].cpu().detach().numpy(), pred_labels[idx_test].cpu().detach().numpy(),
                               average='weighted')
    auc_test = metrics.roc_auc_score(one_hot(labels[idx_test].cpu().detach().numpy()),
                                     output[idx_test].cpu().detach().numpy(), multi_class='ovr', average='weighted')

    # print('loss_test: {:.4f}'.format(loss_test.item()),
    #       'acc_test: {:.4f}'.format(acc_test.item()),
    #       'auc_test: {:.4f}'.format(auc_test.item()),
    #       'f1_test: {:.4f}'.format(f1_test.item()),'\n')

    return loss_test.item(), acc_test, auc_test, f1_test


def single_train_and_test(lambda_matrix, Probability_matrix, features, adj, labels, idx_train, idx_val, idx_test,
                          model_type, normalize=False, file_name='pop'):
    model = MyGNN(nfeat=features.shape[2], nhid=args_hidden, nclass=class_num, dropout=args_dropout)
    if args_cuda:
        model = model.to(torch.device('cuda:0'))  # .cuda()
        features = features.cuda()
        adj = adj.to(torch.device('cuda:0'))
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    # optimizer and train
    optimizer = optim.Adam(model.parameters(), lr=args_lr, weight_decay=args_weight_decay)
    # Train model
    train_time_1 = time.time()
    best_val = 0
    for epoch in range(args_epochs):
        acc_val = train(epoch, model, optimizer, features, adj, labels, idx_train, idx_val, model_type, file_name)

        if acc_val > best_val:
            best_val = acc_val
            loss, acc, auc, f1 = test(model, features, adj, labels, idx_test)
            test_best_val = [loss, acc, auc, f1]
    train_time_2 = time.time()

    print(train_time_2 - train_time_1)

    print("acc= " + str(acc) + ", auc= " + str(auc) + ", f1= " + str(f1) + "\n")
    full_list.append((acc, auc, f1))

    return loss, acc, auc, f1
"""# Run Exp for Spectral Clustering and GCN with Decay Rates

# Run Exp on Simulated and Real Datasets
"""


def load_real_data(dataset_name):
    print(dataset_name)
    dataset_dict = dict()
    dataset_dict["DBLP3"] = "DBLP3.npz"
    dataset_dict["DBLP5"] = "DBLP5.npz"

    print(dataset_dict[dataset_name])

    dataset = np.load(dataset_dict[dataset_name])

    Graphs = torch.LongTensor(dataset['adjs'])  # (n_time, n_node, n_node)
    Graphs = torch.transpose(Graphs, 0, 1)  # (n_node, n_time, n_node)

    now_adj = Graphs[:, 0, :].clone()

    for i in range(1, Graphs.shape[1]):  # time_steps

        now_adj += Graphs[:, i, :].clone()

    d = torch.sum(now_adj, axis=1)

    non_zero_index = torch.nonzero(d, as_tuple=True)[0]

    Graphs = Graphs[non_zero_index, :, :]
    Graphs = Graphs[:, :, non_zero_index]

    Labels = torch.LongTensor(np.argmax(dataset['labels'], axis=1))  # (n_node, num_classes) argmax
    Features = torch.LongTensor(dataset['attmats'])  # (n_node, n_time, att_dim)

    Features = Features[non_zero_index]
    Labels = Labels[non_zero_index]

    # shuffle datasets
    number_of_nodes = Graphs.shape[0]

    nodes_id = list(range(number_of_nodes))

    random.shuffle(nodes_id)


    ## 50% Training Data

    # idx_train = torch.LongTensor(nodes_id[:(5 * number_of_nodes) // 10])
    # idx_val = torch.LongTensor(nodes_id[(5 * number_of_nodes) // 10: (7 * number_of_nodes) // 10])
    # idx_test = torch.LongTensor(nodes_id[(7 * number_of_nodes) // 10: number_of_nodes])


    ## 70% Training Data
    idx_train = torch.LongTensor(nodes_id[:(7 * number_of_nodes) // 10])
    idx_val = torch.LongTensor(nodes_id[(7 * number_of_nodes) // 10: (9 * number_of_nodes) // 10])
    idx_test = torch.LongTensor(nodes_id[(9 * number_of_nodes) // 10: number_of_nodes])

    return Features.float(), Graphs.float(), Labels.long(), idx_train, idx_val, idx_test, []


def test_real_dataset(file_name):
    summary_file = open(file_name+".txt", "a+")

    t = time.time()
    lambda_matrix = None
    total_loss = 0
    total_acc = 0
    total_norm = []
    loss, acc, auc, f1= single_train_and_test(lambda_matrix, Probability_matrix, features, adj, labels,
                                                         idx_train,
                                                         idx_val, idx_test, model_type, normalize=args_normalize,
                                                         file_name=file_name)
    if type(lambda_matrix) != type(None):
        summary_file.write("accuracy= {:.6f}".format(acc) +
                           "\tauc= {:.6f}".format(auc) +
                           "\tf1= {:.6f}".format(f1) +
                           "\n")
    else:
        summary_file.write("accuracy= {:.6f}".format(acc) +
                           "\tauc= {:.6f}".format(auc) +
                           "\tf1= {:.6f}".format(f1) +
                           "\n")

    summary_file.close()
    return file_name


total_time_3 = time.time()

full_list = []

dataset_name = "DBLP3"
# dataset_name = "DBLP5"

features, adj, labels, idx_train, idx_val, idx_test, Probability_matrix = load_real_data(dataset_name)
class_num = int(labels.max()) + 1
total_adj = adj
total_labels = labels
model_type = 'MyGNN'  # GraphSage_BiLSTM_GAT, GCN, GAT, GraphSage #dynamic_spec, DynAERNN, #GCNLSTM,#MyGNN, EGCN, RNNGCN, TRNNGCN, GAT_BiLSTM, MyGNN
args_hidden = class_num
args_dropout = 0.5
args_lr = 0.0025
args_weight_decay = 5e-4
args_epochs = 30
args_no_cuda = True
args_cuda = not args_no_cuda and torch.cuda.is_available()
args_normalize = True
file_name = dataset_name + '_' + model_type

print("\n" + model_type + "\n")
for i in range(30):
    print("\nIteration: " + str(i) + "\n")
    test_real_dataset(file_name)
print(full_list)


acc_list = [x for (x, y, z) in full_list]
auc_list = [y for (x, y, z) in full_list]
f1_list = [z for (x, y, z) in full_list]
print(len(acc_list))
print(Average(acc_list), Average(auc_list), Average(f1_list), sep=',')
total_time_4 = time.time()
print(total_time_4 - total_time_3)

import csv
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import yaml
import torch
import os
from model.GraphTransformer import GraphTransformer
from dataloader import init_dataloader
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

count = {}
matrix_pearson = []
matrix_attention = []
scores = []
scores_all = []

def test_score(dataloader, model):
    result = []
    sc = []
    for data_in, pearson, label, pseudo in dataloader:
        label = label.long()
        data_in, pearson, label, pseudo = data_in.to(
            device), pearson.to(device), label.to(device), pseudo.to(device)
        [output, score, cor_matrix], matrix, _ = model(data_in, pearson, pseudo)

        result += F.softmax(output, dim=1)[:, 1].tolist()

        global matrix_pearson
        matrix_pearson += data_in.detach().cpu().numpy().tolist()
        global matrix_attention
        matrix_attention += matrix.detach().cpu().numpy().tolist()



with open('./setting/abide_RGTNet.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    model = GraphTransformer(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    model.load_state_dict(torch.load('./model.pt'))

    torch.cuda.set_device(0)

    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)

    csv_reader = csv.reader(open('aal_labels2.csv', encoding='utf-8'))
    label = [row[1] for row in csv_reader]
    # visulize the FC matrix using Pearson
    matrix_pearson = np.array(matrix_pearson)

    connectivity = ConnectivityMeasure(kind='correlation')
    connectivity_matrices = connectivity.fit_transform(matrix_pearson.swapaxes(1, 2))
    mean_connectivity_matrices = connectivity_matrices.mean(axis=0)

    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=1.2, vmin=-0.05, colorbar=True, reorder=False, title="")
    plotting.show()


    # visulize the FC matrix using GraphEncoder
    matrix_attention = np.array(matrix_attention)


    mean_connectivity_matrices = matrix_attention.mean(axis=0)
    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8758, vmin=0.862,
                         colorbar=True,
                         reorder=False, title="")

    plotting.show()
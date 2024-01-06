import csv
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
matrix_con = []
matrix_asd = []
matrix_asd_label = []
matrix_con_label = []
scores = []
scores_all = []

def test_score(dataloader, model):
    result = []
    matrix_all= []

    for data_in, pearson, label, pseudo in dataloader:
        label = label.long()
        data_in, pearson, label, pseudo = data_in.to(
            device), pearson.to(device), label.to(device), pseudo.to(device)
        [output, score, cor_matrix], matrix, _ = model(data_in, pearson, pseudo)

        result += F.softmax(output, dim=1)[:, 1].tolist()
        matrix_all += matrix.detach().cpu().numpy().tolist()

    result = np.array(result)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    matrix_all = np.array(matrix_all)
    global matrix_asd
    matrix_asd += matrix_all[result == 1].tolist()
    global matrix_con
    matrix_con += matrix_all[result == 0].tolist()



with open('./setting/abide_RGTNet.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    # print(config['data']['batch_size'])
    model = GraphTransformer(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    model.load_state_dict(torch.load('./model.pt'))


    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)

    csv_reader = csv.reader(open('aal_labels2.csv', encoding='utf-8'))
    label = [row[1] for row in csv_reader]
    # visulize the FC matrix based on ASD samples using GraphEncoder
    matrix_asd = np.array(matrix_asd)

    mean_connectivity_matrices = matrix_asd.mean(axis=0)
    np.save('./asd.npy', mean_connectivity_matrices)

    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8643, vmin=0.848, colorbar=True, reorder=False, title="")
    plotting.show()
    plotting.savefig("brain_image.png")


    # visulize the FC matrix based on NC samples using GraphEncoder
    matrix_con = np.array(matrix_con)

    mean_connectivity_matrices = matrix_con.mean(axis=0)
    np.save('./health.npy', mean_connectivity_matrices)
    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8965, vmin=0.873,
                         colorbar=True,
                         reorder=False, title="")
    plotting.show()
import numpy
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

        cor = cor_matrix[:, :, 0, :].to(device)
        score = cor[:, 0, :]
        for i in range(3):
            score += cor[:, i + 1, :]

        score = score[:, 1:]
        sc += score.detach().cpu().numpy().tolist()
        global scores_all
        scores_all += score.detach().cpu().numpy().tolist()

    result = np.array(result)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    sc = np.array(sc)
    global scores
    # only considering the ASD samples
    scores += sc[result == 1].tolist()



with open('setting/abide_RGTNet.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    model = GraphTransformer(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    model.load_state_dict(torch.load('./model.pt'))


    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)

    # calculating the inductive score based on ASD samples
    scores = np.array(scores)
    scores = scores.mean(0)
    index = numpy.argsort(scores)
    scores = numpy.sort(scores)
    print(scores[-10:])
    print(index[-10:])
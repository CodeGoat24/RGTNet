import os

import torch

from datetime import datetime
from util import Logger, accuracy, TotalMeter
import numpy as np
from pathlib import Path
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from util.prepossess import mixup_criterion, mixup_data
from util.loss import mixup_cluster_loss, topk_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"


class BasicTrain:

    def __init__(self, train_config, model, optimizers, dataloaders, log_folder) -> None:
        self.logger = Logger()
        self.model = model.to(device)
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = train_config['epochs']
        self.optimizers = optimizers
        self.best_acc = 0
        self.best_model = None
        self.best_acc_val = 0
        self.best_auc_val = 0
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        self.pool_ratio = train_config['pool_ratio']
        self.group_loss = train_config['group_loss']

        self.sparsity_loss = train_config['sparsity_loss']
        self.sparsity_loss_weight = train_config['sparsity_loss_weight']

        self.save_path = log_folder

        self.save_learnable_graph = True

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss, self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy, self.edges_num = [
                TotalMeter() for _ in range(7)]

        self.loss1, self.loss2, self.loss3 = [TotalMeter() for _ in range(3)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy, self.test_accuracy,
                      self.train_loss, self.val_loss, self.test_loss, self.edges_num,
                      self.loss1, self.loss2, self.loss3]:
            meter.reset()

    def train_per_epoch(self, optimizer):


        self.model.train()
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        for data_in, pearson, label, pseudo in self.train_dataloader:

            label = label.long()

            data_in, pearson, label, pseudo = data_in.to(
                device), pearson.to(device), label.to(device), pseudo.to(device)

            inputs, nodes, targets_a, targets_b, lam = mixup_data(
                data_in, pearson, label, 1, device)

            [output, score, cor_matrix], learnable_matrix, edge_variance = self.model(inputs, nodes, pseudo)

            loss = 2 * mixup_criterion(
                self.loss_fn, output, targets_a, targets_b, lam)

            # if self.group_loss:
            #     loss += mixup_cluster_loss(learnable_matrix,
            #                                targets_a, targets_b, lam)

            # if self.sparsity_loss:
            #     sparsity_loss = self.sparsity_loss_weight * \
            #         torch.norm(learnable_matrix, p=1)
            #     loss += sparsity_loss


            loss += 0.001*topk_loss(score, self.pool_ratio)


            self.train_loss.update_with_weight(loss.item(), label.shape[0])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step()
            top1 = accuracy(output, label)[0]
            self.train_accuracy.update_with_weight(top1, label.shape[0])
            self.edges_num.update_with_weight(edge_variance, label.shape[0])

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()

        for data_in, pearson, label, pseudo in dataloader:
            label = label.long()
            data_in, pearson, label, pseudo = data_in.to(
                device), pearson.to(device), label.to(device), pseudo.to(device)
            [output, score, cor_matrix], _, _ = self.model(data_in, pearson, pseudo)
            loss = self.loss_fn(output, label)
            loss_meter.update_with_weight(
                loss.item(), label.shape[0])
            top1 = accuracy(output, label)[0]
            acc_meter.update_with_weight(top1, label.shape[0])
            result += F.softmax(output, dim=1)[:, 1].tolist()
            labels += label.tolist()

        auc = roc_auc_score(labels, result)

        result = np.array(result)
        result[result > 0.5] = 1
        result[result <= 0.5] = 0
        metric = precision_recall_fscore_support(
            labels, result, average='micro')
        con_matrix = confusion_matrix(labels, result)
        return [auc] + list(metric), con_matrix

    def generate_save_learnable_matrix(self):
        learable_matrixs = []

        labels = []

        for data_in, nodes, label, pseudo in self.test_dataloader:
            label = label.long()
            data_in, nodes, label, pseudo = data_in.to(
                device), nodes.to(device), label.to(device), pseudo.to(device)
            _, learable_matrix, _ = self.model(data_in, nodes, pseudo)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)

    def save_result(self, results, txt, train_loss, test_loss):

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)
        
        np.save(self.save_path/"train_loss.npy",
                train_loss, allow_pickle=True)
        np.save(self.save_path/"test_loss.npy",
                test_loss, allow_pickle=True)
        
        with open(self.save_path / "training_info.txt", 'a', encoding='utf-8') as f:
            f.write(txt)
        torch.save(self.best_model.state_dict(), self.save_path/f"model_{self.best_acc}%.pt")
        



    def train(self):
        training_process = []
        txt = ''
        train_loss = []
        test_loss = []
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0])
            val_result, _ = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result, con_matrix = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)

            # if round(self.best_acc_val,1) < round(self.val_accuracy.avg,1):
            #     self.best_acc_val = self.val_accuracy.avg
            #     self.best_model = self.model
            #     self.best_acc = self.test_accuracy.avg
            # elif round(self.best_acc_val,1) == round(self.val_accuracy.avg,1):
            #     if self.test_accuracy.avg > self.best_acc:
            #         self.best_model = self.model
            #         self.best_acc = self.test_accuracy.avg

            # if round(self.best_auc_val,2) < round(val_result[0],2):
            #     self.best_auc_val = val_result[0]
            #     self.best_model = self.model
            #     self.best_acc = self.test_accuracy.avg
            # elif round(self.best_auc_val,2) == round(val_result[0],2):
            #     if self.test_accuracy.avg > self.best_acc:
            #         self.best_model = self.model
            #         self.best_acc = self.test_accuracy.avg

            if self.best_acc <= self.test_accuracy.avg:
                self.best_acc = self.test_accuracy.avg
                self.best_model = self.model

            if (con_matrix[0][0] + con_matrix[1][0]) != 0:
                SEN = con_matrix[0][0] / (con_matrix[0][0] + con_matrix[1][0])
            else:
                SEN = 0

            if (con_matrix[1][1] + con_matrix[0][1]) != 0:
                SPE = con_matrix[1][1] / (con_matrix[1][1] + con_matrix[0][1])
            else:
                SPE = 0

            self.logger.info(" | ".join([
                f'Epoch[{epoch}/{self.epochs}]',
                f'Train Loss:{self.train_loss.avg: .3f}',
                f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                # f'Edges:{self.edges_num.avg: .3f}',
                f'Test Loss:{self.test_loss.avg: .3f}',
                f'Val Accuracy:{self.val_accuracy.avg: .3f}%',
                f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                f'Val AUC:{val_result[0]:.2f}',
                f'Test AUC:{test_result[0]:.4f}',
                f'Test SEN:{SEN:.4f}',
                f'Test SPE:{SPE:.4f}'
            ]))

            txt += f'Epoch[{epoch}/{self.epochs}] '+f'Train Loss:{self.train_loss.avg: .3f} '+f'Train Loss:{self.test_loss.avg: .3f} '+f'Train Accuracy:{self.train_accuracy.avg: .3f}% '+f'Val Accuracy:{self.val_accuracy.avg: .3f}% '+f'Test Accuracy:{self.test_accuracy.avg: .3f}% '+f'Val AUC:{val_result[0]:.3f} '+f'Test AUC:{test_result[0]:.4f}'+f'Test SEN:{SEN:.4f}'+f'Test SPE:{SPE:.4f}'+'\n'

            training_process.append([self.train_accuracy.avg, self.train_loss.avg,
                                     self.val_loss.avg, self.test_loss.avg]
                                    + val_result + test_result)
            train_loss.append(self.train_loss.avg)
            test_loss.append(self.test_loss.avg)

        now = datetime.now()
        date_time = now.strftime("%m-%d-%H-%M-%S")
        self.save_path = self.save_path/Path(f"{self.best_acc: .3f}%_{date_time}")
        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process, txt, train_loss, test_loss)



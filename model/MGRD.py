import os
import time
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from model.dataset import TSADDataset
import numpy as np
import torch.nn as nn
from model.model import Former


class MGRD():

    def __init__(self):
        self.dataset = 'PSM'
        self.batch_size = 256
        self.num_epochs = 100
        self.win_size = 100
        self.lr = 0.0001
        self.anormly_ratio = 1
        self.k = 3
        self.e_layers = 3
        self.d_model = 128
        self.d_ff = 128
        self.checkpoints = './checkpoints'
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def norm_data(self, adds: TSADDataset):
        if adds is None:
            return None
        cold_data, data = adds.cold_data, adds.data
        cold_data = np.nan_to_num(cold_data)
        data = np.nan_to_num(data)
        cold_size = len(cold_data)
        all_data = np.concatenate([cold_data, data], axis=0)
        X = all_data[:, :-1]

        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        all_data[:, :-1] = X
        adds.cold_data = all_data[:cold_size, :]
        adds.data = all_data[cold_size:, :]

        return adds

    def train(self, ds, valid_ds = None, cb_progress=lambda x:None):
        ds = self.norm_data(ds)
        valid_ds = self.norm_data(valid_ds)

        self.num_fea = ds.data.shape[-1] - 1

        self.train_loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = DataLoader(valid_ds, batch_size=self.batch_size, shuffle=False)

        self.model = Former(self.d_model,self.d_ff,self.e_layers,self.num_fea)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

        self.criterion = nn.MSELoss()

        setting = '{}_{}_{}_{}_{}'.format(
            self.dataset,
            self.batch_size,
            self.anormly_ratio,
            self.d_model,
            self.d_ff)

        path = os.path.join(self.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)
        self.path = path

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss_list = []
            train_steps = len(self.train_loader)
            epoch_time = time.time()
            self.model.train()
            for i, input_data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                cold_data, x, y = self.filter_nan(*input_data)
                if len(x) == 0: continue
                input = x.float().to(self.device)

                x_G, x_L, Global_Attn, Local_Attn = self.model(input)

                kl_G_to_L = 0.0
                kl_L_to_G = 0.0

                for u in range(len(Global_Attn)):
                    kl_G_to_L += (torch.mean(my_kl_loss(Global_Attn[u], (
                            Local_Attn[u] / torch.unsqueeze(torch.sum(Local_Attn[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                             self.win_size)).detach())) + torch.mean(
                        my_kl_loss(
                            (Local_Attn[u] / torch.unsqueeze(torch.sum(Local_Attn[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                              self.win_size)).detach(),
                            Global_Attn[u])))

                    kl_L_to_G += (torch.mean(my_kl_loss(
                        (Local_Attn[u] / torch.unsqueeze(torch.sum(Local_Attn[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                          self.win_size)),
                        Global_Attn[u].detach())) + torch.mean(
                        my_kl_loss(Global_Attn[u].detach(), (
                                Local_Attn[u] / torch.unsqueeze(torch.sum(Local_Attn[u], dim=-1), dim=-1).repeat(1, 1,
                                                                                                                 1,
                                                                                                                 self.win_size)))))

                kl_G_to_L = kl_G_to_L / len(Global_Attn)
                kl_L_to_G = kl_L_to_G / len(Global_Attn)

                recon_loss1 = self.criterion(x_G, input)
                recon_loss2 = self.criterion(x_L, input)

                KL_loss1 = kl_G_to_L
                KL_loss2 = kl_L_to_G

                loss1 = recon_loss1 - self.k * KL_loss1
                loss2 = recon_loss2 - self.k * KL_loss2

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss_list)

            vali_loss1 = self.eval_data(self.valid_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))

            early_stopping(vali_loss1, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def eval_data(self, dataloader) -> float:
        self.model.eval()
        loss_list = []
        criterion = nn.MSELoss()
        for i, input_data in enumerate(dataloader):
            cold_data, x, y = self.filter_nan(*input_data)
            if len(x) == 0: continue
            x = x.float().to(self.device)

            x_G, x_L, Global_Attn, Local_Attn = self.model(x)

            recon_loss1 = criterion(x_G, x)
            recon_loss2 = criterion(x_L, x)

            loss = -(recon_loss1+recon_loss2).detach().cpu().numpy()
            loss_list.append((loss).item())

        vali_loss = np.average(loss_list)
        return vali_loss

    def predict(self, ds, cb_progress=lambda x: None):

        test_ds = self.norm_data(ds)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        self.model.eval()

        criterion = nn.MSELoss(reduce=False)

        ds.step = ds.win_size
        pre_data = None
        x_valid = None
        test_labels = []
        attens_energy = []

        for i, input_data in enumerate(self.test_loader):
            cold_data, x, y = input_data
            nandata = x.isnan()
            if nandata.any():
                dims = list(range(nandata.ndim))
                x_mask = torch.logical_not(nandata[-1].sum(dim=dims[1:-1]).type(torch.bool))
                x_valid = x_mask.sum()
                x_nan = self.win_size - x_valid
                if x.shape[0] > 1:
                    pre_data = (cold_data[-2, :], x[-2, :], y[-2, :])
                assert pre_data is not None, f"test data is less than one window size{self.win_size},Impossible to predict"
                raw_data = torch.cat((pre_data[1], pre_data[2].unsqueeze(-1)), dim=1)
                last_cold = torch.cat((pre_data[0], raw_data))[-x_nan - cold_data[1].shape[0]:-x_nan]
                last_x = torch.cat((pre_data[1], x[-1]))[-x_nan - self.win_size:-x_nan]
                last_y = torch.cat((pre_data[2], y[-1]))[-x_nan - self.win_size:-x_nan]
                cold_data[-1] = last_cold
                x[-1] = last_x
                y[-1] = last_y
            pre_data = (cold_data[-1, :], x[-1, :], y[-1, :])
            labels = y
            x = x.float().to(self.device)

            x_G, x_L, Global_Attn, Local_Attn = self.model(x)

            loss = torch.mean(criterion(x_G, x_L), dim=-1)
            cri = loss

            cri = cri.detach().cpu().numpy()
            cri = cri.reshape(-1)
            labels = labels.reshape(-1)

            if x_valid is not None:

                cri = np.concatenate((cri[:-self.win_size], cri[-x_valid:]))
                labels = np.concatenate((labels[:-self.win_size], labels[-x_valid:]))
            attens_energy.append(cri)
            test_labels.append(labels)


        attens_energy = np.concatenate(attens_energy, axis=0)

        test_labels = np.concatenate(test_labels, axis=0)
        test_energy = np.array(attens_energy)

        test_labels = np.array(test_labels)

        combined_energy = test_energy
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        pred = (test_energy > thresh).astype(int)

        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        return pred

    def filter_nan(self,cold_data,x,y):
        nandata = x.isnan()
        if nandata.any():
            dims = list(range(nandata.ndim))
            keep_data = torch.logical_not(nandata.sum(dim=dims[1:]))
            x = x[keep_data]
            y = y[keep_data]
            cold_data = cold_data[keep_data]
        return cold_data,x,y

def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='',delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name


    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss











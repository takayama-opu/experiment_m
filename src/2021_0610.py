# coding: utf-8
from collections import defaultdict
import torch
from transformers import BertConfig, BertModel
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from sklearn.metrics import classification_report
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.models import resnet34
from util.visualizer import Visualizer
from util.history import History
from stair_dataset import StairAudioDataset

ACTION_NUM = ["a001", "a002", "a018", "a033", "a094"]

ex_path = {"dataset_root":'../dataset/STAIR_ACTIONS_DATASET/'}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class MelSpecNet(nn.Module):
    def __init__(self, fine_tuning=True, n_class=5):
        super(MelSpecNet, self).__init__()
        self.n_class = n_class

        self.resnet = models.resnet34(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                      bias=False)  # RGB だと 3 だけど音声なので 2 チャンネル
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, n_class)
        )

        if fine_tuning:
            set_parameter_requires_grad(self.resnet, feature_extracting=True)
            for name, param in self.resnet.fc.named_parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

class Experiment():
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None,
                 n_class=None, n_epoch=50, n_batch=32, lr=0.001, net=None,
                 load_pretrained_state=None, test_last_hidden_layer=False,
                 show_progress=True, show_chart=False, save_state=False):
        '''
        前処理、学習、検証、推論を行う
        train_dataset: 学習用データセット
        val_dataset: 検証用データセット
        test_dataset: 評価用データセット
        （検証とテストでデータを変えたい場合は一度学習してステートセーブした後に
          テストのみでステート読み出しして再実行すること）
        （正解ラベルが無い場合は検証はスキップする）
        n_class: 分類クラス数（Noneならtrain_datasetから求める）
        n_epoch: 学習エポック数
        n_batch: バッチサイズ
        lr: 学習率
        net: 使用するネットワークのインスタンス
        load_pretrained_state: 学習済ウエイトを使う場合の.pthファイルのパス
        test_last_hidden_layer: テストデータの推論結果に最終隠れ層を使う
        show_progress: エポックの学習状況をprintする
        show_chart: 結果をグラフ表示する
        save_state: val_acc > 更新の時のstateを保存
       　　　　　　　 （load_pretrained_stateで使う）
        返り値: テストデータの推論結果
        '''

        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.n_class, self.n_epoch, self.n_batch, self.lr = n_class, n_epoch, n_batch, lr
        self.net = MelSpecNet(fine_tuning=True, n_class=self.n_class).to(device)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.n_batch, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.n_batch, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.n_batch, shuffle=True)
        self.study_loaders = {'train': self.train_loader, 'val': self.val_loader}

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.history = History()

        self.new_model_path = '../model/20210610.bin'
        self.log_path = '../result/20210610.txt'

    def train(self):
        # if self.is_study == False:
        #     log_f = open(self.log_path, 'a', encoding='utf-8')
        #     print("class weight : {}".format(self.w), file=log_f)
        #     print("best:lr {}".format(lr), file=log_f)
        for epoch in range(self.n_epoch):
            time_start = time.time()
            print('Epoch {}/{}'.format(epoch + 1, self.n_epoch))
            print('-------------')

            for phase in ['train', 'val']:

                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                self.reset_count()

                for x, y in self.study_loaders[phase]:
                    x = x.to(device)
                    y = y.to(device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = self.net(x)
                        _, predicted = torch.max(y_pred.data, 1)
                        self.total += y.size(0)
                        # loss 計算・加算
                        loss = self.criterion(y_pred, y.argmax(1))
                        self.total_loss += loss.item()
                        # 正解数 加算
                        self.correct += (predicted == y.argmax(1)).sum().item()
                        # n x n matrix 更新
                        for i in range(len(predicted)):
                            self.c_mat[torch.max(y.data, 1)[1][i]][predicted[i]] += 1

                        # 訓練時のみバックプロパゲーション
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                # ロスの合計を len(data_loader)で割る
                mean_loss = self.total_loss / len(self.study_loaders[phase])
                acc = (self.correct / self.total)

                # Historyに追加
                self.history.update(phase, mean_loss, acc)

                if (phase == 'val'):
                    if (epoch == 0) or (self.history.enable_save):
                        self.save_model()

                # Validation 結果
                if phase == 'valid':
                    print("---Validation---")
                else:
                    print("---TRAIN---")
                print(self.c_mat)
                print("Acc : %.4f" % acc)
                print("loss : {}".format(mean_loss))

            time_finish = time.time() - time_start
            print("====================================")
            print("残り時間 : {0}".format(time_finish * (self.n_epoch - epoch)))
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}\n\n".format(self.history.history['val']['loss'][-1],
                                                                 self.history.history['val']['acc'][-1]))


        v = Visualizer(self.history.history)
        v.visualize()

        return self.history.best['val']['acc']

    def test(self):
        log_f = open(self.log_path, 'a', encoding='utf-8')
        self.load_model()
        self.reset_count()
        test_index = 0
        label = []
        pred = []
        self.net.eval()
        with torch.no_grad():
            for x, y in self.test_loader:
                x = x.to(device)
                y = y.to(device)
                self.optimizer.zero_grad()

                y_pred = self.net(x)

                _, predicted = torch.max(y_pred.data, 1)

                self.correct += (predicted == torch.max(y.data, 1)[1]).sum().item()

                self.total += y.size(0)

                pred.append(predicted[0])
                label.append(torch.max(y.data, 1)[1][0])
                # 5 x 5 matrix 更新
                for i in range(len(predicted)):
                    self.c_mat[torch.max(y.data, 1)[1][i]][predicted[i]] += 1


            print("------------------------test acc------------------------", file=log_f)
            print("Test Acc : %.4f" % (self.correct / self.total), file=log_f)
            print("correct: {0}, total: {1}".format(self.correct, self.total), file=log_f)
            print("------------------------------------------------", file=log_f)
        d = classification_report([la.tolist() for la in label], [pr.tolist() for pr in pred],
                                  target_names=[self.p_label, 'その他'],
                                  output_dict=True)
        df = pd.DataFrame(d)
        print(df, file=log_f)
        log_f.close()

    def save_model(self):
        torch.save(self.net.state_dict(), self.new_model_path)
        print('\nbest score updated, Pytorch model was saved!! f1:{}\n'.format(self.history.best['valid']['f1']))

    def load_model(self):
        load_weights = torch.load(self.new_model_path,
                                  map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(load_weights)

    def reset_count(self):
        self.total_loss = 0
        self.total = 0
        self.correct = 0
        self.c_mat = np.zeros((self.n_class, self.n_class), dtype=int)



def main():
    print("miria")
    train_dataset = StairAudioDataset(mode='train', augment=False)
    val_dataset = StairAudioDataset(mode='val', augment=False)
    test_dataset = StairAudioDataset(mode='test', augment=False)
    ex = Experiment(train_dataset, val_dataset, test_dataset, n_class=5, n_epoch=1, n_batch=32)

if __name__ == '__main__':
    main()
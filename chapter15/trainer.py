from copy import deepcopy
import numpy as np

import torch

class Trainer():

    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _batchify(self, x, y, batch_size, random_split=True): # 매 epoch마다 SGD를 수애하기 위해 셔플링 후 미니배치를 만드는 과정
        if random_split:                                      # 검증과정에서 random_split가 필요없어 False로 넘어올 수 있다.
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)

        x = x.split(batch_size, dim=0)
        y = y.split(batch_size, dim=0)

        return x, y

    def _train(self, x, y, config):
        self.model.train()

        x, y = self._batchify(x, y, config.batch_size)
        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x,y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            # initialize the gradients of the model.
            self.optimizer.zero_grad()
            loss_i.backward()

            self.optimizer.step()

            if config.verbose >= 2: # 현재 학습 상황을 출력한다. config는 train.py에서 사용자의 실행시 파라미터 입력에 따른 설정값이 들어있는 객체이다.
                print('Train Iteration(%d/%d): loss=%.4e' % (i+1, len(x), float(loss_i)))

            # Don't forget to deatch to prevent memory leak.
            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn evaluation mode on.
        self.model.eval()

        # Turn on the no_grad mode to make more efficintly.
        with torch.no_grad():
            x, y = self._batchify(x, y, config.batch_size, random_split=False)
            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Valid Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            # You Must use deep copy to take a snapshot of current best weights.
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict()) # state_dict() : 모델의 가중치파라미터 값을 json 형태로 변환하여 리턴

            print('Epoch(%d/%d): trian_loss=%.4e valid_loss=%.4e lowest_loss=%.4e' % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))
        
        # Restore to best model
        self.model.load_state_dict(best_model) # best_model에 젖아된 가중치 파라미터 json값을 load_state_dict를 통해
                                               # self.model에 다시 로드한다. 
                                               # 이 코드를  통해서 학습 종료 후 오버피팅 되지 않은 가장 좋은 상태의 모델로 복원할 수 있다.


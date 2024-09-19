import os
import numpy as np
import torch
from pathlib import Path

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent) # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)

bikes_path = os.path.join(BASE_PATH, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

bikes_numpy = np.loadtxt(
  fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
  converters={
    1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7
  }
)
bikes_data = torch.from_numpy(bikes_numpy).to(torch.float)
print(bikes_data.shape)    # >>> torch.Size([17520, 17])
bikes_target = bikes_data[:, -1].unsqueeze(dim=-1)  # 'cnt'
bikes_data = bikes_data[:, :-1]   # >>> torch.Size([17520, 16])

eye_matrix = torch.eye(4)

data_torch_list = []
for idx in range(bikes_data.shape[0]):  # range(730)
  hour_data = bikes_data[idx]  # hour_data.shape: [17]
  weather_onehot = eye_matrix[hour_data[9].long() - 1]
  concat_data_torch = torch.cat(tensors=(hour_data, weather_onehot), dim=-1)
  # concat_data_torch.shape: [20]
  data_torch_list.append(concat_data_torch)

bikes_data = torch.stack(data_torch_list, dim=0)
bikes_data = torch.cat([bikes_data[:, 1:9], bikes_data[:, 10:]], dim=-1)
# Drop 'instant' and 'whethersit' columns

print(bikes_data.shape)
print(bikes_data[0])
# 이전엔 view 함수를 통해 3차원으로 바꾸었지만 이번에는 2차원 데이터 그대로 1열과 9열의 데이터를
# 없애는 과정이였습니당.
#################################################################################################
# 데이터를 어디까지 볼껀지 정하는 sequence_size
sequence_size = 24
validation_size = 96
test_size = 24
y_normalizer = 100

data_size = len(bikes_data) - sequence_size + 1
print("data_size: {0}".format(data_size))
train_size = data_size - (validation_size + test_size)
print("train_size: {0}, validation_size: {1}, test_size: {2}".format(train_size, validation_size, test_size))

print("#" * 50, 1)

#################################################################################################

row_cursor = 0 # 현재 보고 있는 행을 알려주는 지시자

X_train_list = []
y_train_regression_list = []
for idx in range(0, train_size):
  sequence_data = bikes_data[idx: idx + sequence_size] 
  # index+ sequence_size 만큼 때서 데이터를 가져온다
  sequence_target = bikes_target[idx + sequence_size - 1]
  X_train_list.append(sequence_data)
  y_train_regression_list.append(sequence_target)
  row_cursor += 1

X_train = torch.stack(X_train_list, dim=0).to(torch.float)
print(X_train.shape)
y_train_regression = torch.tensor(y_train_regression_list, dtype=torch.float32) / y_normalizer

m = X_train.mean(dim=0, keepdim=True)
s = X_train.std(dim=0, keepdim=True)
X_train = (X_train - m) / s

print(X_train.shape, y_train_regression.shape)
# >>> torch.Size([17376, 24, 19]) torch.Size([17376])

print("#" * 50, 2)
#################################################################################################

X_validation_list = []
y_validation_regression_list = []
# rowcursor가 존재하는 위치 부터 validation set을 구성한다 라는 의미
#아까는 0부터 실행 시키면 됐지만 validation set에는 train에 들어갔던
#값이 들어가면 안되기 때문
for idx in range(row_cursor, row_cursor + validation_size):
  sequence_data = bikes_data[idx: idx + sequence_size]
  sequence_target = bikes_target[idx + sequence_size - 1]
  X_validation_list.append(sequence_data)
  y_validation_regression_list.append(sequence_target)
  row_cursor += 1

X_validation = torch.stack(X_validation_list, dim=0).to(torch.float)
y_validation_regression = torch.tensor(y_validation_regression_list, dtype=torch.float32) / y_normalizer

X_validation = (X_validation - m) / s

print(X_validation.shape, y_validation_regression.shape)
# >>> torch.Size([96, 24, 19]) torch.Size([96])

print("#" * 50, 3)
#################################################################################################

X_test_list = []
y_test_regression_list = []
for idx in range(row_cursor, row_cursor + test_size):
  sequence_data = bikes_data[idx: idx + sequence_size]
  sequence_target = bikes_target[idx + sequence_size - 1]
  X_test_list.append(sequence_data)
  y_test_regression_list.append(sequence_target)
  row_cursor += 1

X_test = torch.stack(X_test_list, dim=0).to(torch.float)
y_test_regression = torch.tensor(y_test_regression_list, dtype=torch.float32) / y_normalizer

X_test -= (X_test - m) / s

print(X_test.shape, y_test_regression.shape)
# >>> torch.Size([24, 24, 18]) torch.Size([24])
import os
import numpy as np
import torch

torch.set_printoptions(edgeitems=2, threshold=50, linewidth=75)

bikes_path = os.path.join(os.path.pardir, os.path.pardir, "_00_data", "e_time-series-bike-sharing-dataset", "hour-fixed.csv")

bikes_numpy = np.loadtxt(
  fname=bikes_path, dtype=np.float32, delimiter=",", skiprows=1,
  converters={
    1: lambda x: float(x[8:10])  # 2011-01-07 --> 07 --> 7.0
  }
)
bikes = torch.from_numpy(bikes_numpy)
print(bikes.shape)

daily_bikes = bikes.view(-1, 24, bikes.shape[1])
print(daily_bikes.shape)  # >>> torch.Size([730, 24, 17])

daily_bikes_data = daily_bikes[:, :, :-1]
daily_bikes_target = daily_bikes[:, :, -1].unsqueeze(dim=-1) # 가장 끝의 값을 가져오는 과정

print(daily_bikes_data.shape)
print(daily_bikes_target.shape)

print("#" * 50, 1)

first_day_data = daily_bikes_data[0]
print(first_day_data.shape)

# Whether situation: 1: clear, 2:mist, 3: light rain/snow, 4: heavy rain/snow
print(first_day_data[:, 9].long())
eye_matrix = torch.eye(4)
print(eye_matrix)

weather_onehot = eye_matrix[first_day_data[:, 9].long() - 1] #날씨 column의 값을 one-hot 인코딩 하는 과정
print(weather_onehot.shape)
print(weather_onehot)

first_day_data_torch = torch.cat(tensors=(first_day_data, weather_onehot), dim=1)
 # 기존의 값 뒤에 추가적으로 원핫 인코딩된 정보를 넣는 과정                                                   
print(first_day_data_torch.shape)
print(first_day_data_torch)

print("#" * 50, 2)

day_data_torch_list = []

for daily_idx in range(daily_bikes_data.shape[0]):  # range(730)
  day = daily_bikes_data[daily_idx]  # day.shape: [24, 16]
  weather_onehot = eye_matrix[day[:, 9].long() - 1]
  day_data_torch = torch.cat(tensors=(day, weather_onehot), dim=1)  # day_data_torch.shape: [24, 20]
  day_data_torch_list.append(day_data_torch)

print(len(day_data_torch_list)) # 730개의 데이터를 
daily_bikes_data = torch.stack(day_data_torch_list, dim=0) # 스택이라는 함수를 통해 통으로 만들어 준다. 
print(daily_bikes_data.shape)

print("#" * 50, 3)

print(daily_bikes_data[:, :, :9].shape, daily_bikes_data[:, :, 10:].shape)
daily_bikes_data = torch.cat(
  [daily_bikes_data[:, :, 1:9], daily_bikes_data[:, :, 10:]], dim=2
) # Drop 'instant' and 'whethersit' columns
# 이 코드는 슬라이싱 하는 방식인데 [::,1:9]는 1번째 인덱스부터 9까지 이지만 9번 인덱스의 값은 가져오지 않는다.
# [10:] 부분은 10번째 인덱스 부터는 다 가져온다 라는 의미 이다. 
print(daily_bikes_data.shape)

temperatures = daily_bikes_data[:, :, 8]
daily_bikes_data[:, :, 8] = (daily_bikes_data[:, :, 8] - torch.mean(temperatures)) / torch.std(temperatures)

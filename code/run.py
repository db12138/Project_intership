import torch
import ujson
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
from functools import cmp_to_key

class Config():
    def __init__(self):
        self.epoches = 1000
        self.input_dimension = 16
        self.learning_rate = 0.05
        self.train_dataset_path = "data/original_dataset.json"
        self.saved_path = "code/saved/"
        self.test_dataset_path = "data/original_dataset.json"
        self.test_model = "regression_model_1000.pt"


class RegressionModule(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.input_dimension,1)
        self.relu = nn.ReLU()


    def forward(self, cur_feature):
        out = self.linear1(cur_feature)
        #out = self.relu(out)
        return out


def train(config,train_features,labels):
    # print(train_features.shape)
    # print(labels.shape)
    samples_num  = len(train_features)
    model = RegressionModule(config)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()
    for epoch in  range(config.epoches): 
        epoch_loss = 0
        for index,each_case in enumerate(train_features):
            predict = model(each_case)
            cur_label = labels[index][0]
            cur_loss = loss_function(predict,cur_label)
            epoch_loss += cur_loss

        epoch_loss /= samples_num
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()    
        print(f"epoch:{epoch} cur_epoch_loss:{epoch_loss.item()}")
    torch.save(model.state_dict(),config.saved_path+ f"regression_model_{config.epoches}.pt")

def test(config):
    model  = RegressionModule(config)
    model.load_state_dict(torch.load(config.saved_path + config.test_model))
    test_data_loader = DataLoader(config.test_dataset_path)
    test_features,test_labels,input_scaler,output_scaler ,time_list= test_data_loader.feature_select_norm()
    model.eval()

    predict_result = []
    with torch.no_grad():
        for each_feature in test_features:
            predict = model(each_feature)
            predict_result.append(predict.item())
    predict_result = np.array(predict_result).reshape(-1,1)
    predict_result = output_scaler.inverse_transform(predict_result)
    predict_result = predict_result.reshape(-1)
    print(predict_result.shape)
    result = {}
    assert len(predict_result) == len(time_list)
    for i,time in enumerate(time_list):
        result[time] = predict_result[i]
    print(result)
    return result
        
class DataLoader():
    def __init__(self,dataset_path):
        self.original_dataset = ujson.load(open(dataset_path,'r'))
    def to_tensor(self,matrix):
        return torch.tensor(matrix,dtype=torch.float32)

    def feature_select_norm(self):
        #feature list
        #['发生额', '余额', '到期量', '1Y_percent', '9M_percent', '6M_percent', '3M_percent', '1M_percent', 'avg_rate0', 'avg_rate1', 'avg_rate2', 'avg_rate3', 'avg_rate4', 'avg_rate5', 'avg_rate6', 'avg_rate7', 'turn_over_cnt']
        input_features = ['发生额', '余额', '1Y_percent', '9M_percent', '6M_percent', '3M_percent', '1M_percent', 'avg_rate0', 'avg_rate1', 'avg_rate2', 'avg_rate3', 'avg_rate4', 'avg_rate5', 'avg_rate6', 'avg_rate7', 'turn_over_cnt']
        output_feature = '到期量'

        input_feature_matrix = []
        output_matrix = []
        time_list = []
        for time,feature in self.original_dataset.items():
            if len(feature) == len(input_features)+1:
                time_list.append(time)
                cur_input_feature = []
                for feature_name in input_features:
                    cur_input_feature.append(feature[feature_name])
                
                cur_output_feature = feature[output_feature]

                input_feature_matrix.append(cur_input_feature)
                output_matrix.append(cur_output_feature)
        
        input_feature_matrix = np.array(input_feature_matrix)
        output_matrix = np.array(output_matrix).reshape(-1,1)

        input_scaler = MinMaxScaler()
        norm_input_matrix = input_scaler.fit_transform(input_feature_matrix)
        #origin_data = mm.inverse_transform(mm_data)
        output_scaler = MinMaxScaler()
        norm_output_matrix = output_scaler.fit_transform(output_matrix)
        #output = output_scaler.inverse_transform(norm_output_matrix)
        
        print(time_list)
        return self.to_tensor(norm_input_matrix),self.to_tensor(norm_output_matrix),input_scaler,output_scaler,time_list

def cmp_function(tuple_a,tuple_b):
    time_a = tuple_a[0]
    time_b = tuple_b[0]
    time_a = [int(t) for t in time_a.split("/")]
    time_b = [int(t) for t in time_b.split("/")]
    if time_a[0] < time_b[0]:
        return -1
    elif time_a[0] > time_b[0]:
        return 1
    else:
        if time_a[1] < time_b[1]:
            return -1
        elif time_a[1] > time_b[1]:
            return 1
        else:
            return 0

def run():
    config = Config()
    #loading data
    data_loader = DataLoader(config.train_dataset_path)
    input_normalized,output_normalized,input_scaler,output_scaler,_ = data_loader.feature_select_norm()
    #training 
    #train(config,input_normalized,output_normalized)

    #testing
    result = test(config)
    result = sorted(result.items(),key=cmp_to_key(cmp_function),reverse=True)
    # result = [(k,v) for k,v in result.items()]
    # result = sorted(result,key=lambda x:x[0],reverse=True)
    for item in result:
        print(item[1])

if __name__ == "__main__":
    run()
        


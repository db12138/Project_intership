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
        self.epoches = 3
        self.input_dimension = 16
        self.learning_rate = 0.05
        self.dataset_path = "data/original_dataset.json"
        self.saved_path = "code/saved/"
        self.test_dataset_path = "data/original_dataset.json"
        self.test_model = "regression_model_1000.pt"


class RegressionModule(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.linear1 = nn.Linear(config.input_dimension,10)
        self.linear2 = nn.Linear(10,1)
        self.relu = nn.ReLU()


    def forward(self, cur_feature):
        out = self.linear1(cur_feature)
        out = self.linear2(out)
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
    #torch.save(model.state_dict(),config.saved_path+ f"regression_model_{config.epoches}.pt")
    return model,loss_function

def load_and_test(config):
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


def test(model,test_features,test_output,output_scaler,loss_function):
    model.eval()
    predict_result = []
    avg_loss = 0
    with torch.no_grad():
        for index,each_feature in enumerate(test_features):
            predict = model(each_feature)
            cur_loss = loss_function(predict,test_output[index])
            avg_loss += cur_loss
            predict_result.append(predict.item())
    avg_loss /= len(test_features)
    predict_result = np.array(predict_result).reshape(-1,1)
    predict_result = output_scaler.inverse_transform(predict_result)
    predict_result = predict_result.reshape(-1)
    # print(predict_result.shape)
    print(f"avg_loss:{avg_loss},predict_result:{predict_result}")
    return predict_result,avg_loss


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
    data_loader = DataLoader(config.dataset_path)
    input_normalized,output_normalized,input_scaler,output_scaler,time_list = data_loader.feature_select_norm()
    print(type(input_normalized),input_normalized.shape)
    
    start_split_index = 0
    step = 2
    result = [-1] * len(input_normalized)
    while(start_split_index < len(input_normalized)):
        test_data_input = input_normalized[start_split_index:start_split_index+step] 
        test_data_output = output_normalized[start_split_index:start_split_index+step]

        input_part_A = input_normalized[0:start_split_index]
        input_part_B = input_normalized[(start_split_index+step):] 
        train_data_input =  torch.cat((input_part_A,input_part_B),0) 
        output_part_A = output_normalized[0:start_split_index]
        output_part_B = output_normalized[(start_split_index+step):] 
        train_data_output =  torch.cat((output_part_A,output_part_B),0)
        #training 
        cur_model,loss_function = train(config,train_data_input,train_data_output)
        #testimg
        predict_result,avg_loss = test(cur_model,test_data_input,test_data_output,output_scaler,loss_function)
        result[start_split_index:start_split_index+step] = predict_result
        start_split_index += step
    
    time_and_result = {}
    for index,time in enumerate(time_list):
        time_and_result[time] = result[index]
    print(time_and_result)
    time_and_result = sorted(time_and_result.items(),key=cmp_to_key(cmp_function),reverse=True)
    for item in time_and_result:
        print(item[1])
    
def init_fun():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
if __name__ == "__main__":
    init_fun()
    run()
    # result = [16024.954476124969, 16955.134009669437, 17234.54537133819, 13381.395616713966, 17122.403418243208, 16737.097674109365, 13855.079297325965, 14331.128151925464, 13785.590188245136, 14813.118046075368, 14757.889782223094, 14109.891575261567, 15088.905403351182, 15025.539695035717, 17236.740501987828, 14685.84690820886, 16857.007818524206, 15364.927225046229, 14220.876072663774, 15836.107261741978, 14799.533834578408, 14220.248002516599, 15370.27176785857, 14088.08086246585]
    # for t in result:
    #     print(t)
    
        


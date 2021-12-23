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
        self.epoches = 500
        self.learning_rate = 30
        self.dataset_path = "data/original_dataset.json"
        self.saved_path = "code/saved/"
        self.test_dataset_path = "data/original_dataset.json"
        self.test_model = "term_structure_model_1000.pt"

class TermStructureModel(nn.Module):
    def __init__(self,input_features) -> None:
        super().__init__()
        # self.relu = nn.ReLU()
        self.input_features = input_features #从2018年 7 月开始可以预测 因为需要 2017年的发生量 2017年7月发生量最大
        self.date2index = {}
        index_cnt = 0
        for year in ["2017","2018","2019","2020","2021"]:
            for month in range(1,13):
                if year == "2017" and month <7:
                    continue
                if year == "2021" and month >10:
                    continue
                cur_date = year + "/" + str(month)
                self.date2index[cur_date] = index_cnt
                index_cnt += 1
        self.term_matrix = nn.Parameter(torch.randn(52,5)) # 2017/7 - 2021/10 (6 + 12+ 12 + 12 + 10 = 52 ) * 5 (1Y,9M,6M,3M,1M)
        self.sigmoid = nn.Sigmoid()
    def percentage_function(self,matrix):
        term_matrix = self.sigmoid(matrix)
        term_sum = torch.sum(term_matrix,1).unsqueeze(1)
        term_matrix  = term_matrix / term_sum
        return term_matrix

    def forward(self,cur_date):
        cur_date_index = self.date2index[cur_date]
        term_matrix = self.percentage_function(self.term_matrix)

        weight_1Y = term_matrix[cur_date_index -12][0]
        fasheng_1Y = self.input_features[cur_date_index -12][0]

        weight_9M = term_matrix[cur_date_index -9][1]
        fasheng_9M = self.input_features[cur_date_index -9][0]

        weight_6M = term_matrix[cur_date_index -6][2]
        fasheng_6M = self.input_features[cur_date_index -6][0]

        weight_3M = term_matrix[cur_date_index -3][3]
        fasheng_3m = self.input_features[cur_date_index -3][0]

        weight_1M = term_matrix[cur_date_index -1][4]
        fasheng_1M = self.input_features[cur_date_index -1][0]

        predict = weight_1Y * fasheng_1Y + weight_9M * fasheng_9M + weight_6M * fasheng_6M + weight_3M * fasheng_3m +weight_1M * fasheng_1M

        return predict


def train(config,train_features,labels,time_list):
    # print(train_features.shape)
    # print(labels.shape)
    samples_num  = len(train_features)
    model = TermStructureModel(train_features)
    loss_function = nn.MSELoss()

    # for name, p in model.named_parameters():
    #     print(name, p.requires_grad)
    # assert 0

    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    model.train()
    print(model.percentage_function(model.term_matrix))

    for epoch in  range(config.epoches): 
        epoch_loss = 0
        for index,each_case in enumerate(time_list):
            predict = model(each_case)
            cur_label = labels[index][0]
            cur_loss = loss_function(predict,cur_label)
            epoch_loss += cur_loss

        epoch_loss /= samples_num
        print(f"epoch:{epoch} cur_epoch_loss:{epoch_loss.item()}")
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()    
    #torch.save(model.state_dict(),config.saved_path+ f"regression_model_{config.epoches}.pt")

    model.eval()
    print("after trainning : ------------------")
    print(model.percentage_function(model.term_matrix))
    return model,loss_function

class DataLoader():
    def __init__(self,dataset_path):
        self.original_dataset = ujson.load(open(dataset_path,'r'))
    def to_tensor(self,matrix):
        return torch.tensor(matrix,dtype=torch.float32)

    def feature_select_norm(self):
        #feature list
        #['发生额', '余额', '到期量', '1Y_percent', '9M_percent', '6M_percent', '3M_percent', '1M_percent', 'avg_rate0', 'avg_rate1', 'avg_rate2', 'avg_rate3', 'avg_rate4', 'avg_rate5', 'avg_rate6', 'avg_rate7', 'turn_over_cnt']
        input_features = ['发生额', '余额']
        output_feature = '到期量'
        input_feature_matrix = []
        output_matrix = []
        time_list = []
        for time,feature in self.original_dataset.items():
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
        #return self.to_tensor(norm_input_matrix),self.to_tensor(norm_output_matrix),input_scaler,output_scaler,time_list
        return self.to_tensor(norm_input_matrix),self.to_tensor(norm_output_matrix),input_scaler,output_scaler,time_list

def run():
    config = Config()
    #loading data
    data_loader = DataLoader(config.dataset_path)
    input_normalized,output_normalized,input_scaler,output_scaler,time_list = data_loader.feature_select_norm()
    print(type(input_normalized),input_normalized.shape)
    
    #train 
    train(config,input_normalized,output_normalized,time_list)
    #test
    
    
    
def init_fun():
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
if __name__ == "__main__":
    init_fun()
    run()

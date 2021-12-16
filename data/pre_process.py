import pandas as pd
import numpy as np
import ujson
root_path = ""
#"C:/Users/10266/Desktop/中信证券/票据到期量课题/"
#10月票据承兑到期量 = 9月电票承兑余额+10月票据承兑发生量-10月电票承兑余额
def calculate_expire():
    data = pd.read_excel(root_path + "data/票据201706_202110.xlsx",header=0)
    #发生额
    month = data.iloc[:,0]
    fasheng = data.iloc[:,1]
    yue = data.iloc[:,2]
    print(fasheng)
    print(month)
    daoqi = [0]* len(month)
    for i in range(len(month)):
        if i!= 0:
            daoqi[i] = yue[i-1] + fasheng[i] - yue[i]

    data['到期量'] = daoqi 
    print(data)
    print(data.columns.values)
    data.to_excel(root_path + "data/票据_data.xlsx",index=False)
def trading_data_process():
    data = pd.read_excel(root_path + "data/票据_data.xlsx",header=0,sheet_name="前台交易数据")
    print(data.columns.values)
    print(len(data))
    month_group = {}
    # dateTime	billAgent	billType	limitDayType	rate0 最新利率	rate1 加权平均利率	rate2 最高利率	rate3 最低利率	rate4 开盘利率	rate5 收盘利率	rate6 前收盘利率	rate7 前加权平均利率	turnOver 成交量（亿）
    #each row keys

    for index,row in data.iterrows():
        # print(row.keys())
        # assert 0
        year = row['dateTime'].year
        month = row['dateTime'].month
        cur_group_key = f"{year}/{month}"
        if cur_group_key not in month_group:
            month_group[cur_group_key] = [row]
        else:
            month_group[cur_group_key].append(row)
    
    #one_month_feature.keys() = #(limit_day_1Y , limit_day_9M ,limit_day_6M , limit_day_3M , limit_day_1M, avg_rate0 , avg_rate1 ,avg_rate2,avg_rate3, avg_rate4, avg_rage5, avg_rate6,avg_rate7 ,turn_over)
    month_features = {}
    for key,values in month_group.items():
        row_num_in_cur_month = len(values)

        cur_month_features = {}
        limit_day_1Y_cnt = 0
        limit_day_9M_cnt = 0
        limit_day_6M_cnt = 0
        limit_day_3M_cnt = 0
        limit_day_1M_cnt = 0
        avg_rate0_cnt = 0
        avg_rate1_cnt = 0
        avg_rate2_cnt = 0
        avg_rate3_cnt = 0
        avg_rate4_cnt = 0
        avg_rate5_cnt = 0
        avg_rate6_cnt = 0
        avg_rate7_cnt = 0
        turn_over_cnt = 0
        for cur_row in values:
            if cur_row["limitDayType"] == "1Y":
                limit_day_1Y_cnt += 1
            elif cur_row["limitDayType"] == "9M":
                limit_day_9M_cnt += 1
            elif cur_row["limitDayType"] == "6M":
                limit_day_6M_cnt += 1
            elif cur_row["limitDayType"] == "3M":
                limit_day_3M_cnt += 1
            elif cur_row["limitDayType"] == "1M":
                limit_day_1M_cnt += 1
            else:
                raise "No limitDayType error"

            avg_rate0_cnt += cur_row['rate0 最新利率']
            avg_rate1_cnt += cur_row['rate1 加权平均利率']
            avg_rate2_cnt += cur_row['rate2 最高利率']
            avg_rate3_cnt += cur_row['rate3 最低利率']
            avg_rate4_cnt += cur_row['rate4 开盘利率']
            avg_rate5_cnt += cur_row['rate5 收盘利率']
            avg_rate6_cnt += cur_row['rate6 前收盘利率']
            avg_rate7_cnt += cur_row['rate7 前加权平均利率']
            turn_over_cnt += cur_row['turnOver 成交量（亿）']
        
        limit_day_1Y_cnt /= row_num_in_cur_month
        limit_day_9M_cnt /= row_num_in_cur_month
        limit_day_6M_cnt /= row_num_in_cur_month
        limit_day_3M_cnt /= row_num_in_cur_month
        limit_day_1M_cnt /= row_num_in_cur_month
        avg_rate0_cnt /= row_num_in_cur_month
        avg_rate1_cnt /= row_num_in_cur_month
        avg_rate2_cnt /= row_num_in_cur_month
        avg_rate3_cnt /= row_num_in_cur_month
        avg_rate4_cnt /= row_num_in_cur_month
        avg_rate5_cnt /= row_num_in_cur_month
        avg_rate6_cnt /= row_num_in_cur_month
        avg_rate7_cnt /= row_num_in_cur_month
        turn_over_cnt /= row_num_in_cur_month
        cur_month_features["1Y_percent"] = limit_day_1Y_cnt
        cur_month_features["9M_percent"] = limit_day_9M_cnt
        cur_month_features["6M_percent"] = limit_day_6M_cnt
        cur_month_features["3M_percent"] = limit_day_3M_cnt
        cur_month_features["1M_percent"] = limit_day_1M_cnt
        cur_month_features["avg_rate0"] = avg_rate0_cnt
        cur_month_features["avg_rate1"] = avg_rate1_cnt
        cur_month_features["avg_rate2"] = avg_rate2_cnt
        cur_month_features["avg_rate3"] = avg_rate3_cnt
        cur_month_features["avg_rate4"] = avg_rate4_cnt
        cur_month_features["avg_rate5"] = avg_rate5_cnt
        cur_month_features["avg_rate6"] = avg_rate6_cnt
        cur_month_features["avg_rate7"] = avg_rate7_cnt
        cur_month_features["turn_over_cnt"] = turn_over_cnt
        # dateTime	billAgent	billType	limitDayType	rate0 最新利率	rate1 加权平均利率	rate2 最高利率	rate3 最低利率	rate4 开盘利率	rate5 收盘利率	rate6 前收盘利率	rate7 前加权平均利率	turnOver 成交量（亿）
        month_features[key] = cur_month_features
    return month_features

def wind_data_process():
    wind_data = pd.read_excel(root_path + "data/票据_data.xlsx",header=0,sheet_name="发生额_余额_到期量")

    wind_data_month = {}
    for index,row in wind_data.iterrows():
        year = row['时间'].year
        month = row['时间'].month
        fasheng_value = row["票据承兑发生额:电子银票"]
        yue_value = row["票据承兑余额:电子银票"]
        daoqi_value = row["到期量"]
        cur_group_key = f"{year}/{month}"
        wind_data_month[cur_group_key] = {"发生额":fasheng_value,"余额":yue_value,"到期量":daoqi_value}

    return wind_data_month
    #print(wind_data_month)
def merge_trading_wind_features(wind_data,trading_data):
    for time,feature in wind_data.items():
        if time in trading_data:
            wind_data[time].update(trading_data[time])
    #print(wind_data)
    return wind_data
if __name__ == "__main__":
    #calculate_expire()
    trading_features = trading_data_process()
    print(trading_features["2019/11"])
    wind_data_features = wind_data_process()
    merged_features = merge_trading_wind_features(wind_data_features , trading_features)
    
    print(merged_features["2021/9"].keys())
    ujson.dump(merged_features,open("data/original_dataset.json",'w'),ensure_ascii=False)


# col1 = sheet.iloc[:,0:2]
# studentNoList = col1[3:]

# print(studentNoList)
# print(studentNoList.shape)
# print(len(studentNoList))

# mod0 = pd.DataFrame(columns=["学号","姓名"])
# mod1 = pd.DataFrame(columns=["学号","姓名"])
# mypartStudentList = pd.DataFrame(columns=["学号","姓名"])
# for i in range(len(studentNoList)):
#     cur = studentNoList.iloc[i,0]
#     name = studentNoList.iloc[i,1]
#     suffix = cur % 100
#     if suffix % 3 == 0:
#         mod0 = mod0.append({"学号":cur,"姓名":name},ignore_index=True)
#     elif suffix % 3 == 1:
#         mod1 = mod1.append({"学号":cur,"姓名":name},ignore_index=True)
#     elif suffix % 3 == 2:
#         mypartStudentList = mypartStudentList.append({"学号":cur,"姓名":name},ignore_index=True)
#     else:
#         print("error")

# #print(mypartStudentList)
# print(len(studentNoList))
# print(len(mod0))
# print(len(mod1))
# print(len(mypartStudentList))

# mypartStudentList.to_excel("cuiz_studentNo.xlsx")
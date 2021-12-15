import pandas as pd
import numpy as np
root_path = "C:/Users/10266/Desktop/中信证券/票据到期量课题/"
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

# def trading_data():
#     data = pd.read_excel("前台交易数据.xlsx")
#     print(len(data))
#     cnt = 0
#     for index,row in data.iterrows():
#         if row['billType'] == "银票":
#             cnt += 1
#     print(cnt)
#     print(data.columns.values)


if __name__ == "__main__":
    calculate_expire()
    #trading_data()

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
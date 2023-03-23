import numpy as np
from Control.ImportData import *

def deltuple(timeList):
    del_tuple = ()
    for num in range(0,len(timeList)):
        del_tuple += tuple([sum(timeList[:num])+num])
    return del_tuple
def insertTuple(timeList):
    insert_tuple = ()
    for num in range(0,len(timeList)):
        insert_tuple += tuple([sum(timeList[:num+1])])
    return insert_tuple

def forsTuple(fors,forcasting):
    fors_tuple = ()
    for i in range(len(fors)-1):
        fors_tuple += tuple([fors[i]])
    fors_tuple += tuple([forcasting])
    return fors_tuple

def msf1D(Fors, X_test,Y_test,forcasting,timeList, endTimeList):
        "多步階預報 t+2開始"
        from keras import backend as K
        # timeList = [3,3,3,3,3,6,3] #(0,4,8,12,16,23,27)
        #3,3,3,6,6,12,12 # 1:維度(由外到內) 移除3維第一行
        newList = np.sum([timeList , endTimeList],axis = 0).tolist()
        new_x = np.delete(X_test, deltuple(newList))    ##要改0,+1,+3,+3,+12 1,3,3,12,6
        
        if Y_test != "":
            new_y = np.delete(Y_test, 0)       # 刪掉第一個
        else: 
             new_y = ""
        ## 新增最後一行
        New_x = np.insert(new_x,  insertTuple(newList), (forsTuple(Fors,forcasting)))
        New_x = K.cast_to_floatx(New_x)
        New_x = np.reshape(New_x,(1,1,len(New_x)))
        #Add new ROW
        # df = pd.DataFrame()
        # df["X_test"] = X_test.flatten()
        # df["new_x"] = New_x.flatten()
        # df["Fors"] = pd.Series(forsTuple(Fors,forcasting))
        # DF2CSV(df.T, "msf")
        return New_x, new_y

def msf(Fors, X_test,Y_test,forcasting):
        "多步階預報 t+2開始"
        from keras import backend as K
        new_x = np.delete(X_test, 0, 1)     # 1:維度(由外到內) 移除2維第一行
        if Y_test != "":
            new_y = np.delete(Y_test, 0)       # 刪掉第一個
        else: 
            new_y = ""
        ## 新增最後一行
        f = forcasting
        
        # ForsList = Fors[TStep+time-1:,:-1].tolist() 
        # New_x = []
        # for i in range(len(f)):
        #     j = i 
        #     Fors[-1] = f[i]
        #     other_forcast = [Fors[j]]
        #     add=np.append(new_x[i],other_forcast,axis=0)
        #     New_x.append(add) #3維
        # New_x = K.cast_to_floatx(New_x)
        Fors[-1] = f[0]

        new_x = np.append(new_x, [[Fors]] ,axis=1)
        return new_x, new_y

import time
import threading
import queue
class workSame(threading.Thread):
    def __init__(self, queue, num):
        threading.Thread.__init__(self)
        self.queue = queue
        self.num = num

    def run(self):
         while self.queue.qsize() > 0:
            msg = self.queue.get()

            # 執行鬍鬚圖
# 建立佇列
my_queue = queue.Queue()

# 放入佇列
for i in range(10):
    my_queue.put("para")

worker1 = workSame(my_queue,1)
worker2 = workSame(my_queue,2)

# 讓 Worker 開始處理資料
worker1.start()
worker2.start()

# 等待所有 Worker 結束
worker1.join()
worker2.join()

# if __name__ == "__main__":
#     timeList = [1,3,3,12,6]
#     endTimeList=[-1,-1,-1,-1,0]
#     del_tuple = ()
#     for num in range(0,len(timeList)):
#         del_tuple += tuple([sum(timeList[:num])+num])

# def msf(Fors, X_test,Y_test,forcasting,time,TStep):
#         "多步階預報 t+2開始"
#         from keras import backend as K
#         new_x = np.delete(X_test, 0, 1)     # 1:維度(由外到內) 移除3維第一行
#         new_y = np.delete(Y_test, 0)       # 刪掉第一個
#         ## 新增最後一行
#         f = forcasting
        
#         #預報值 時間跟test一樣 #要正規化
#         # Shihimen = Fors[TStep+time-1:,0]
#         # Feitsui = Fors[TStep+time-1:,1]
#         # TPBRain = Fors[TStep+time-1:,2]
#         # SMOutflow = Fors[TStep+time-1:,3]
#         # FTOutflow = Fors[TStep+time-1:,4]
#         # Tide = Fors[TStep+time-1:,5]
#         ForsList = Fors[TStep+time-1:,:-1].tolist() 
#         New_x = []
#         for i in range(len(f)-1):
#             j = i 
#             ForsList[j].append(f[i])
#             other_forcast = [ForsList[j]]
#             # other_forcast = [[Shihimen[j],Feitsui[j],TPBRain[j],SMOutflow[j],FTOutflow[j],f[i]]] # 各個因子預報值
#             # other_forcast.append(f[j])  # 加入水位預測值
#             add=np.append(new_x[i],other_forcast,axis=0)
#             New_x.append(add) #3維
#         New_x = K.cast_to_floatx(New_x)
#         return New_x, new_y
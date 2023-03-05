import numpy as np

def deltuple(timeList):
    del_tuple = ()
    for num in range(0,len(timeList)):
        del_tuple += tuple([sum(timeList[1:num+1])+num])
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

def msf1D(Fors, X_test,Y_test,forcasting,timeList):
        "多步階預報 t+2開始"
        from keras import backend as K
        # timeList = [3,3,3,3,3,6,3] #(0,4,8,12,16,23,27)
        #3,3,3,6,6,12,12 # 1:維度(由外到內) 移除3維第一行
        new_x = np.delete(X_test, deltuple(timeList))    
        if Y_test != "":
            new_y = np.delete(Y_test, 0)       # 刪掉第一個
        else: 
             new_y = ""
        ## 新增最後一行
        New_x = np.insert(new_x,  insertTuple(timeList), (forsTuple(Fors,forcasting)))
        New_x = K.cast_to_floatx(New_x)
        New_x = np.reshape(New_x,(1,1,len(New_x)))
        return New_x, new_y

def msf(Fors, X_test,Y_test,forcasting,time,TStep):
        "多步階預報 t+2開始"
        from keras import backend as K
        new_x = np.delete(X_test, 0, 1)     # 1:維度(由外到內) 移除3維第一行
        if Y_test != "":
            new_y = np.delete(Y_test, 0)       # 刪掉第一個
        else: 
             new_y = ""
        ## 新增最後一行
        f = forcasting
        
        ForsList = Fors[TStep+time-1:,:-1].tolist() 
        New_x = []
        for i in range(len(f)):
            j = i 
            ForsList[j].append(f[i])
            other_forcast = [ForsList[j]]
            add=np.append(new_x[i],other_forcast,axis=0)
            New_x.append(add) #3維
        New_x = K.cast_to_floatx(New_x)
        return New_x, new_y

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
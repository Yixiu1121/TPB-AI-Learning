from Model.Rain import *


def DataList(data,paraNum,datalist):
    datalist.TypePara = [data[i][paraNum] for i in range(len(data))]
    return datalist  
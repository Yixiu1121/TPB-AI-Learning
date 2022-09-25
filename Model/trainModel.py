
from abc import ABC, abstractmethod


class DataSet(ABC):
    TrainingData = "訓練集"
    VaildationData = "驗證集"
    TestData = "測試集"

class Train(ABC):
    @abstractmethod
    def DataProcessing(self):
        "資料前處理"
    def __Normalization(self):
        "正規化"
    def __FeatureScaling(self):
        "資料格式"
    def __Reshape(self):
        "維度修正"

    @abstractmethod
    def ModelSetting(self):
        "建立模型"
    def __LayerSetting(self):
        "層數"
    def __CompilingModel(self):
        "組成模型"
    def __FittingModel(self):
        "訓練模型"
    def __Prediction(self):
        "預測"
    
    
    @abstractmethod
    def GridSearching(self):
        "網格搜尋法"
    @abstractmethod
    def PlotResult(self):
        "視覺化結果"
class TPB(DataSet):
    ""    


class L(Train):
    def ModelSetting(self):
        return super().ModelSetting()
    def __CompilingModel(self):
        return super().__CompilingModel()

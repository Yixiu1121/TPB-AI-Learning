class TPB():#報表
    Date:str 
    ShihmenOutflow = 'not found'     #石門出流 ok
    ShihmenInflow = 'not found'     #石門入流ok
    ShihmenRainfall:int             #石門雨量
    FeitsuiOutflow = 'not found'     #翡翠出流 ok
    FeitsuiRainfall:int             #翡翠雨量
    Tide = 'not found'             #淡水潮位 ok
    Rainfall:int                    #雨量
    WaterLevel = 'not found'     #水位ok

class Reservoir():
    Date:str
    Name:str    
    Outflow:int
    Inflow:int
    Rainfall:int
    
class Rainfall():
    Date:str
    Rainfall:int

class WaterLevel():
    Date:str
    WaterLevel:int   

class Tide():
    Date:str
    Tide:int

class Case():
    Casenum:int
    StrTime:str
    EndTime:str
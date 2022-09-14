from Model.Rain import *

def CalcTide(tide:Tide,result:dict):
    if(tide.Date not in result.keys):
        result[tide.Date]=TPB()
        result[tide.Date].Date=tide.Date
    t=result[tide.Date]
    t.Tide=tide.Tide


def CalcRainfall(rain:Rainfall,result:dict):
    if(rain.Date not in result.keys):
        result[rain.Date]=TPB()
        result[rain.Date].Date=rain.Date
    t=result[rain.Date]
    t.Rainfall+=rain.Rainfall

def CalcReservoir(reservoir:Reservoir,result:dict):
    if(reservoir.Date not in result.keys):
        result[reservoir.Date]=TPB()
        result[reservoir.Date].Date=reservoir.Date
    t=result[reservoir.Date]
    if(reservoir.Name=="石門"):
        pass
    elif(reservoir.Name=="翡翠"):
        pass

    # t.Inflow+=reservoir.Inflow
    # t.Outflow+=reservoir.Outflow

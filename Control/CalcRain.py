from Model.TPBdata import *


def CalcTide(tide:Tide,result:dict):
    if(tide.Date not in result.keys()):
        result[tide.Date]=TPB()         ## new object--result[tide.Date]--have TPB property 
        result[tide.Date].Date=tide.Date
    t=result[tide.Date]
    t.Tide = tide.Tide


def CalcRainfall(rain:Rainfall,result:dict):
    if(rain.Date not in result.keys()):
        result[rain.Date]=TPB()
        result[rain.Date].Date=rain.Date
    t=result[rain.Date]
    t.Rainfall = rain.Rainfall

def CalcWaterLevel(wl:WaterLevel,result:dict):
    if(wl.Date not in result.keys()):
        result[wl.Date]=TPB()
        result[wl.Date].Date=wl.Date
    t=result[wl.Date]
    t.WaterLevel = wl.WaterLevel

def CalcReservoir(reservoir:Reservoir,result:dict):
    if(reservoir.Date not in result.keys()):
        result[reservoir.Date]=TPB()
        result[reservoir.Date].Date=reservoir.Date
    t=result[reservoir.Date]
    if(reservoir.Name=="石門"):
        t.ShihmenOutflow = reservoir.Outflow
        t.ShihmenInflow = reservoir.Inflow
        # t.ShihmenRainfall = reservoir.Rainfall
    elif(reservoir.Name=="翡翠"):
        t.FeitsuiOutflow = reservoir.Outflow
        # t.FeitsuiRainfall = reservoir.Rainfall

    # t.Inflow+=reservoir.Inflow
    # t.Outflow+=reservoir.Outflow

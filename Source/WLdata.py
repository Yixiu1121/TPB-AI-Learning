# from Control.ImportData import DF2CSV
import pandas as pd
##台北橋水位

def TimetoInt(date:str):
    "2022/9/15 3:50 PM convert to 202101080600"
    x = date.split(' ')
    YMD, hour = x[0],x[1]
    hr, min = hour.split(':')
    YMD = YMD.split('/')
    Y,M,D = YMD[0], YMD[1], YMD[2]
    if int(M)<10:
        M = "0"+M
    if int(D)<10:
        D = "0"+D
    if int(hr)<10:
        hr = "0"+hr
    date = int(Y+M+D+hr)
    return date

def Time24(time):
    "2021/6/1 3:00 PM" 
    t = time.split(' ')

    if int(t[1].split(':')[0])==12 and t[2]=='PM':
        t24 = t[0]+' '+t[1]
    elif int(t[1].split(':')[0])==12 and t[2]=='AM':
        t24 = t[0]+' '+'00:00'
    elif t[2]=='PM':
        hour = int(t[1].split(':')[0])+12
        t24 = t[0]+' '+str(hour)+':00'    
    else:
        hour = t[1]
        if int(t[1].split(':')[0])<10:
            hour = '0'+t[1]
        t24 = t[0]+' '+hour
    return t24

if __name__ == "__main__":
    wl = pd.read_csv('TPB-AI-Learning\Source\WST.csv',usecols=['STA_NO','INFO_DATE','MM','H'])
    print(TimetoInt(wl.iloc[1,1]))
    P01 = []
    P04 = []
    for i in range(1,len(wl)):   
        if int(wl.iloc[i,2])==0:
            if TimetoInt(wl.iloc[i,1]) > 2021060100:
                if str(wl.iloc[i,0])=='P01':
                    P01.append(wl.iloc[i,1:5])
                elif str(wl.iloc[i,0])=='P04':
                    P04.append(wl.iloc[i,1:5])
            if TimetoInt(wl.iloc[i,1]) < 2020123123:
                break
    print('DONE')
    # pd.DataFrame(P01).to_csv("P01(1).csv",index=None)
    pd.DataFrame(P04).to_csv("P04ALL.csv",index=None)

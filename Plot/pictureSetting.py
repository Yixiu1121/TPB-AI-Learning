import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import pandas as pd
from matplotlib.ticker import MultipleLocator
import datetime
import matplotlib.dates as md
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import HOURLY
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_squared_log_error,r2_score
Path ='C:/Users/309/Documents/GitHub/TPB Code/Seq2Seq/best model'
SP = '/0525-364(3)/364(3)'
BP = '/0525-B/364(5)'
LP = '/0525-L/3128(5)'
MP = '/0530-M/(1)'
# Dujan0, Megi121, Lekima430  ,Mitag584, 
Dujan = {'name':'Dujan', 'start_time':'2015/9/27 16:00:00', 'H':-1.05, 'Max':3.78, 'num':0}
Megi = {'name':'Megi', 'start_time':'2016/9/26  13:00:00', 'H':-0.84, 'Max':3.35, 'num':121}
Lekima = {'name':'Lekima', 'start_time':'2019/8/7  23:00:00', 'H':-0.59, 'Max':1.81, 'num':430}
Mitag = {'name':'Mitag', 'start_time':'2019/9/29  17:00:00', 'H':1.61, 'Max':2.6, 'num':584}
Matmo = {'name':'Matmo', 'start_time':'2014/7/24  03:00:00', 'H':-0.13, 'Max':1.8, 'num':50}
Soudelor = {'name':'Soudelor', 'start_time':'2015/8/6  15:00:00', 'H':1.44, 'Max':5.17, 'num':440}
Malakas = {'name':'Malakas', 'start_time':'2016/9/17  15:00:00', 'H':0.22, 'Max':2.63, 'num':1084-33}
Meranti = {'name':'Meranti', 'start_time':'2016/9/14  18:00:00', 'H':-0.61, 'Max':2.13, 'num':1084-33-77-1}

def onestepIndex():
    path = Path
    Sd = pd.read_csv(f'{path}{SP}(train)index.csv')
    Bd= pd.read_csv(f'{path}{BP}(train)index.csv')
    Ld= pd.read_csv(f'{path}{LP}(train)index.csv')
    Md= pd.read_csv(f'{path}{MP}(train)index.csv')
    model = []
    model.append(Sd[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Bd[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Ld[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Md[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model = pd.DataFrame(model) 
    model['model'] = ['Seq2Seq','BiLSTM','LSTM','SVM']
    model.to_csv(f"{path}/train(IndexAll)_modify.csv",index=False)

    Sd = pd.read_csv(f'{path}{SP}index.csv')
    Bd= pd.read_csv(f'{path}{BP}index.csv')
    Ld= pd.read_csv(f'{path}{LP}index.csv')
    Md= pd.read_csv(f'{path}{MP}index.csv')
    model = []
    model.append(Sd[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Bd[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Ld[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model.append(Md[["RMSE", "MAE", "CE", "CC"]].loc[0])
    model = pd.DataFrame(model) 
    model['model'] = ['Seq2Seq','BiLSTM','LSTM','SVM']
    model.to_csv(f"{path}/IndexAll.csv",index=False)

def scatter(modelName, p):
    mpl.rc('font',family = 'Times New Roman')
    path = Path
    data = pd.read_csv(f'{path}{p}(train)index.csv')
    obs = data.Observation
    forecast = data.Forcast
    residual = obs-forecast
    fig1 = plt.figure(1)
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    plt.title(f'{modelName}')
    plt.plot(obs, obs, label = "Decision Boundary",color='black')
    plt.scatter(obs, forecast, label = "Data",color='orange')

    # plt.grid()
    plt.legend()
    frame2 = fig1.add_axes((.1,.1,.8,.2))        
    plt.scatter(obs, residual, label = "Residual")
    # plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(f'C:/Users/309/Documents/GitHub/TPB Code/出圖/scatter/{modelName}(train).png',bbox_inches='tight')
    return plt.close()

def plotOne(eventName,start_time,start=0):
    len = 77
    path = Path
    Sdata = pd.read_csv(f'{path}{SP}(train)index.csv')

    # Mdata = pd.read_csv(f'{path}/0525-364(3)/364(3){eventName}.csv',header=None) ## 注意OBS有沒有對齊
    # MdataT = Mdata.T
    Mdata = pd.read_csv(f'{path}{MP}(train)index.csv')
    Ldata = pd.read_csv(f'{path}{LP}(train)index.csv')
    Bdata = pd.read_csv(f'{path}{BP}(train)index.csv')
    
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rc('font',family = 'Times New Roman')

    plt.rcParams['mathtext.fontset'] = 'cm'
    # start_time = '20201001 10:00:00'
    t = pd.date_range(start=start_time, periods= len, freq="1H")
    
    mm = 1/25.4 # inch 換 毫米
    fig,ax = plt.subplots(figsize=(91*mm,72*mm)) # 刚才构思的图片长、宽
    locator = AutoDateLocator(minticks=10)
    locator.intervald[HOURLY] = [8]  # 10min 为间隔
    ax.xaxis.set_major_locator(locator=locator)

    mlocator = AutoDateLocator(minticks=10) # 值越小可能只能按小时间隔显示
    mlocator.intervald[HOURLY] = [4]  # 10min 为间隔
    ax.xaxis.set_minor_locator(locator=mlocator)

    ymajorLocator = MultipleLocator(1)
    yminorLocator = MultipleLocator(.5/2) #将此y轴次刻度标签设置为0.1的倍数
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)

    ax.plot(t,Sdata.Observation[start:start+len],color="black", label="Observation", linestyle=":") #, linestyle=":"
    ax.plot(t,Mdata.Forcast[start:start+len],color="#1e488f", label="SVM",linewidth=1) #, linestyle=":"
    ax.plot(t,Ldata.Forcast[start:start+len],color="#10a674", label="LSTM",linewidth=1)
    ax.plot(t,Bdata.Forcast[start:start+len],color="#ff9408", label="BiLSTM",linewidth=1)
    ax.plot(t,Sdata.Forcast[start:start+len],color="darkred", label="Seq2Seq",linewidth=1)
    ax.axhline(2.2,color="#696969", label="Alert Level 3\n(EL. 2.2)",linewidth=0.5) #, linestyle="--"
    ax.set_ylim([-2,7])
    ax.set_xlabel('Time (h)',fontsize=9)
    ax.set_ylabel('Water Level (m)',fontsize=9)
    ax.set_title(f"Typhoon {eventName}",fontsize=9)
    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False) #邊框
    ax.spines['top'].set_visible(False)

    ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
    plt.legend(loc='best',ncol=3,fontsize=7,facecolor= None , framealpha=0,edgecolor=None)
    # plt.show()
    # 根据自己定义的方式去画时间刻度
    formatter = plt.FuncFormatter(time_ticks)
    # 在图中应用自定义的时间刻度
    ax.xaxis.set_major_formatter(formatter)
    import os 
    Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
    folder = os.path.exists(Eventdir)
    if not folder:
        os.makedirs(Eventdir)
    fig.autofmt_xdate()
    return plt.savefig(f'{Eventdir}t1.png',bbox_inches='tight', dpi=300)

def plotMSF(eventName,time,start_time):
    path = Path
    Sdata = pd.read_csv(f'{path}{SP}{eventName}load.csv',header=None).drop(0,axis=1).T
    Sdata.rename(columns = {0:'obs'},inplace=True)

    # Mdata = pd.read_csv(f'{path}/0525-364(3)/364(3){eventName}.csv',header=None) ## 注意OBS有沒有對齊
    # MdataT = Mdata.T
    Mdata = pd.read_csv(f'{path}{MP}{eventName}.csv',header=None).drop(0,axis=1).T
    Ldata = pd.read_csv(f'{path}{LP}{eventName}load.csv',header=None).drop(0,axis=1).T
    Bdata = pd.read_csv(f'{path}{BP}{eventName}load.csv',header=None).drop(0,axis=1).T
    
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rc('font',family = 'Times New Roman')

    plt.rcParams['mathtext.fontset'] = 'cm'
    # start_time = '20201001 10:00:00'
    t = pd.date_range(start=start_time, periods= 77, freq="1H")
    
    mm = 1/25.4 # inch 換 毫米
    fig,ax = plt.subplots(figsize=(91*mm,72*mm)) # 刚才构思的图片长、宽
    locator = AutoDateLocator(minticks=10)
    locator.intervald[HOURLY] = [8]  # 10min 为间隔
    ax.xaxis.set_major_locator(locator=locator)

    mlocator = AutoDateLocator(minticks=10) # 值越小可能只能按小时间隔显示
    mlocator.intervald[HOURLY] = [4]  # 10min 为间隔
    ax.xaxis.set_minor_locator(locator=mlocator)

    ymajorLocator = MultipleLocator(1)
    yminorLocator = MultipleLocator(.5/2) #将此y轴次刻度标签设置为0.1的倍数
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)

    ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":") #, linestyle=":"
    ax.plot(t,Mdata[time],color="#1e488f", label="SVM",linewidth=1.5) #, linestyle=":"
    ax.plot(t,Ldata[time],color="#10a674", label="LSTM",linewidth=1.5)
    ax.plot(t,Bdata[time],color="#ff9408", label="BiLSTM",linewidth=1.5)
    ax.plot(t,Sdata[time],color="darkred", label="Seq2Seq",linewidth=1.5)
    ax.axhline(2.2,color="#696969", label="Alert Level 3\n(EL. 2.2)",linewidth=0.5) #, linestyle="--"
    ax.set_ylim([-2,7])
    ax.set_xlabel('Time (h)',fontsize=9)
    ax.set_ylabel('Water Level (m)',fontsize=9)
    ax.set_title(f"Start Time: {t[time-1]}\nEnd Time: {t[time+24-1]}",fontsize=9)
    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False) #邊框
    ax.spines['top'].set_visible(False)

    ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
    plt.legend(loc='best',ncol=3,fontsize=7,facecolor='white', framealpha=1,edgecolor='white')
    # plt.show()
    # 根据自己定义的方式去画时间刻度
    formatter = plt.FuncFormatter(time_ticks)
    # 在图中应用自定义的时间刻度
    ax.xaxis.set_major_formatter(formatter)
    import os 
    Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
    folder = os.path.exists(Eventdir)
    if not folder:
        os.makedirs(Eventdir)
    fig.autofmt_xdate()
    return plt.savefig(f'{Eventdir}{time}.png',bbox_inches='tight', dpi=300)

def time_ticks(x, pos):
    # 在 pandas 中，按 10min 生成的时间序列与 matplotlib 要求的类型不一致
    # 需要转换成 matplotlib 支持的类型
    x = md.num2date(x)
    
    # 时间坐标是从坐标原点到结束一个一个标出的
    # 如果是坐标原点的那个刻度则用下面的要求标出刻度
    if pos == 0:
        # %Y-%m-%d
        # 时间格式化的标准是按 2020-10-01 10:10:10.0000 标记的
        fmt = '%Y-%m-%d %H:%M.%f'
    # 如果不是是坐标原点的那个刻度则用下面的要求标出刻度
    else:
        # 时间格式化的标准是按 10:10:10.0000 标记的
        fmt = '%m-%d %H:%M.%f'
    # 根据 fmt 的要求画时间刻度
    label = x.strftime(fmt)
    
    # 当 fmt 有%s时需要下面的代码
    label = label.rstrip("0")
    label = label.rstrip(".")
    
    # 截断了秒后面的 .000
    return label

def plotbox(indexlist):
    bp = plt.boxplot(indexlist,labels=['SVM', 'LSTM', 'BiLSTM','Seq2Seq'],patch_artist=True)
    for patch,median, color in zip(bp['boxes'],bp['medians'],["#1e488f","#10a674","#ff9408","darkred"]):
        patch.set_facecolor(color)
        median.set(color='black')
    
def plotALLMSF(eventName,start_time):
    path = Path
    Sdata = pd.read_csv(f'{path}{SP}{eventName}load.csv',header=None).drop(0,axis=1).T
    Sdata.rename(columns = {0:'obs'},inplace=True)

    # Mdata = pd.read_csv(f'{path}/0525-364(3)/364(3){eventName}.csv',header=None) ## 注意OBS有沒有對齊
    # MdataT = Mdata.T

    Mdata = pd.read_csv(f'{path}{MP}{eventName}.csv',header=None).drop(0,axis=1).T
    Ldata = pd.read_csv(f'{path}{LP}{eventName}load.csv',header=None).drop(0,axis=1).T
    Bdata = pd.read_csv(f'{path}{BP}{eventName}load.csv',header=None).drop(0,axis=1).T
    # start_time = '20201001 10:00:00'

    for model,d in zip(["Seq2Seq", "SVM", "LSTM" ,"BiLSTM"],[Sdata,Mdata,Ldata,Bdata]):
        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rc('font',family = 'Times New Roman')

        plt.rcParams['mathtext.fontset'] = 'cm'

        t = pd.date_range(start = start_time, periods= 77, freq="1H")
        # t = np.linspace(0,len(Sdata.obs)-1,len(Sdata.obs))
        mm = 1/25.4 # inch 換 毫米
        fig,ax = plt.subplots() # 刚才构思的图片长、宽
        
        # xmajorLocator = MultipleLocator(10) #設置間隔
        # xminorLocator = MultipleLocator(5)
        # ax.xaxis.set_minor_locator(xminorLocator)
        # ax.xaxis.set_major_locator(xmajorLocator)
        locator = AutoDateLocator(minticks=10)
        locator.intervald[HOURLY] = [6]  # 10min 为间隔
        ax.xaxis.set_major_locator(locator=locator)

        mlocator = AutoDateLocator(minticks=10) # 值越小可能只能按小时间隔显示
        mlocator.intervald[HOURLY] = [3]  # 10min 为间隔
        ax.xaxis.set_minor_locator(locator=mlocator)

        ymajorLocator = MultipleLocator(1)
        yminorLocator = MultipleLocator(.5/2) #将此y轴次刻度标签设置为0.1的倍数
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)

        ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":") #, linestyle=":"
        ax.plot(t,d[1],color="cornflowerblue", label=f"{model}",linewidth=1)
        for time in range(2,len(Sdata.columns)):
                plt.plot(t,d[time],color="cornflowerblue",linewidth=1)
        ax.axhline(2.2,color="#696969", label="E.L.",linewidth=0.5) #, linestyle="--"
        # ax.set_xlim([0,len(Sdata.obs)])
        ax.set_ylim([-2,7])
        ax.set_xlabel('Time (h)',fontsize=12)
        ax.set_ylabel('Water level (m)',fontsize=12)
        ax.set_title(f"Typhoon {eventName}",fontsize=14)
        ax.set_facecolor('white')
        ax.spines['right'].set_visible(False) #邊框
        ax.spines['top'].set_visible(False)

        ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
        plt.legend(loc='best',ncol=1,fontsize=10,facecolor='white', framealpha=1,edgecolor='white')
        # 根据自己定义的方式去画时间刻度
        formatter = plt.FuncFormatter(time_ticks)
        # 在图中应用自定义的时间刻度
        ax.xaxis.set_major_formatter(formatter)
        # plt.show()
        import os 
        Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
        folder = os.path.exists(Eventdir)
        if not folder:
            os.makedirs(Eventdir)
        fig.autofmt_xdate()
        plt.savefig(f'{Eventdir}{model}.png',bbox_inches='tight', dpi=300)

def MSFAverageindex(eventList = ['Dujan','Megi','Lekima','Mitag']):
    ALL_df = []
    
    for eventName in eventList:
        rmse_list = []
        mae_list = []
        cc_list = []
        ce_list = []
        # 讀取四個 CSV 檔案
        file_paths = [ f'{Path}{MP}{eventName}index.csv',
                    f'{Path}{LP}{eventName}loadindex.csv', 
                    f'{Path}{BP}{eventName}loadindex.csv',
                    f'{Path}{SP}{eventName}loadindex.csv',]
        dataframes = [pd.read_csv(file_path) for file_path in file_paths]
        
        # 計算 'CC' 和 'CE' 列的平均值
        averages = {}
        cc_avg = []
        ce_avg = []
        rmse_avg = []
        mae_avg = []
        for df in dataframes:
            cc_avg.append(df['CC'].mean())
            ce_avg.append(df['CE'].mean())
            rmse_avg.append(df['RMSE'].mean())
            mae_avg.append(df['MAE'].mean())

            rmse_list.append(df['RMSE'])
            mae_list.append(df['MAE'])
            cc_list.append(df['CC'])
            ce_list.append(df['CE'])

        averages = {'CC': cc_avg, 'CE': ce_avg, 'RMSE':rmse_avg,'MAE':mae_avg}
        averages[f'{eventName}'] = ['SVM', 'LSTM', 'BiLSTM','Seq2Seq']
        # 將結果存入新的 CSV 檔案
        Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
        plt.rcParams['figure.figsize']=(12,5)
        plt.rc('font',family = 'Times New Roman',size = 26)
        for i, li in zip(['RMSE','MAE','CC','CE'],[rmse_list,mae_list,cc_list,ce_list]):
            plotbox(li)
            if i == 'RMSE' or 'MAE':
                plt.ylabel(f"{i} (m)") 
                
            else:
                plt.ylabel(f"{i}")
              
            plt.ylim(0,1)
            plt.savefig(f'{Eventdir}{i}.png', dpi=300)
            plt.close()
        ALL_df.append(pd.DataFrame(averages))
    concat_df = pd.concat(ALL_df,axis=1)
    output_path = 'averages24.csv'
    concat_df.to_csv(f'{Path}/{output_path}',index=None)

    print('平均值已成功存入 averages.csv 檔案。')   

def p3(eventName,time,start_time):
    path = Path
    Sdata = pd.read_csv(f'{path}{SP}{eventName}_P3t24.csv',header=None).drop(0,axis=1).T
    Sdata.rename(columns = {0:'obs'},inplace=True)

    MSFdata = pd.read_csv(f'{path}{SP}{eventName}load.csv',header=None).drop(0,axis=1).T
    
    
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rc('font',family = 'Times New Roman')

    plt.rcParams['mathtext.fontset'] = 'cm'
    # start_time = '20201001 10:00:00'
    t = pd.date_range(start=start_time, periods= 65, freq="1H")
    
    mm = 1/25.4 # inch 換 毫米
    fig,ax = plt.subplots(figsize=(91*mm,72*mm)) # 刚才构思的图片长、宽
    locator = AutoDateLocator(minticks=10)
    locator.intervald[HOURLY] = [8]  # 10min 为间隔
    ax.xaxis.set_major_locator(locator=locator)

    mlocator = AutoDateLocator(minticks=10) # 值越小可能只能按小时间隔显示
    mlocator.intervald[HOURLY] = [4]  # 10min 为间隔
    ax.xaxis.set_minor_locator(locator=mlocator)

    ymajorLocator = MultipleLocator(1)
    yminorLocator = MultipleLocator(.5/2) #将此y轴次刻度标签设置为0.1的倍数
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)

    ax.plot(t,Sdata.obs[:65],color="black", label="Observation", linestyle=":") #, linestyle=":"
    ax.plot(t,MSFdata[time][:65],color="darkred", label="Seq2Seq-Obs",linewidth=1.5)
    ax.plot(t,Sdata[time][:65],color="orange", label="Seq2Seq-Fct",linewidth=1.5)
    ax.axhline(2.2,color="#696969", label="Alert Level 3\n(EL. 2.2)",linewidth=0.5) #, linestyle="--"
    ax.set_ylim([-2,7])
    ax.set_xlabel('Time (h)',fontsize=9)
    ax.set_ylabel('Water Level (m)',fontsize=9)
    ax.set_title(f"Start Time: {t[time-1]}\nEnd Time: {t[time+12-1]}",fontsize=9)
    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False) #邊框
    ax.spines['top'].set_visible(False)

    ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
    plt.legend(loc='best',ncol=2,fontsize=7,facecolor='white', framealpha=1,edgecolor='white')
    # plt.show()
    # 根据自己定义的方式去画时间刻度
    formatter = plt.FuncFormatter(time_ticks)
    # 在图中应用自定义的时间刻度
    ax.xaxis.set_major_formatter(formatter)
    import os 
    Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
    folder = os.path.exists(Eventdir)
    if not folder:
        os.makedirs(Eventdir)
    fig.autofmt_xdate()
    return plt.savefig(f'{Eventdir}{time}_p3c.png',bbox_inches='tight', dpi=300)
if __name__ == '__main__':
    # onestepIndex()
    # for n,p in zip(['Seq2Seq','BiLSTM','LSTM','SVM'],[SP,BP,LP,MP]):
    #     scatter(n,p)
    # for e in ['Dujan','Megi','Lekima','Mitag']: #,'Megi','Lekima','Mitag'
    #     for t in [12,15,18,21,24,27,30,33]: #,15,18,21,24,27
    #         plotMSF(e, t)

    '''
    for e in [Megi]: #,'Megi','Lekima','Mitag'
        # plotALLMSF(e['name'])
        # plotOne(e['name'],e['start_time'],e['num'])
        for t in [6,7]:
            plotMSF(e['name'], t, e['start_time'])
    '''
    
    # for e in [Meranti]: #,'Megi','Lekima','Mitag'
    #     plotOne(e['name'],e['start_time'],e['num'])
    # MSFAverageindex()?

    
    for e in [Megi]:
        for t in [7,8,9]:
            p3(e['name'], t, e['start_time'])
    
    time = 1
    ty = Megi
    for ty in [Megi, Lekima, Mitag]:
        eventName = ty['name'] #Dujan Megi Lekima Mitag
        
        path = Path
        Sdata = pd.read_csv(f'{path}{SP}{eventName}load.csv', header=None).drop(0, axis=1).T
        Sdata.rename(columns={0: 'obs'}, inplace=True)

        Mdata = pd.read_csv(f'{path}{SP}{eventName}_P3t24.csv', header=None).drop(0, axis=1).T

        mpl.rcParams['xtick.labelsize'] = 8
        mpl.rcParams['ytick.labelsize'] = 8
        mpl.rc('font', family='Times New Roman')

        plt.rcParams['mathtext.fontset'] = 'cm'
        t = pd.date_range(start=ty['start_time'], periods=77, freq="1H")

        mm = 1 / 25.4  # inch 換 毫米
        fig, ax = plt.subplots(figsize=(91 * mm, 72 * mm))  # 刚才构思的图片长、宽
        locator = AutoDateLocator(minticks=10)
        locator.intervald[HOURLY] = [8]  # 10min 为间隔
        ax.xaxis.set_major_locator(locator=locator)

        mlocator = AutoDateLocator(minticks=10)  # 值越小可能只能按小时间隔显示
        mlocator.intervald[HOURLY] = [4]  # 10min 为间隔
        ax.xaxis.set_minor_locator(locator=mlocator)

        ymajorLocator = MultipleLocator(1)
        yminorLocator = MultipleLocator(.5 / 2)  # 将此y轴次刻度标签设置为0.1的倍数
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":") # , linestyle=":"
        seq2seq_line, = ax.plot([], [], color="darkred", label="Seq2Seq-Obs", linewidth=1.5)
        p3t24_line, = ax.plot([], [], color="orange", label="Seq2Seq-Fct", linewidth=1.5)
        alert_line = ax.axhline(2.2, color="#696969", label="Alert Level 3\n(EL. 2.2)", linewidth=0.5)  # , linestyle="--"
        ax.set_ylim([-2, 7])
        ax.set_xlabel('Time (h)', fontsize=9)
        ax.set_ylabel('Water Level (m)', fontsize=9)
        # ax.set_title(f"Start Time: {t[time - 1]}", fontsize=5 ,loc='left')
        ax.set_title(f"Typhoon {eventName}", fontsize=9 ,loc='center')
        ax.set_facecolor('white')
        ax.spines['right'].set_visible(False)  # 邊框
        ax.spines['top'].set_visible(False)
        plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files/ffmpeg-6.0-full_build/bin/ffmpeg.exe'
        ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
        plt.legend(loc='best',ncol=2,fontsize=7,facecolor='white', framealpha=1,edgecolor='white')
        # plt.show()
        # 根据自己定义的方式去画时间刻度
        formatter = plt.FuncFormatter(time_ticks)
        # 在图中应用自定义的时间刻度
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate()
        def init():
            ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":")
            seq2seq_line.set_data([], [])
            p3t24_line.set_data([], [])
            return seq2seq_line, p3t24_line

        def update_frame(frame):
            seq2seq_line.set_data(t, Sdata[frame+1])
            p3t24_line.set_data(t, Mdata[frame+1])
            ax.set_title(f"Start Time: {t[frame - 1]}", fontsize=5,loc='left')
            return seq2seq_line, p3t24_line
        
        animation_frames = 53  # Number of frames in the animation
        ani = animation.FuncAnimation(fig, update_frame, init_func=init, frames=animation_frames, blit=True)
        
        Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
        FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec','libx264'])
        ani.save(f'{Eventdir}_p3.mp4',writer=FFwriter, dpi=300)


    # 動畫
    time = 1
    ty = Mitag
    eventName = ty['name'] #Megi Lekima Mitag
    
    path = Path
    Sdata = pd.read_csv(f'{path}{SP}{eventName}load.csv', header=None).drop(0, axis=1).T
    Sdata.rename(columns={0: 'obs'}, inplace=True)

    Mdata = pd.read_csv(f'{path}{MP}{eventName}.csv', header=None).drop(0, axis=1).T
    Ldata = pd.read_csv(f'{path}{LP}{eventName}load.csv', header=None).drop(0, axis=1).T
    Bdata = pd.read_csv(f'{path}{BP}{eventName}load.csv', header=None).drop(0, axis=1).T

    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rc('font', family='Times New Roman')

    plt.rcParams['mathtext.fontset'] = 'cm'
    t = pd.date_range(start=ty['start_time'], periods=77, freq="1H")

    mm = 1 / 25.4  # inch 換 毫米
    fig, ax = plt.subplots(figsize=(91 * mm, 72 * mm))  # 刚才构思的图片长、宽
    locator = AutoDateLocator(minticks=10)
    locator.intervald[HOURLY] = [8]  # 10min 为间隔
    ax.xaxis.set_major_locator(locator=locator)

    mlocator = AutoDateLocator(minticks=10)  # 值越小可能只能按小时间隔显示
    mlocator.intervald[HOURLY] = [4]  # 10min 为间隔
    ax.xaxis.set_minor_locator(locator=mlocator)

    ymajorLocator = MultipleLocator(1)
    yminorLocator = MultipleLocator(.5 / 2)  # 将此y轴次刻度标签设置为0.1的倍数
    ax.yaxis.set_minor_locator(yminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":") # , linestyle=":"
    svm_line, = ax.plot([], [], color="#1e488f", label="SVM", linewidth=1.5)  # , linestyle=":"
    lstm_line, = ax.plot([], [], color="#10a674", label="LSTM", linewidth=1.5)
    bilstm_line, = ax.plot([], [], color="#ff9408", label="BiLSTM", linewidth=1.5)
    seq2seq_line, = ax.plot([], [], color="darkred", label="Seq2Seq", linewidth=1.5)
    alert_line = ax.axhline(2.2, color="#696969", label="Alert Level 3\n(EL. 2.2)", linewidth=0.5)  # , linestyle="--"
    ax.set_ylim([-2, 7])
    ax.set_xlabel('Time (h)', fontsize=9)
    ax.set_ylabel('Water Level (m)', fontsize=9)
    # ax.set_title(f"Start Time: {t[time - 1]}", fontsize=5 ,loc='left')
    ax.set_title(f"Typhoon {eventName}", fontsize=9 ,loc='center')
    ax.set_facecolor('white')
    ax.spines['right'].set_visible(False)  # 邊框
    ax.spines['top'].set_visible(False)
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Program Files/ffmpeg-6.0-full_build/bin/ffmpeg.exe'
    ax.set_aspect(1.0/ax.get_data_ratio()*0.55) # 長宽比
    plt.legend(loc='best',ncol=3,fontsize=7,facecolor='white', framealpha=1,edgecolor='white')
    # plt.show()
    # 根据自己定义的方式去画时间刻度
    formatter = plt.FuncFormatter(time_ticks)
    # 在图中应用自定义的时间刻度
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    def init():
        ax.plot(t,Sdata.obs,color="black", label="Observation", linestyle=":")
        svm_line.set_data([], [])
        lstm_line.set_data([], [])
        bilstm_line.set_data([], [])
        seq2seq_line.set_data([], [])
        return svm_line, lstm_line, bilstm_line, seq2seq_line,

    def update_frame(frame):
        svm_line.set_data(t, Mdata[frame+1])
        lstm_line.set_data(t, Ldata[frame+1])
        bilstm_line.set_data(t, Bdata[frame+1])
        seq2seq_line.set_data(t, Sdata[frame+1])
        ax.set_title(f"Start Time: {t[frame - 1]}", fontsize=5,loc='left')
        return svm_line, lstm_line, bilstm_line, seq2seq_line
    
    animation_frames = 53  # Number of frames in the animation
    ani = animation.FuncAnimation(fig, update_frame, init_func=init, frames=animation_frames, blit=True)
    
    Eventdir = f'C:/Users/309/Documents/GitHub/TPB Code/出圖/{eventName}/'
    FFwriter = animation.FFMpegWriter(fps=10, extra_args=['-vcodec','libx264'])
    ani.save(f'{Eventdir}fps5.mp4',writer=FFwriter, dpi=300)
    
    # ani.save(f'{Eventdir}{time}.gif', writer='pillow', dpi=300)

    
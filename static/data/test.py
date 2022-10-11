import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
import xesmf as xe
import redis
import pandas as pd
import datetime
import math
class ec_data_point:
    def __init__(self, select_time,select_type,select_lat,select_lon): 
        self.var_list = ['u10','v10','tcc','t2','skt','lsp','cp','tp','r'] # 风向风速、云量、温度、tp（总降水）、相对湿度
        self.data = { }
        self.read_data()
    # 创建对象读取io的核心代码####################################################################
    def regrid_data(self,data):
        ds_out = xr.Dataset(
            {   
                
                "lat": (["lat"], np.arange(27.8, 29.5, 0.05)),
                "lon": (["lon"], np.arange(120, 122, 0.05)),
            }
        )
        regridder = xe.Regridder(data, ds_out, "bilinear")
        dr_out = regridder(data)
        return dr_out
    def read_data(self):
        file_path = "/home/liyuan3970/Data/My_Git/2022041700/*.nc" 
        f = xr.open_mfdataset(file_path, parallel=False)
        # 列表数据--
        u10 = f.u10.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        u10 =  u10.swap_dims({'latS':'lat','lonS':'lon'})
        v10 = f.v10.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        v10 =  v10.swap_dims({'latS':'lat','lonS':'lon'})
        tcc = f.tcc.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        tcc =  tcc.swap_dims({'latS':'lat','lonS':'lon'})
        t2 = f.t2.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        t2 =  t2.swap_dims({'latS':'lat','lonS':'lon'})
        skt = f.skt.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        skt =  skt.swap_dims({'latS':'lat','lonS':'lon'})
        lsp = f.lsp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        lsp =  lsp.swap_dims({'latS':'lat','lonS':'lon'})
        cp = f.cp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8)) 
        cp =  cp.swap_dims({'latS':'lat','lonS':'lon'})    
        tp = f.tp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8)) 
        tp =  tp.swap_dims({'latS':'lat','lonS':'lon'}) 
        r = f.r.sel(lev=1000,lonP=slice(120,122),latP=slice(29.5,27.8)) # ！！！如果经纬度放大就会报错。。。
        r =  r.swap_dims({'latP':'lat','lonP':'lon'})
        r['lon'] = r['lonP']
        r['lat'] = r['latP']  
        # print(tp,r)
        # 空间插值----------------------------------
        # ----------------------------------------
        # ----------------------------------------
        # 此处为空间插值函数0.05
        grid_u10 = self.regrid_data(u10)
        grid_v10 = self.regrid_data(v10)
        grid_tcc = self.regrid_data(tcc)
        grid_t2 = self.regrid_data(t2)
        grid_skt = self.regrid_data(skt)
        grid_lsp = self.regrid_data(lsp)
        grid_cp = self.regrid_data(cp)
        grid_tp = self.regrid_data(tp)
        grid_r = r#self.regrid_data(r)
        # 数据的重新加载
        
        self.data['u10'] = grid_u10
        self.data['v10'] = grid_v10
        self.data['tcc'] = grid_tcc
        self.data['t2'] = grid_t2
        self.data['skt'] = grid_skt
        self.data['lsp'] = grid_lsp
        self.data['cp'] = grid_cp    
        self.data['tp'] = grid_tp 
        self.data['r'] = grid_r 
        # print(grid_vis.sel(lon=121.5, lat=28.5,method='nearest'))    
    # 以下为处理html表格的核心代码######################################################################################################################
    #  1. 计算超出指定日期的时间段数
    def time_point_len(self,times,step):
        '''计算时间步长'''
        start = str(times[0:10] + ' 00:00:00')
        t_start =datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end = str(times+':00')
        t_end =datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        timedelta = (t_end - t_start).seconds
        hours = timedelta/3600
        if step == 'hours':
            if hours <= 6:
                step_len = 0
            elif hours >6 and hours <= 12:
                step_len = 1
            elif hours >12 and hours <= 18:
                step_len = 2
            elif hours >18 and hours <= 24:
                step_len = 3
            return step_len
        elif step == '3hours':
            if hours <= 3:
                step_len = 0
            elif hours >3 and hours <= 6:
                step_len = 1
            elif hours >6 and hours <= 9:
                step_len = 2
            elif hours >9 and hours <= 12:
                step_len = 3
            elif hours >12 and hours <= 15:
                step_len = 4
            elif hours >15 and hours <= 18:
                step_len = 5
            elif hours >18 and hours <= 21:
                step_len = 6
            elif hours >21 and hours <= 24:
                step_len = 7   
            return step_len
        else:
            if hours <= 6:
                step_len = 0
            elif hours >6 and hours <= 12:
                step_len = 1
            elif hours >12 and hours <= 18:
                step_len = 2
            elif hours >18 and hours <= 24:
                step_len = 3
            return step_len  
    def return_dates_step(self):
        '''用于返回dates 和start_len,end_len'''
        # 起始时间
        start_day = r'2022-04-18 00:00'
        end_day = r'2022-04-22 13:00'
        # EC数据
        single_point_data = {
            'wind_speed':[i for i in range(241)],
            'wind_dir':[i for i in range(241)],
            'tcc':[i for i in range(241)],
            'skt':[i for i in range(241)],
            't2':[i for i in range(241)],
            'lsp':[i for i in range(241)],
            'cp':[i for i in range(241)],
            'tp':[i for i in range(241)],
            'r':[i for i in range(241)]
            
        }
        step = '3hours'
        time_data = pd.date_range(start='2022-04-17 00:00:00',end='2022-04-27 00:00:00',freq='1H')
        data_index = [i for i in range(241) ]
        ts = pd.Series(data_index, time_data)
        start_len = self.time_point_len(start_day,step)
        end_len = self.time_point_len(end_day,step)
        # 计算日期
        dates = []
        dt = datetime.datetime.strptime(start_day[0:10], "%Y-%m-%d")
        date = start_day[0:10] + ' 00:00'
        while date <= end_day:
            dates.append(date)
            dt = dt + datetime.timedelta(1)
            date = dt.strftime("%Y-%m-%d") + ' 00:00'
        return dates,start_len,end_len,ts  
    def return_timestep(self,dates,step,start_len,end_len,ts,single_point_data):
        '''返回表第一行的日期数据'''
        step_data = []
        if step == 'hours':
            # 6是每段时间的个数
            day_step = ['凌晨','上午','下午','晚上']
            step_data_num = 6
            step_list = [
                ['00:00~01:00','01:00~02:00','02:00~03:00','03:00~04:00','04:00~05:00','05:00~06:00'],
                ['06:00~07:00','07:00~08:00','08:00~09:00','09:00~10:00','10:00~11:00','11:00~12:00'],
                ['12:00~13:00','13:00~14:00','14:00~15:00','15:00~16:00','16:00~17:00','17:00~18:00'],
                ['18:00~19:00','19:00~20:00','20:00~21:00','21:00~22:00','22:00~23:00','23:00~24:00']
            ]
        elif step == '3hours':
            step_data_num = 1
            step_list = [
                ['00:00~03:00'],['03:00~06:00'],['06:00~09:00'],['09:00~12:00'],['12:00~15:00'],['15:00~18:00'],['18:00~21:00'],['21:00~24:00']   
            ]
        else:
            step_data_num = 1
            step_list = [
                ['凌晨'],['上午'],['下午'],['晚上']   
            ]
        # day_list
        if step != 'hours':
            day_list = [(i[5:7] + '月' + i[8:10] + '日') for i in dates]
        else:
            day_list = []
        for i in range(len(dates)):
            start_date , start_index = dates[i],ts[dates[i]]
            if i == 0: 
                # 第一天的数据         
                if step == 'hours':
                    # 计算天的段数
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    start_day_list = [day_str + i for i in day_step ]
                    day_list.extend(start_day_list[start_len:]) 
                    len_max = 4
                    for start_num in range(len_max-start_len):
                        single_step_data = {}   
                        single_step_data['step_list'] = step_list[(start_num+start_len)]
                        #  6是每段时间的个数
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                elif step == '3hours':
                    len_max = 8
                    for start_num in range(len_max-start_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[(start_num+start_len)]   
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*3  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                else:
                    len_max = 4  
                    for start_num in range(len_max-start_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[(start_num+start_len)]   
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*6  + step_num ])    
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*6  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)      
            elif i ==len(dates)-1:    
                # 最后一天的数据
                if step == 'hours':
                    # 最后一天的day
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    end_day_list = [day_str + i for i in day_step ]
                    day_list.extend(end_day_list[:end_len]) 
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
                elif step == '3hours':
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*3  + step_num])
                            single_step_data['step_data'].append(appen_data)                                    
                        step_data.append(single_step_data)
                else:
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
            else:
                # 中间天
                if step == 'hours':
                    # 添加 数据
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    middle_day_list = [day_str + i for i in day_step ]
                    day_list.extend(middle_day_list[:]) 
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
                elif step == '3hours':
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*3  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                else:
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)        
                        step_data.append(single_step_data)
        data = {
            'day':day_list,
            'time_step':step_data
        }
        return data
    
    def decode_html_table(self,data):
        '''解析html的数据'''
        # 用来解析数据并返回表格的html数据
        # 日期、间隔
        step = 'prehours'# 'sixhours';'thrdhours'
        len_day = len(data['day'])
        len_setp = 3#len(data['time_step']['step_list'])
        html_table = ""
        return html_table
    # 实例创建后和用户交互的核心代码###############################################################################################
    def interp1d_data(self,data):
        ''' 将get_data和get_single_data的数据进行时间插值'''
        date = data.index.tolist()
        value = data.tolist()
        d = pd.DataFrame()
        d['date'] = pd.to_datetime(date)
        d['val'] = value
        helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})
        d = pd.merge(helper, d, on='date', how='left').sort_values('date')
        d['val'] = d['val'].interpolate(method='linear')        
        return d['val'].tolist() 
    def accum_data(self,data):
        '''累计降水的计算'''
        div = []
        accum = data.tolist()
        for i in range(len(data.tolist())):
            if i==0:
                div.append(0)
            else:
                div.append(accum[i] - accum[i-1] )
        output = []
        for i in range(len(div)):
            if i<=24:
                for j in range(3):
                    output.append(accum[i]/3)
            else:
                for j in range(6):
                    output.append(accum[i]/6)
        return output
    def step_single_data(self,data):
        '''分时次处理输出数据'''
        step ="6hours"
        rain = []
        if step=="hours":
            init_item = len(pd.date_range(start=init_day,end=start_day,freq='1H'))     
        elif step=="3hours":
            for i in range(len(data)):
                if np.isnan(data[i]):
                    data[i] = 0
                if i ==0:
                    rain_all = 0
                    rain.append(rain_all)
                elif (i+1)%3 ==0:
                    rain_all = data[i] + data[i-1] + data[i-2]
                    rain.append(rain_all)
                else:
                    rain_all = 0.0
                    rain.append(rain_all)
        elif step=="6hours":
            for i in range(len(data)):
                if np.isnan(data[i]):
                    data[i] = 0
                if i ==0:
                    rain_all = 0
                    rain.append(rain_all)
                elif (i+1)%6 ==0:
                    rain_all = data[i] + data[i-1] + data[i-2] + data[i-3] + data[i-4] +  data[i-5]
                    rain.append(rain_all)
                else:
                    rain_all = 0.0
                    rain.append(rain_all)
        return rain
    def get_single(self,select_lat,select_lon):
        '''用于处理特定经纬度数据'''
        # single_point_data = {
        #     'vis':vis
        # }
        single_point_data = {}
        single_point_inter1d_data = {}
        for var in self.var_list:
            select_data = self.data[var]
            single_point_data[var] = select_data.sel(lon=select_lon, lat=select_lat,method='nearest').to_pandas()
            if var in ['u10','v10','tcc','t2','skt','r']:
                single_point_inter1d_data[var] = self.interp1d_data(single_point_data[var])
            else:
                single_point_inter1d_data[var] = self.accum_data(single_point_data[var])
        return single_point_inter1d_data 
    # 处理所有数据的核心data #############################################################################################################
    def comput_all_data(self):
        '''用于计算所有数据的核心代码
        1.依据step 返回single_data
        2.依据step 返回曲线图的data 4个list 一个字典list
        '''
        # 计算单点数据
        # single_point_data = self.get_single(27.5,125.7)
        single_point_data = {
            'wind_speed':[i for i in range(243)],
            'wind_dir':[i for i in range(243)],
            'tcc':[i for i in range(243)],
            'skt':[i for i in range(243)],
            't2':[i for i in range(243)],
            'lsp':[i for i in range(243)],
            'cp':[i for i in range(243)],
            'tp':[i for i in range(243)],
            'r':[i for i in range(243)]
            
        }
        dates,start_len,end_len,ts  = self.return_dates_step()
        # dates = ['2022-04-18 00:00', '2022-04-19 00:00', '2022-04-20 00:00', '2022-04-21 00:00', '2022-04-22 00:00']
        step = '3hours'
        # 
        single_point_data = self.single_point_data(step)
        data = self.return_timestep(dates,step,start_len,end_len,ts,single_point_data)
        html_table = self.decode_html_table(data)
        
        # 计算datalist
        wind_list = single_point_data['r']
        r_list = single_point_data['r']
        temp_list = single_point_data['r']
        pre_list = single_point_data['r']
        return dates,pre_list,html_table,data
    def wind_dir(self,u,v):
        deg = 180.0/np.pi
        rad = np.pi/180.0 
        wdir =  180.0 + np.arctan2(u, v)*deg
        widsped = math.sqrt(u*u + v*v)
        return wdir,widsped
    def single_point_data(self,step):
        '''用于返回满足echart图标的数据'''
        # single_point_data = self.get_single(27.5,125.7)
        init_day =  r'2022-04-17 08:00'
        start_day = r'2022-04-18 00:00'
        end_day = r'2022-04-22 13:00'
        step ="3hours"
        single_point_inter1d_data = self.get_single(28.5,121.7)
        wind_speed = []
        wind_dir = []
        tcc = []
        for i in range(len(single_point_inter1d_data['u10'])):
            wdir,widsped = self.wind_dir(single_point_inter1d_data['u10'][i],single_point_inter1d_data['v10'][i])
            wind_speed.append(widsped)
            wind_dir.append(wdir)       
        rain = self.step_single_data(single_point_inter1d_data['tp'])
        single_point_data = {
            'wind_speed':wind_speed,
            'wind_dir':wind_dir,
            'tcc':single_point_inter1d_data['tcc'],
            't2':single_point_inter1d_data['t2'],
            'tp':rain,
            'r':single_point_inter1d_data['r']
            
        }  
        # list_data  1.相对湿度r 2.降水pre 3.风向风速wind_all 4温度temp 5.date
        # 编码逻辑  241个时次，每个时次的数据都是卡在对应的时间点上的
        # r   list
        windall = []
        if step =="hours":
            init_item = len(pd.date_range(start=init_day,end=start_day,freq='1H'))
            time_data = pd.date_range(start=start_day,end=end_day,freq='1H')
            num_time = len(time_data) - len(time_data)%1
            pre = single_point_data['r'][num_time::1]
            temp = single_point_data['t2'][num_time::1]
            r = single_point_data['r'][num_time::1]
            for i in range(num_time):
                windinfo = {
                    'symbol': 'path://M10 10L60 10 60 20 20 20 20 40 60 40 60 50 20 50 20 100 10 100 10 10z',
                    'symbolRotate': single_point_data['wind_speed'][init_item+i],
                    'value': single_point_data['wind_dir'][init_item+i]
                }
                windall.append(windinfo)
        elif step =="3hours":
            init_item = len(pd.date_range(start=init_day,end=start_day,freq='1H'))
            time_data = pd.date_range(start=start_day,end=end_day,freq='1H')
            num_time = len(time_data) - len(time_data)%3
            pre = single_point_data['r'][num_time::3]
            temp = single_point_data['t2'][num_time::3]
            r = single_point_data['r'][num_time::3]
            # 具体的业务逻辑
            for i in range(num_time):
                windinfo = {
                    'symbol': 'path://M10 10L60 10 60 20 20 20 20 40 60 40 60 50 20 50 20 100 10 100 10 10z',
                    'symbolRotate': single_point_data['wind_speed'][init_item+i],
                    'value': single_point_data['wind_dir'][init_item+i]
                }
                windall.append(windinfo)
        elif step =="6hours":
            init_item = len(pd.date_range(start=init_day,end=start_day,freq='1H'))
            time_data = pd.date_range(start=start_day,end=end_day,freq='1H')
            num_time = len(time_data) - len(time_data)%6
            pre = single_point_data['r'][num_time::6]
            temp = single_point_data['t2'][num_time::6]
            r = single_point_data['r'][num_time::6]
            for i in range(num_time):
                windinfo = {
                    'symbol': 'path://M10 10L60 10 60 20 20 20 20 40 60 40 60 50 20 50 20 100 10 100 10 10z',
                    'symbolRotate': single_point_data['wind_speed'][init_item+i],
                    'value': single_point_data['wind_dir'][init_item+i]
                }
                windall.append(windinfo)

        # 返回有 single_data 和 date 
        return single_point_data 
        
        
# 开始数据

select_time,select_type,select_lat,select_lon = '2022041700','t',27.5,125.7
ec_worker = ec_data_point(select_time,select_type,select_lat,select_lon) 
a,b,c,d = ec_worker.comput_all_data()
print(d)
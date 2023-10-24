import base64
import json
import os
from io import BytesIO
from math import isnan
import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
import shapefile
import xarray as xr
from affine import Affine
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from rasterio import features
import matplotlib as mpl
import redis
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import datetime as dtt
from datetime import timezone
from datetime import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from pylab import *
from matplotlib.font_manager import FontProperties
import pymysql
from pymysql.converters import escape_string
import pickle
# 平滑
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D
# 雷达
import cinrad
from cinrad.visualize import Section
import geojsoncontour
# 自定义画图类
class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels/ self.levels.max()
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._y = np.linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)
# 实况数据class
class station_zdz:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
    def time_today(self):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        return today
    def get_redis(self,date_type):
        '''根据date_type向redis中获取数据'''
        value = self.rs.get(date_type)
        if value:
            data = pickle.loads(value)
        else:
            data = None       
        #data = pickle.loads(self.rs.get(date_type))
        # 解析数据
        return data
    def hours_wind(self,data):
        data['TTime'] = pd.to_datetime(data['TTime'])
        data['Year'] = data['TTime'].dt.year
        data['Month'] = data['TTime'].dt.month
        data['Day'] = data['TTime'].dt.day
        data['Hour'] = data['TTime'].dt.hour
        grouped_IIIII = data.groupby(['Day','Hour'])
        time_list = []
        dir_list = []
        speed_list = []
        for i in grouped_IIIII.size().index:
            single = grouped_IIIII.get_group(i)
            value = single[ single['fFy']== single['fFy'].max()].head(1)
            time_list.append(value['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S').values[0])
            speed_list.append(value['fFy'].values[0])
            dir_list.append(value['dFy'].values[0])
        return time_list,speed_list,dir_list
    def single_station(self,station):
        '''获取单站数据并解析'''
        # 获取当前时间
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        end = today.strftime('%Y-%m-%d %H:%M:%S')
        offset = dtt.timedelta(days=-1)
        start = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        # 造假数据
        start = '2019-08-08 16:53:00'
        end = '2019-08-09 16:53:00' 
        yesday = start[0:10] + " 20:00:00"
        today = end[0:10] + " 20:00:00"
        hours = dtt.datetime.strptime(end,'%Y-%m-%d %H:%M:%S').hour
        # 数据库中读取单站数据并解析
        sql = """select TTime,IIIII,Ri,T,V,FF as fFy,DF as dFy 
        from Tab_AM_M where ( TTime between '{start_time}' and '{end_time}' and IIIII='{station}')  order by TTime  """
        rsql = sql.format(start_time=start,end_time=end,station=station)
        data = pd.read_sql(rsql , con=self.conn) 
        wind_time,wind_speed,wind_dir =  self.hours_wind(data)
        wind_data = pd.DataFrame(data={'TTime':wind_time,'dFy':wind_dir,'fFy':wind_speed})   
        data['TTime'] = data['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        wind_data['TTime'] = pd.to_datetime(wind_data['TTime'])
        if hours>=20:
            # 实时数据
            now_data = data[data['TTime']>=today]
            now_data = now_data[now_data['TTime']<=end]
            now_wind = wind_data[wind_data['TTime']>=today]
            now_wind = now_wind[now_wind['TTime']<=end]
            # 历史数据
            his_data = data[data['TTime']>start]
            his_data = his_data[his_data['TTime']<=today]
            his_wind = wind_data[wind_data['TTime']>start]
            his_wind = his_wind[his_wind['TTime']<=today]
        else:
            # 实时数据
            now_data = data[data['TTime']>=yesday]
            now_data = now_data[now_data['TTime']<=end]
            now_wind = wind_data[wind_data['TTime']>=yesday]
            now_wind = now_wind[now_wind['TTime']<=end]
            # 历史数据
            his_data = data[data['TTime']>start]
            his_data = his_data[his_data['TTime']<=yesday]
            his_wind = wind_data[wind_data['TTime']>start]
            his_wind = his_wind[his_wind['TTime']<=yesday]
        his_wind['TTime'] = his_wind['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        now_wind['TTime'] = now_wind['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        output = pd.concat([now_data,his_data])  
        history = his_data.to_json(orient='values',force_ascii=False)
        nowdata = now_data.to_json(orient='values',force_ascii=False)
        windhis = his_wind.to_json(orient='values',force_ascii=False)
        windnow = now_wind.to_json(orient='values',force_ascii=False)
        output = {
            "now":nowdata,
            "his":history,
            "wind_time":wind_time,
            "wind_speed":wind_speed,
            "wind_dir":wind_dir
        }
        # 解析数据成两个序列
        return nowdata,history,windhis,windnow
    def sql_hours(self,boundary,value_index):
        """同步小时数据表"""
        start_times = "2023-07-18 01:15:00"
        end_times = "2023-07-18 13:27:00"
        if value_index==3:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(wind) as wind
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and wind>0 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['WIN_S_Gust_Max'] = data.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
            data['WIN_D_Gust_Max'] = data.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
            data['value'] = data['WIN_S_Gust_Max']
        elif value_index==0:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, sum(rain) as PRE_sum
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and rain<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['PRE_sum']
        elif value_index==1:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(tmax) as tmax
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and tmax<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmax']
        elif value_index==2:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(tmin) as tmin
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and tmin<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmin']
        elif value_index==4:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(view) as view
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and view<30000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['view']
        return data 
    def sql_range(self,boundary,value_index):
        # 尽量两小时为主---
        start_times = "2023-07-18 11:15:00"
        end_times = "2023-07-18 13:27:00"
        if value_index==3:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(WIN_S_Gust_Max*10000 + WIN_D_Gust_Max) as wind
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and WIN_S_Gust_Max<5000 and WIN_D_Gust_Max<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['WIN_S_Gust_Max'] = data.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
            data['WIN_D_Gust_Max'] = data.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
            data['value'] = data['WIN_S_Gust_Max']
        elif value_index==0:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, sum(PRE) as PRE_sum
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and PRE<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['PRE_sum']
        elif value_index==1:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(TEM) as tmax
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and TEM<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmax']
        elif value_index==2:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(TEM) as tmin
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and TEM<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmin']
        elif value_index==4:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(VIS_HOR_1MI) as view
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and VIS_HOR_1MI<30000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['view']
        return data
    def sql_now(self,boundary,value_index):
        '''根据date_type向redis中获取数据'''
        now_times = "2023-07-18 13:27:00"
        value_list = ["PRE","TEM","TEM","WIN_S_Gust_Max","VIS_HOR_1MI"]
        if value_index!=3:
            sql = """select City,Cnty,Station_Id_C,Station_levl,Province,Station_Name,Town,Alti,Lat,Lon, {value} 
            from table_station_min 
            where Datetime= '{time}' and {value}<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}"""
        else:
            sql = """select City,Cnty,Station_Id_C,Station_levl,Province,Station_Name,Town,Alti,Lat,Lon, {value}, WIN_D_Gust_Max
            from table_station_min 
            where Datetime= '{time}' and {value}<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}"""            
        rsql = sql.format(time=now_times,value=value_list[value_index],lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
        data = pd.read_sql(rsql, con=self.conn)
        data['value'] = data[value_list[value_index]]
        return data
    def get_regin(self,boundary,table_type,tables_name,value_index,zoom):
        '''解码单站数据'''
        ascending_list = [False,False,True,False,True]
        if tables_name=="now":
            data = self.sql_now(boundary,value_index)
            boundary_data =  data
        else:
            if tables_name in ["3hours","2hours","1hours","45mins","30mins"]:
                data = self.sql_range(boundary,value_index)
                boundary_data =  data
            else:
                data = self.sql_hours(boundary,value_index)
                lat0 = boundary[0]
                lat1 = boundary[1]
                lon0 = boundary[2]
                lon1 = boundary[3]
                boundary_data =  data[(data['Lat']>lat0) & (data['Lat']<lat1)  &  (data['Lon']<lon1) & (data['Lon']>lon0)]  
        if table_type=="nation":
            remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
        if table_type=="regin":
            remain = boundary_data[boundary_data['Station_levl']==14]
        elif table_type=="all":
            remain = boundary_data
        elif table_type=="main":
            remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==15)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
        elif table_type=="auto":   
            if value_index!=0:        
                if (boundary[3]-boundary[2])<2.9:
                    remain = boundary_data
                elif (boundary[3]-boundary[2])<6:
                    remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==15)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
                else:
                    remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
            else:
                remain = boundary_data


        remain.sort_values(by="value",axis=0,ascending=ascending_list[value_index],inplace=True) # 从大到小 
            
        output = remain.to_json(orient='records',force_ascii=False)
        return output


# 气象服务快报的相关
class station_text:
    def __init__(self,city_code,start,end):
        #self.rs = redis.Redis(host='127.0.0.1', port=6379)
#         self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="tzqxj58660",db="ZJSZDZDB")
        self.userId = "BEHZ_TZSJ_TZSERVICE" 
        self.pwd = "Liyuan3970!@" 
        self.dataFormat = "json"
        self.city_code = city_code
        self.start = start
        self.end = end
        self.data = self.comupt_city_csv(city_code,start,end)
        self.city_codes = ["331000"]
    def comput_city(self,city_code,start,end):
        """快报或者统计数据的接口"""
        # getSurfEleInRegionByTime
        date_start = dtt.datetime.strptime(start,'%Y-%m-%d %H:%M:%S') 
        date_end = dtt.datetime.strptime(end,'%Y-%m-%d %H:%M:%S') 
        offset = dtt.timedelta(minutes=-60*8)
        label_start = (date_start + offset).strftime('%Y%m%d%H%M') + "00"
        label_end = (date_end + offset).strftime('%Y%m%d%H%M') + "00"   
        labels = "[" + label_start + "," + label_end + "]"
        #timelabel = self.decode_time_str(timesdelay)
        client = DataQueryClient()
        interfaceId = "statSurfEleInRegion"  
        params = {
            'dataCode':"SURF_CHN_MUL_MIN",  #SURF_CHN_MUL_HOR
            'adminCodes':"331000",
            'elements':"Province,City,Cnty,Town,Lat,Lon,Alti,Station_levl,Station_Name,Station_Id_C,",
            'timeRange':labels,
            'statEles':'MAX_TEM,MIN_TEM,SUM_PRE,MAX_PRE_1h,MIN_VIS_HOR_1MI,MAX_WIN_S_Gust_Max,MAX_WIN_S_Avg_2mi',
            #'orderBy':"MAX_WIN_S_Inst_Max:desc",
            'staLevels':"011,012,013,014,015,016", # 12国家站 14区域站
            'limitCnt':"100000000"
        }
        result = client.callAPI_to_serializedStr(self.userId, self.pwd, interfaceId, params, self.dataFormat)
        #rint(result)
        result_json = json.loads(result)
        clomns = "Province,City,Cnty,Town,Lat,Lon,Alti,Station_levl,Station_Name,Station_Id_C,MAX_TEM,MIN_TEM,SUM_PRE,MAX_PRE_1h,MIN_VIS_HOR_1MI,MAX_WIN_S_Gust_Max,MAX_WIN_S_Avg_2mi".split(",")
        data = pd.DataFrame(result_json['DS'],columns=clomns)
        data.columns = "Province,City,Cnty,Town,Lat,Lon,Height,Stationlevl,StationName,IIIII,tmax,tmin,rain,rainhour,view,wind,windave".split(",")
        return data
    def comupt_city_csv(self,city_code,start,end):
        data = pd.read_csv("static/data/downfile/comput.csv")
        # data = pd.read_csv("downfile/comput.csv")
        return data
    def text_wind(self,plot_data):
        if plot_data=="none":
            orig = self.data
            data = orig.sort_values(by="wind",ascending=False)#.head(5).to_dict()
            wind = data[(data['wind']>10.7) & (data['wind']<5009)]   
            del wind['Unnamed: 0'] # 天擎不需要
            wind.reset_index(drop=True)# 天擎不需要
        else:
            wind = pd.read_json(json.dumps(plot_data), orient='records')
        ## 数据分级
        bins=[0,10.7,13.8,17.1,20.7,24.4,28.4,32.6,36.9,41.4,46.1,50.9,56,80]
        labels=['6级风以下','6级风','7级风','8级风','9级风','10级风','11级风','12级风','13级风','14级风','15级风','16级风','17级风']
        wind['rank']=pd.cut(wind['wind'],bins,right=False,labels=labels)
        # 获取单站较大的有
        testframe = wind.head(5)
        wind_json = wind.sort_values(by="wind",ascending=False).to_json(orient='records',force_ascii=False) 
        indextext = ""
        for index,row in testframe.iterrows():
            if self.city_code in self.city_codes:
                indextext = indextext + row['Cnty'] + row['StationName'] + str(row['wind']) + "米/秒，"
                #print(index,row['City'],row['Cnty'],row['Town'],row['wind'])
            else:
                indextext = indextext + row['Town'] + row['StationName'] + str(row['wind']) + "米/秒，"
                #print(index,row['City'],row['Cnty'],row['Town'],row['wind'])
        indextext = indextext[:-1] + "。"
        rank_station = {
            '17级及以上风':len(wind[(wind['wind']>=56.4) & (wind['wind']<80)].value_counts()),
            '16级风':len(wind[(wind['wind']>=50.9) & (wind['wind']<56.4)].value_counts()),
            '15级风':len(wind[(wind['wind']>=46.1) & (wind['wind']<50.9)].value_counts()),
            '14级风':len(wind[(wind['wind']>=41.4) & (wind['wind']<46.1)].value_counts()),
            '13级风':len(wind[(wind['wind']>=36.9) & (wind['wind']<41.4)].value_counts()),
            '12级风':len(wind[(wind['wind']>=32.6) & (wind['wind']<36.9)].value_counts()),
            '11级风':len(wind[(wind['wind']>=28.4) & (wind['wind']<32.6)].value_counts()),
            '10级风':len(wind[(wind['wind']>=24.4) & (wind['wind']<28.4)].value_counts()),
            '9级风':len(wind[(wind['wind']>=20.7) & (wind['wind']<24.4)].value_counts()),
            '8级风':len(wind[(wind['wind']>=17.1) & (wind['wind']<20.7)].value_counts()),
            '7级风':len(wind[(wind['wind']>=13.8) & (wind['wind']<17.1)].value_counts()),
            '6级风':len(wind[(wind['wind']>=10.7) & (wind['wind']<13.8)].value_counts()),
            '6级风以下':len(wind[(wind['wind']>0) & (wind['wind']<10.7)].to_dict())
        }
        #  统计各等级风力的个数
        numbertext = ""
        for key, value in rank_station.items():
            if value>0:
                numbertext = numbertext + key +"有" + str(value) + "站,"
        numbertext = numbertext[:-1] + "。"
        if rank_station['17级及以上风'] >0 :
            text = "【风力通报】 全市出现17级及以上大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['16级风'] >0 :
            text = "【风力通报】 全市出现16级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext  
        elif rank_station['15级风'] >0 :
            text = "【风力通报】 全市出现15级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['14级风'] >0 :
            text = "【风力通报】 全市出现14级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['13级风'] >0 :
            text = "【风力通报】 全市出现13级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['12级风'] >0 :
            text = "【风力通报】 全市出现12级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['11级风'] >0 :
            text = "【风力通报】 全市出现11级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['10级风'] >0 :
            text = "【风力通报】 全市出现10级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['9级风'] >0 :
            text = "【风力通报】 全市出现7～9级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['8级风'] >0 :
            text = "【风力通报】 全市出现6～8级大风，风力较大的有："
            text = text + indextext + "其中，"  + numbertext
        else:
            text = ""
        return text,wind_json
    def text_view(self,plot_data):
        if plot_data=="none":
            orig = self.data
            data = orig.sort_values(by="view",ascending=True)#.head(5).to_dict()
            view = data[(data['view']>0) & (data['view']<1000)]
            del view['Unnamed: 0'] # 天擎不需要
            view.reset_index(drop=True)# 天擎不需要
        else:
            view = pd.read_json(json.dumps(plot_data), orient='records')
        bins=[0,50,200,500,99999]
        labels=['强浓雾','浓雾','大雾','正常']
        view['rank']=pd.cut(view['view'],bins,right=False,labels=labels)
        rank_station = {
           '强浓雾':len(view[(view['view']>0) & (view['view']<50)].to_dict()),
            '浓雾':len(view[(view['view']>=50) & (view['view']<200)].value_counts()),
            '大雾':len(view[(view['view']>=200) & (view['view']<500)].value_counts()),
            '正常':len(view[(view['view']>=500) & (view['view']<99990)].value_counts()) 
        }
        # 获取单站较大的有
        testframe = view.head(5) 
        view_json = view.sort_values(by="view",ascending=True).to_json(orient='records',force_ascii=False)
        indextext = ""
        for index,row in testframe.iterrows():
            if self.city_code in self.city_codes:
                indextext = indextext + row['Cnty'] + row['StationName'] + str(row['view']) + "米，"
            else:
                indextext = indextext + row['Town'] + row['StationName'] + str(row['view']) + "米，"
        indextext = indextext[:-1] + "。"
        #  统计各等级风力的个数
        numbertext = ""
        for key, value in rank_station.items():
            if value>0 and key!="正常":
                numbertext = numbertext + key +"的有" + str(value) + "站,"
        numbertext = numbertext[:-1] + "。"
        if rank_station['强浓雾'] >0 :
            text = "【能见度】 全市出现低于50米的强浓雾，能见度较低的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['浓雾'] >0 :
            text = "【能见度】 全市出现低于200米的浓雾，能见度较低的有："
            text = text + indextext + "其中，"  + numbertext  
        elif rank_station['大雾'] >0 :
            text = "【能见度】 全市出现低于500米的大雾，能见度较低的有："
            text = text + indextext + "其中，"  + numbertext
        else:
            text =""
        return text,view_json
    def text_tmax(self,plot_data):
        if plot_data=="none":
            orig = self.data
            data = orig.sort_values(by="tmax",ascending=False)#.head(5).to_dict()
            tmax = data[(data['tmax']>-40) & (data['tmax']<100)]
            del tmax['Unnamed: 0'] # 天擎不需要
            tmax.reset_index(drop=True)# 天擎不需要
        else:
            tmax = pd.read_json(json.dumps(plot_data), orient='records')
        bins=[-40,35,37,40,80]
        labels=['正常','35度以上','37度以上','40度以上']
        tmax['rank']=pd.cut(tmax['tmax'],bins,right=True,labels=labels)
        rank_station = {
            '40度以上':len(tmax[(tmax['tmax']>=40) & (tmax['tmax']<80)].value_counts()),
            '37度以上':len(tmax[(tmax['tmax']>=37) & (tmax['tmax']<40)].value_counts()),
            '35度以上':len(tmax[(tmax['tmax']>=35) & (tmax['tmax']<37)].value_counts()),
           '正常':len(tmax[(tmax['tmax']>-40) & (tmax['tmax']<35)].to_dict())      
        }
        # 获取单站较大的有
        testframe = tmax.head(5) 
        tmax_json = tmax.sort_values(by="tmax",ascending=False).to_json(orient='records',force_ascii=False)
        indextext = ""
        for index,row in testframe.iterrows():
            if self.city_code in self.city_codes:
                indextext = indextext + row['Cnty'] + row['StationName'] + str(row['tmax']) + "℃，"
            else:
                indextext = indextext + row['Town'] + row['StationName'] + str(row['tmin']) + "℃，"
        indextext = indextext[:-1] + "。"
        #  统计各等级风力的个数
        numbertext = ""
        for key, value in rank_station.items():
            if value>0 and key not in ["正常","35度以上"]:
                numbertext = numbertext + key +"的有" + str(value) + "站,"
        numbertext = numbertext[:-1] + "。"
        if rank_station['40度以上'] >0 :
            text = "【高温通报】 全市出现40℃以上的高温，温度较高的有："
            text = text + indextext + "其中，"  + numbertext
        elif rank_station['37度以上'] >0 :
            text = "【高温通报】 全市出现40℃以上的高温，温度较高的有："
            text = text + indextext + "其中，"  + numbertext  
        elif rank_station['35度以上'] >0 :
            text = "【高温通报】 全市出现40℃以上的高温，温度较高的有："
            text = text + indextext 
        else:
            text = ""
        return text,tmax_json
    def text_rain(self,plot_data):
            # 面雨量
        if plot_data=="none":
            orig = self.data
            data = orig.sort_values(by="rain",ascending=False)
            rain = data[(data['rain']>0) & (data['rain']<5009)]
            del rain['Unnamed: 0'] # 天擎不需要
            rain.reset_index(drop=True)
        else:
            rain = pd.read_json(json.dumps(plot_data), orient='records')
        bins=[0,30,50,80,100,150,200,250,300,400,500,800,1000,2000]
        labels=['≥0毫米','≥30毫米','≥50毫米','≥80毫米','≥100毫米','≥150毫米','≥200毫米','≥250毫米','≥300毫米','≥400毫米','≥500毫米','≥800毫米','≥1000毫米']
        bins_text = [0,5,16.9,37.9,74.5,5000]
        labels_text = ['小雨','小到中雨','中到大雨','大到暴雨','大暴雨']
        rain['rank']=pd.cut(rain['rain'],bins,right=False,labels=labels)
        rain['rank_label']=pd.cut(rain['rain'],bins_text,right=False,labels=labels_text)
        # 数据校验
        def vaild_rain(col):
            lat = col['Lat']
            lon = col['Lon']
            pre = col['rain']
            IIIII = col['IIIII']
            s = lat - 0.07
            n = lat + 0.07
            w = lon - 0.07
            e = lon + 0.07
            stdf = rain[(rain['Lat']>s)&(rain['Lat']<n)&(rain['Lon']<e)&(rain['Lon']>w)]
            stdq = stdf[(True^stdf['IIIII'].isin([IIIII]))]
            stdmean = stdq['rain'].mean()
            stds = stdq['rain'].std()
            stdabs = abs(pre - stdmean)
            if stdabs > stds*3:
                #value = stdabs - stds
                value = 1
            else:
                #value = stdabs - stds
                value = 0
            del stdf,stdq
            return value
        rain['std'] = rain.apply(vaild_rain,axis=1) 
        # 面雨量
        if self.city_code in self.city_codes:
            cnty = rain.groupby(['Cnty'])['rain'].mean().to_dict()
            
            text_average_city = "面雨量：" + str(round(rain['rain'].mean(),2)) + "毫米，" +"各市（县）面雨量如下:"
        else:
            cnty = rain.groupby(['Town'])['rain'].mean().to_dict()
            text_average_city = "各乡镇面雨量如下:"
        items = sorted(cnty.items())
        sorted_cnty = {k: v for k, v in sorted(cnty.items(), key=lambda x: x[1], reverse=True)}
        for key, value in sorted_cnty.items():
            text_average_city = text_average_city + key + ":" + str(round(value,2)) + "毫米,"
        text_average_city = text_average_city[:-1] + "。"  
        # 流域面雨量
        river =  pd.read_csv("static/data/river_tz.csv")
        river_average = {
            "永宁江流域":pd.merge(rain,river.query('Property.str.contains("YN")'),on='IIIII',how='inner'),
            "始丰溪流域":pd.merge(rain,river.query('Property.str.contains("SF")'),on='IIIII',how='inner'),
            "永安溪流域":pd.merge(rain,river.query('Property.str.contains("YA")'),on='IIIII',how='inner'),
            "牛头山流域":pd.merge(rain,river.query('Property.str.contains("NTS")'),on='IIIII',how='inner'),
            "大田流域":pd.merge(rain,river.query('Property.str.contains("DT")'),on='IIIII',how='inner'),
            "金清港流域":pd.merge(rain,river.query('Property.str.contains("JQG")'),on='IIIII',how='inner'),
            "长潭水库流域":pd.merge(rain,river.query('Property.str.contains("CT")'),on='IIIII',how='inner')
        }
        text_river = ""
        sort_river = {}
        for key, value in river_average.items():
            river_data = value
            river_ave = round(river_data['rain'].mean(),2)
            if river_ave>0:
                sort_river[key] = river_ave
        river_list = sorted(sort_river.items(),key = lambda x:x[1],reverse = True)
        for item in river_list:
            text_river = text_river + item[0] + ":" + str(item[1]) + "毫米,"  
        if len(text_river)>1:
            text_river = "【流域雨量】面雨量较大的有"+text_river[:-1] + "。"
        else:
            text_river =""    
        # 单站前十
        rain_max = rain.sort_values(by="rain",ascending=False).head(10)
        rain_json = rain.sort_values(by="rain",ascending=False).to_json(orient='records',force_ascii=False)
        indextext = "【单站雨量】雨量较大的有："
        for index,row in rain_max.iterrows():
            if self.city_code in self.city_codes:
                indextext = indextext + row['Cnty'] + row['StationName'] + str(row['rain']) + "毫米，"
            else:
                indextext = indextext + row['Town'] + row['StationName'] + str(row['rain']) + "毫米，"
        indextext = indextext[:-1] + "。"
        # 小时雨强较大的有 ------------------------------
        rain_hours_max = pd.read_csv("static/data/downfile/rain_max.csv").sort_values(by="PRE",ascending=False).head(5)
        if max(rain_hours_max['PRE'])>0.1:
            rain_hours_max_text = ""
            for index,row in rain_hours_max.iterrows():
                rain_hours_max_text = rain_hours_max_text + row['Cnty'] + row['Station_Name'] + str(row['PRE']) + "毫米，"
            rain_hours_max_text = "【小时雨强】雨强较大的有" +  rain_hours_max_text[:-1] + "。" 
        else:
            rain_hours_max_text = ""
        # 乡镇前十
        town_max = rain.groupby(['Town','Cnty'])['rain'].max().sort_values(ascending=False).head(10)
        indextext_town = "【乡镇雨量】雨量较大的有："
        for index,row in rain_max.iterrows():
            if self.city_code in self.city_codes:
                indextext_town = indextext_town + row['Cnty'] + row['StationName'] + str(row['rain']) + "毫米，"
            else:
                indextext_town = indextext_town + row['Town'] + row['StationName'] + str(row['rain']) + "毫米，"
        indextext_town = indextext_town[:-1] + "。"
        # 雨量个数统计
        rank_station = {
            '≥1000毫米':len(rain[(rain['rain']>=1000) & (rain['rain']<2000)].value_counts()),
            '≥800毫米':len(rain[(rain['rain']>=800) & (rain['rain']<1000)].value_counts()),
            '≥500毫米':len(rain[(rain['rain']>=500) & (rain['rain']<800)].value_counts()),
            '≥400毫米':len(rain[(rain['rain']>=400) & (rain['rain']<500)].value_counts()),
            '≥300毫米':len(rain[(rain['rain']>=300) & (rain['rain']<400)].value_counts()),
            '≥250毫米':len(rain[(rain['rain']>=250) & (rain['rain']<300)].value_counts()),
            '≥200毫米':len(rain[(rain['rain']>=200) & (rain['rain']<250)].value_counts()),
            '≥150毫米':len(rain[(rain['rain']>=150) & (rain['rain']<200)].value_counts()),
            '≥100毫米':len(rain[(rain['rain']>=100) & (rain['rain']<150)].value_counts()),
            '≥80毫米':len(rain[(rain['rain']>=80) & (rain['rain']<100)].value_counts()),
            '≥50毫米':len(rain[(rain['rain']>=50) & (rain['rain']<80)].value_counts()),
            '≥30毫米':len(rain[(rain['rain']>=30) & (rain['rain']<50)].value_counts()),
            '≥10毫米':len(rain[(rain['rain']>=10) & (rain['rain']<30)].value_counts()),
            '≥0毫米':len(rain[(rain['rain']>0) & (rain['rain']<10)].value_counts()),
        }
        numbertext = "其中，"
        for key, value in rank_station.items():
            if value>0 and key not in ["≥0毫米"]:
                numbertext = numbertext + key +"的有" + str(value) + "站,"
        numbertext = numbertext[:-1] + "。"
        # 雨量乡镇数统计
        town_group = rain.groupby(['Town'])['rain'].max()  
        dict_town = {'Town':town_group.index,'rain':town_group.values}
        town_range = pd.DataFrame(data =dict_town)
        town_range['rank']=pd.cut(town_range['rain'],bins,right=False,labels=labels)
        rank_station_town = {
            '≥1000毫米':len(town_range[(town_range['rain']>=1000) & (town_range['rain']<2000)].value_counts()),
            '≥800毫米':len(town_range[(town_range['rain']>=800) & (town_range['rain']<1000)].value_counts()),
            '≥500毫米':len(town_range[(town_range['rain']>=500) & (town_range['rain']<800)].value_counts()),
            '≥400毫米':len(town_range[(town_range['rain']>=400) & (town_range['rain']<500)].value_counts()),
            '≥300毫米':len(town_range[(town_range['rain']>=300) & (town_range['rain']<400)].value_counts()),
            '≥250毫米':len(town_range[(town_range['rain']>=250) & (town_range['rain']<300)].value_counts()),
            '≥200毫米':len(town_range[(town_range['rain']>=200) & (town_range['rain']<250)].value_counts()),
            '≥150毫米':len(town_range[(town_range['rain']>=150) & (town_range['rain']<200)].value_counts()),
            '≥100毫米':len(town_range[(town_range['rain']>=100) & (town_range['rain']<150)].value_counts()),
            '≥80毫米':len(town_range[(town_range['rain']>=80) & (town_range['rain']<100)].value_counts()),
            '≥50毫米':len(town_range[(town_range['rain']>=50) & (town_range['rain']<80)].value_counts()),
            '≥30毫米':len(town_range[(town_range['rain']>=30) & (town_range['rain']<50)].value_counts()),
            '≥10毫米':len(town_range[(town_range['rain']>=10) & (town_range['rain']<30)].value_counts()),
            '≥0毫米':len(town_range[(town_range['rain']>0) & (town_range['rain']<10)].value_counts()),
        }
        numbertext_town = "其中，"
        for key, value in rank_station_town.items():
            if value>0 and key not in ["≥0毫米"]:
                numbertext_town = numbertext_town + key +"的有" + str(value) + "站,"
        numbertext_town = numbertext_town[:-1] + "。"
        # 统计全市雨情
        rank_text = {
            '大暴雨':sum(rain[rain['rank_label']=='大暴雨']['rain']),
            '大到暴雨':sum(rain[rain['rank_label']=='大到暴雨']['rain']),
            '中到大雨':sum(rain[rain['rank_label']=='中到大雨']['rain']),
            '小到中雨':sum(rain[rain['rank_label']=='小到中雨']['rain']),
            '小雨':sum(rain[rain['rank_label']=='小雨']['rain'])
        }
        res = max(rank_text, key=lambda x: rank_text[x])
        if max(rank_text, key=lambda x: rank_text[x])=="大暴雨":
            text = "【雨情通报】 全市出现" + res +"。" + text_average_city + "<br>" + indextext + numbertext + "<br>" + rain_hours_max_text + "<br>" + indextext_town + numbertext_town + "<br>" + text_river
        elif max(rank_text, key=lambda x: rank_text[x])=="大到暴雨":
            text = "【雨情通报】 全市出现" + res +"。" + text_average_city + "<br>" + indextext + numbertext + "<br>" + rain_hours_max_text + "<br>" + indextext_town + numbertext_town + "<br>" + text_river
        elif max(rank_text, key=lambda x: rank_text[x])=="中到大雨":
            text = "【雨情通报】 全市出现" + res +"。" + text_average_city + "<br>" + indextext + numbertext + "<br>" + rain_hours_max_text + "<br>" + indextext_town + numbertext_town + "<br>" + text_river
        elif max(rank_text, key=lambda x: rank_text[x])=="小到中雨":
            text = "【雨情通报】 全市出现" + res +"。" + text_average_city + "<br>" + indextext + numbertext + "<br>" + rain_hours_max_text + "<br>" + indextext_town + numbertext_town + "<br>" + text_river
        elif max(rank_text, key=lambda x: rank_text[x])=="小雨":
            text = "【雨情通报】 全市出现" + res +"。" + text_average_city + "<br>" + indextext + numbertext + "<br>" + rain_hours_max_text + "<br>" + indextext_town + numbertext_town + "<br>" + text_river
        return text,rain_json
    def main(self):
        text = ""
        plot_data = "none"
        text_rain,rain_json = self.text_rain(plot_data)
        text_wind,wind_json = self.text_wind(plot_data)
        text_tmax,tmax_json = self.text_tmax(plot_data)
        text_view,view_json = self.text_view(plot_data)
        text = text + text_rain + "<br>" + text_wind + "<br>" + text_tmax + "<br>" + text_view
        return text,rain_json,wind_json,tmax_json,view_json
    def remain(self,plot_rain,plot_wind,plot_tmax,plot_view):
        text = ""
        text_rain,rain_json = self.text_rain(plot_rain)
        text_wind,wind_json = self.text_wind(plot_wind)
        text_tmax,tmax_json = self.text_tmax(plot_tmax)
        text_view,view_json = self.text_view(plot_view)
        text = text + text_rain + "<br>" + text_wind + "<br>" + text_tmax + "<br>" + text_view
        return text,rain_json,wind_json,tmax_json,view_json
    def plot_rain(self,plot_type,plot_data):
        if plot_type =="none":
            orig = self.data
            data = orig.sort_values(by="rain",ascending=False)
            rain = data[(data['rain']>0) & (data['rain']<5009)]
        else:  
            rain = pd.read_json(json.dumps(plot_data), orient='records')
        lat = np.array(rain['Lat'].to_list())
        lon = np.array(rain['Lon'].to_list())
        Zi = np.array(rain['rain'].to_list())
        data_max = max(Zi)
        data_min = min(Zi)
        np.set_printoptions(precision = 2)
        x = np.arange(120.0,122.0,0.015)
        y = np.arange(27.8,29.5,0.015)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        gauss_kernel = Gaussian2DKernel(0.1)
        smoothed_data_gauss = convolve(Z_linear, gauss_kernel)
        data_xr = xr.DataArray(Z_linear, coords=[ y,x], dims=["lat", "lon"])
        lat = data_xr.lat
        lon = data_xr.lon
        lons, lats = np.meshgrid(lon, lat)
        start_time = dtt.datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
        end_time = dtt.datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
        hours = (end_time-start_time).total_seconds()//3600
        if hours >12:
            colorslist = ['#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 24降水
            levels = [0, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250, 350, 500]
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=15)
            cmap_nonlin = nlcmap(cmaps, levels)
        elif hours <12 and hours >=1:
            colorslist =['#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 06降水
            levels = [0, 2, 5, 10, 15, 20, 25, 35, 50, 60, 70, 80, 90, 100, 110]
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=15)
            cmap_nonlin = nlcmap(cmaps, levels)
        elif hours <1:
            colorslist =['#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 01降水
            levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 17, 20]
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=15)
            cmap_nonlin = nlcmap(cmaps, levels) 
        contour = plt.contourf(lons,lats,data_xr,cmap=cmap_nonlin,levels =levels)
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contour,
            ndigits=3,
            unit='mm'
        )
        plt.close()
        return geojson,hours

# redis数据轮询问
class station_sql_data:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        self.redis_name = {
            "24hours":"table_24hours",
            "12hours":"table_12hours",
            "6hours":"table_6hours",
            "3hours":"table_3hours",
            "2hours":"table_2hours",
            "1hours":"table_1hours",
            "45mins":"table_45mins",
            "30mins":"table_30mins",
            "15mins":"table_15mins"         
        }
    def rain_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,sum(Ri) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.Ri!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_rain"
        self.rs.set(redis_name_str, pickle.dumps(data))
        
    def wind_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select wind1.IIiii as IIiii,wind1.StationName,wind1.Province,wind1.City,wind1.County,wind1.Town,wind1.ZoomLevel,wind1.Type, max(wind1.lon) as lon, max(wind1.lat) as lat, wind1.fFy as value, max(wind2.dFy) as dFy  from 
        (SELECT  st.IIiii AS  IIiii,StationName,Province,City,County,Town,ZoomLevel,Type, max(st.lon) AS  lon, max(st.lat) AS  lat, max(sd.fFy) AS  fFy  FROM  TAB_StationInfo  AS  st
        LEFT JOIN Tab_AM_M  AS  sd ON  sd.IIIII = st.IIiii WHERE  (st.Province='浙江' AND  (sd.TTime >'{time}') and sd.fFy!=-9999 and sd.dFy!=-9999 ) 
        GROUP BY  st.IIiii,st.StationName,st.Province,st.City,st.County,st.Town,st.ZoomLevel,st.Type) as wind1
        inner join Tab_AM_M as wind2 on wind2.IIIII = wind1.IIiii and wind2.fFy = wind1.fFy and wind2.TTime >'{time}'
        group by wind1.IIiii, wind1.fFy,StationName,Province,City,County,Town,ZoomLevel,Type"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_wind"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def tmax_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,max(T) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.T!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_tmax"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def tmin_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql ="""select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,min(T) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.T!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_tmin"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def view_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,min(V) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and ta.V!=-9999 ) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_view"
        self.rs.set(redis_name_str, pickle.dumps(data))

# 雷达类
class radar_data:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
    def to_redis(self,imd):
        if self.rs.get("radar"):
            data = pickle.loads(self.rs.get("radar"))
            data['imglist'].append(imd)
        else:
            imglist = [imd]
            data = {
                "imglist":imglist
            }
        self.rs.set("radar", pickle.dumps(data))       
    def plot_data(self):
        path ="static/data/downfile/" 
        f = cinrad.io.CinradReader(path+'Z_RADR_I_Z9576_20150809120400_O_DOR_SA_CAP.bin.bz2')
        tilt_number = 0
        data_radius = 230
        data_dtype = 'REF' # stands for reflectivity
        # 数据加载核心
        ra = f.get_data(tilt_number, data_radius, data_dtype)
        rl = list(f.iter_tilt(230, 'REF'))
        cr = cinrad.easycalc.quick_cr(rl)
        data = ra.data
        m = Basemap(llcrnrlon=119.16,llcrnrlat=26.55,urcrnrlon=123.88,urcrnrlat=30.69)
        ## 颜色 ------
        colorslist = ['#00C800','#019000','#FFFF00','#E7C000','#FF9000','#D60000','#C00000','#FF00F0','#780084','#AD90F0','#AE0AF5']# 组合反射率
        levels1 = [15,20,25,30,35,40,45,50,55,60,65,70]
        cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=11)
        plt.contourf(cr.lon, cr.lat, cr.data,cmap=cmaps,levels = levels1)
        plt.axis('off') 
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight',transparent=True)  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        self.to_redis(imd)
        self.to_vcs(rl)
    def get_radar(self):
        data = pickle.loads(self.rs.get("radar"))
        return data
    def to_vcs(self,rl):
        vcs = cinrad.calc.VCS(rl)
        data = {
            "vcs":vcs
        }
        self.rs.set("radar_vsc", pickle.dumps(data))
    def get_vcs(self):
        data = pickle.loads(self.rs.get("radar_vsc"))
        return data['vcs']
    def plot_sec(self,start,end):
        #fig = plt.figure(figsize=(16, 8))
        vcs = self.get_vcs()
        sec = vcs.get_section(start_cart=(start[0], start[1]), end_cart=(end[0], end[1])) 
        Section(sec)
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight')  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        plt.close()
        return imd

# 数据加载    
class server_plot():
    def __init__(self,time_hours,city,plot_type,js_status,recv_data):
        self.start = "2023-09-18 19:30:25"
        self.end = "2023-09-20 19:30:25"
        self.time_hours = float(time_hours)
        self.city = city
        self.plot_type = plot_type
        self.js_status = js_status
        self.recv_data = recv_data
        self.max = None
        self.min = None
        #self.rs = redis.Redis(host='127.0.0.1', port=6379,password="tzqxj58660")
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        self.userId = "BEHZ_TZSJ_TZSERVICE" 
        self.pwd = "Liyuan3970!@" 
        self.dataFormat = "json"
        self.rain = None
        self.wind = None
    def data_from_js(self):
        data = pd.read_json(json.dumps(self.recv_data), orient='records')
        self.recv_data = data
        return data
    def wind_from_sql(self):
        start_time = dtt.datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
        end_time = dtt.datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
        offset = dtt.timedelta(minutes=(-60*8))
        now = (end_time+offset).strftime('%Y-%m-%d %H:%M:%S')
        old = (start_time+offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(wind) as wind
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and wind>0 and City='台州市'
            group by Station_Id_C""" 
        rsql = sql.format(start_times=old,end_times=now)
        data = pd.read_sql(rsql, con=self.conn)
        data['WIN_S_Gust_Max'] = data.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
        data['WIN_D_Gust_Max'] = data.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
        data['value'] = data['WIN_S_Gust_Max']
        self.city_info[self.city]['data']['平均'] = round(data['WIN_S_Gust_Max'].mean(),1)
        self.city_info[self.city]['data']['最大'] = str(round(max(data['WIN_S_Gust_Max']),1))      
        return data
    def temp_from_sql(self):
        start_time = dtt.datetime.strptime(self.start, "%Y-%m-%d %H:%M:%S")
        end_time = dtt.datetime.strptime(self.end, "%Y-%m-%d %H:%M:%S")
        offset = dtt.timedelta(minutes=(-60*8))
        now = (end_time+offset).strftime('%Y%m%d%H%M')+"00"
        old = (start_time+offset).strftime('%Y%m%d%H%M')+"00"
        label = "["+old+","+now+"]"
        client = DataQueryClient(configFile=r"/home/workspace/Data/My_Git/web_met/zdz/common/utils/client.config")
        interfaceId = "statSurfEleInRegion"
        params = {
            'dataCode':"SURF_CHN_MUL_MIN",  #SURF_CHN_MUL_HOR
            'elements':"Cnty,Province,Town,Station_levl,Station_Name,City,Station_Id_C,Lat,Lon,Alti",
            'statEles':'MAX_TEM,MIN_TEM',
            'timeRange':label,
            'adminCodes':"331000",#330000 浙江省
            'eleValueRanges':"TEM:(0,10000)",
            
            'limitCnt':"100000000"
        }
        result = client.callAPI_to_serializedStr(self.userId, self.pwd, interfaceId, params, self.dataFormat)
        result_json = json.loads(result)
        clomns =['Cnty','Province','Town','Station_levl','Station_Name','City','Station_Id_C','Lat','Lon','Alti','tmax','tmin']
        data = pd.DataFrame(result_json['DS'])
        data.columns = clomns
        data = data.astype({'Lat': 'float', 'Lon': 'float','Station_levl':'int','Alti':'float','tmax':'float','tmin':'float'})
        if self.plot_type=="tmax":
            data['value'] = data['tmax']
            self.min = data['value'].min()
            self.max = data['value'].max()
        else:
            data['value'] = data['tmin']
            self.min = data['value'].min()
            self.max = data['value'].max()
        return data
    def rain_from_cloud(self):
        data = pd.read_csv("static/data/downfile/rain.csv")
        data['value'] = data['PRE']
        self.rain = data
        return data
    def color_map(self):
        if self.plot_type=="rain":
            hours = self.time_hours
            if hours >12:
                colorslist = ['#FFFFFF','#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 24降水
                levels = [-1,0.01, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250, 350, 500]
                cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
                norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引
            elif hours <=12 and hours >=1:
                colorslist =['#FFFFFF','#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 06降水
                levels = [-1,3, 4, 5, 10, 15, 20, 25, 35, 50, 60, 70, 80, 90, 100, 110]
                cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
                norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引       
            elif hours <1:
                colorslist =['#FFFFFF','#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 01降水
                levels = [-1,0.01, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 15, 17, 20]
                cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
                norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引 
        elif self.plot_type=="wind":
            colorslist = ['#FFFFFF','#CED9FF','#9CFFFF','#FFFF9C','#FFCF9C','#FF9E63','#FF6131','#FF3031','#CE0000']
            levels = [0,0.3,1.6,3.4,5.5,8,10.8,13.9,17.2,28]
            #colorslist = ['#FFFFFF','#CED9FF','#9CFFFF',"#42F217","#FF9E63","#DF16EE","red"]# 风力
            #levels = [0,1.6,3.4,5.5,13.9,17.3,32.6,56]
            cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
            norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引 
        elif self.plot_type=="tmax":
            colorslist = ['#0524B7','#092CD2','#0B34F4','#3859F7','#7187F0','#AAB6F3','#C9D1F8','#F8C9CB','#F19599','#F7797D','#F3464D','#F20710','#92080D','#650307']
            level = list(np.linspace(self.min-1, self.max+1, num=14, endpoint=True, retstep=False, dtype=None))
            levels = [round(i,1) for i in level]
            cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
            norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引
        elif self.plot_type=="tmin":
            colorslist = ['#FFFFFF','#CED9FF','#9CFFFF','#FFFF9C','#FFCF9C','#FF9E63','#FF6131','#FF3031','#CE0000']
            level = list(np.linspace(self.min-1, self.max+1, num=14, endpoint=True, retstep=False, dtype=None))
            levels = [round(i,1) for i in level]
            cmap_nonlin = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
            norm = mpl.colors.BoundaryNorm(levels, cmap_nonlin.N)  # 基于离散区间生成颜色映射索引  
        return cmap_nonlin,levels
    def decode_xarray(self):
        if self.plot_type=="rain":
            if self.js_status:      
                data = self.data_from_js()
            else:
                data = self.rain_from_cloud()
                self.recv_data = data
        elif self.plot_type=="wind":
            if self.js_status:
                data = self.data_from_js()
            else:
                data = self.wind_from_sql()
                self.recv_data = data
        elif self.plot_type=="tmax" or self.plot_type=="tmin":
            if self.js_status:
                data = self.data_from_js()
            else:
                data = self.temp_from_sql() 
                self.recv_data = data
        lat = np.array(data['Lat'].to_list())
        lon = np.array(data['Lon'].to_list())
        Zi = np.array(data['value'].to_list())
        data_max = max(Zi)
        data_min = min(Zi)
        self.max = data_max
        self.min = data_min
        np.set_printoptions(precision = 2)
        x = np.arange(120.0,122.0,0.01)
        y = np.arange(27.8,29.5,0.01)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        gauss_kernel = Gaussian2DKernel(0.8)
        smoothed_data_gauss = convolve(Z_linear, gauss_kernel)
        data_xr = xr.DataArray(smoothed_data_gauss, coords=[ y,x], dims=["lat", "lon"])
        return data_xr
    def return_mark(self):
        '''主要返回风场位置'''
        if self.plot_type=="wind":
            mark = self.recv_data
            mark_data = mark[(mark['value']>0)&(mark['City']=="台州市")]
            mark_json = mark_data.to_json(orient='records',force_ascii=False)
        else:
            mark_json = self.recv_data.to_json(orient='records',force_ascii=False)
        return mark_json
    def wind_to_json(self):
        data = pd.read_csv("static/data/downfile/min.csv")
        data = data.astype({'Lat': 'float', 'Lon': 'float','PRE': 'float','WIN_S_Gust_Max': 'float', 'WIN_D_Gust_Max': 'float'})
        # data['WIN_S_Gust_Max'] = data['WIN_S_Inst_Max']
        # data['WIN_D_Gust_Max'] = data['WIN_D_INST_Max']
        data['value'] = data['WIN_S_Gust_Max']
        # wind = data[(data['value']>15)&(data['value']<100)&(data['City']=="台州市")]
        wind = data[(data['value']>15)&(data['value']<100)]
        self.wind = wind
        return wind
    def plot_rain(self):
        wind = self.wind_to_json()
        wind_json = wind.to_json(orient = "records", force_ascii=False)
        data_xr = self.decode_xarray()
        cmaps ,norm = self.color_map()
        lat = data_xr.lat
        lon = data_xr.lon
        lons, lats = np.meshgrid(lon, lat)
        contour = plt.contourf(lons,lats,data_xr,cmap=cmaps,levels = norm)
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contour,
            ndigits=3,
            unit='mm'
        )
        plt.close()
        return geojson,wind_json    
    def return_geojson(self):
        data_xr = self.decode_xarray()
        # ##########色标和大小#############################
        cmaps ,levels = self.color_map()
        lat = data_xr.lat
        lon = data_xr.lon
        lons, lats = np.meshgrid(lon, lat)
        # contour = plt.contourf(lons,lats,data_xr,cmap=cmaps,levels =levels)
        contour = plt.contourf(lons,lats,data_xr,cmap=cmaps,levels = levels)
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contour,
            ndigits=3,
            unit='mm'
        )
        plt.close()
        return geojson
    def text_wind_rain(self):
        wind = self.wind
        rain = self.rain
        text = ""
        # rain = pd.read_csv("rian1.csv")
        # wind = pd.read_csv("wind1.csv")
        if self.time_hours ==0.1:   
            text_rain = "目前:"
            rain_tz = rain[rain['City']=="台州市"]
            bins = [0,5,10,100]
            labels = [0,1,2]
            rain_tz['rank']=pd.cut(rain_tz['PRE'],bins,right=False,labels=labels)
            rain_max = rain_tz.sort_values(by="PRE",ascending=False).head(3)
            raintop = rain_max.head(1)
            if raintop['PRE'].values[0]>0: 
                text_rain = text_rain + raintop['Cnty'].values[0] + raintop['Town'].values[0] + "十分钟降水为" + str(raintop['PRE'].values[0]) +"毫米。"
                if len(rain_tz[rain_tz['rank']==1])>0:
                    indextext = "单站雨量较大的有："
                    for index,row in rain_max.iterrows():
                        if row['PRE']>0:
                            indextext = indextext + row['Cnty'] + row['Station_Name'] + str(row['PRE']) + "毫米，"
                    indextext = indextext[:-1] + "。"
                    text_rain = text_rain + indextext
                    if len(rain_tz[rain_tz['rank']==2])>0:
                        text_rain = text_rain + "其中超过10毫米的站有" + str(len(rain_tz[rain_tz['rank']==2])) + "个；"          
            else:
                text_rain = ""
        elif self.time_hours ==1:
            text_rain = "目前:"
            rain_tz = rain[rain['City']=="台州市"]
            bins = [0,10,20,50,80,100,5000]
            labels = [0,1,2,3,4,5]
            rain_tz['rank']=pd.cut(rain_tz['PRE'],bins,right=False,labels=labels)
            rain_max = rain_tz.sort_values(by="PRE",ascending=False).head(3)
            raintop = rain_max.head(1)
            town_max = rain_tz.groupby(['Town','Cnty'])['PRE'].max().sort_values(ascending=False).head(3).to_frame() 
            if raintop['PRE'].values[0]>0: 
                text_rain = text_rain + raintop['Cnty'].values[0] + raintop['Town'].values[0] + "近一小时降水为" + str(raintop['PRE'].values[0]) +"毫米。"
                if len(rain_tz[rain_tz['rank']>=0])>0:
                    indextext = "单站雨量较大的有："
                    for index,row in rain_max.iterrows():
                        if row['PRE']>0:
                            indextext = indextext + row['Cnty'] + row['Station_Name'] + str(row['PRE']) + "毫米，"    
                    indextext = indextext[:-1] + "。"
                    text_rain = text_rain + indextext
                    if len(rain_tz[rain_tz['rank']>1])>0:
                        indextext_town = "单站雨量较大的乡镇有：" 
                        for index,row in town_max.iterrows():
                            if row['PRE']>0:
                                indextext_town = indextext_town + index[1] + index[0] + str(row['PRE']) + "毫米," 
                        indextext_town = indextext_town[:-1] + "。"
                        text_rain = text_rain + indextext_town                            
        elif self.time_hours ==3:
            text_rain = "目前:"
            rain_tz = rain[rain['City']=="台州市"]
            bins = [0,30,50,80,100,5000]
            labels = [0,1,2,3,4]
            rain_tz['rank']=pd.cut(rain_tz['PRE'],bins,right=False,labels=labels)
            rain_max = rain_tz.sort_values(by="PRE",ascending=False).head(3)
            raintop = rain_max.head(1)
            town = rain_tz.groupby(['Town','Cnty'])['PRE'].mean().head(5).sort_values(ascending=False).to_frame() 
            if raintop['PRE'].values[0]>0: 
                text_rain = text_rain + raintop['Cnty'].values[0] + raintop['Town'].values[0] + "近三小时降水为" + str(raintop['PRE'].values[0]) +"毫米。"
                if raintop['PRE'].values[0]>20:
                    indextext_town = "乡镇面雨量较大的有：" 
                    for index,row in town.iterrows():
                        if row['PRE']>0:
                            indextext_town = indextext_town + index[1] + index[0] + str(round(row['PRE'],1)) + "毫米," 
                    indextext_town = indextext_town[:-1] + "。"
                    text_rain = text_rain + indextext_town 
                    if raintop['PRE'].values[0]>50:
                        town_max_text = "其中，出现暴雨及以上的乡镇有："
                        town_max = rain_tz.groupby(['Town','Cnty'])['PRE'].max().sort_values(ascending=False).to_frame()
                        town_nums = town_max[town_max['PRE']>=50]
                        for index,row in town_nums.iterrows():
                            town_max_text = town_max_text + index[1] + index[0] + ","
                        town_max_text = town_max_text[:-1] + "。" 
                        text_rain = text_rain + town_max_text 
        else: 
            rain_tz = rain[rain['City']=="台州市"]
            city_mean = round(rain_tz[rain_tz['PRE']>0]['PRE'].mean(),2)
            bins = [0,30,50,80,100,5000]
            labels = [0,1,2,3,4]
            rain_tz['rank']=pd.cut(rain_tz['PRE'],bins,right=False,labels=labels)
            rain_max = rain_tz.sort_values(by="PRE",ascending=False).head(3)
            raintop = rain_max.head(1)
            average = rain_tz.groupby(['Cnty'])['PRE'].mean().to_frame().sort_values(by="PRE",ascending=False)
            text_rain = "目前:全市面雨量为" + str(city_mean) + "毫米。"
            text_cnty_average = "雨量较大的有："
            for index,row in average.iterrows():
                if row['PRE']>0:
                    text_cnty_average = text_cnty_average + index + str(round(row['PRE'],2)) + "毫米,"  
            text_cnty_average = text_cnty_average[:-1] + "。"
            text_rain = text_rain + text_cnty_average
            if raintop['PRE'].values[0]>0: 
                text_rain = text_rain +"其中，单站累计最大"+ raintop['Cnty'].values[0] + raintop['Town'].values[0] + str(raintop['PRE'].values[0]) +"毫米。"        
        # 风力的统计
        wind = wind[wind['value']>10].sort_values(by="value",ascending=False)
        wind_text = ""
        if len(wind)>0:
            wind_text = wind_text + "全市出现8级以上大风，风力较大的有"
            text_maxwind = ""
            if len(wind)>3:
                wind = wind.head(3)
            else:
                wind = wind
            for index,row in wind.iterrows():
                text_maxwind = text_maxwind + row['Cnty'] + row['Town'] + "-"+ row['Station_Name'] + str(row['value']) +"m/s,"
            text_maxwind = text_maxwind[:-1] + "。"
            wind_text =  wind_text + text_maxwind  
        text = text + text_rain + wind_text
        return text
    def return_province(self):
        data = self.rain_from_cloud()
        lat = np.array(data['Lat'].to_list())
        lon = np.array(data['Lon'].to_list())
        Zi = np.array(data['value'].to_list())
        np.set_printoptions(precision = 2)
        x = np.arange(118.0,123.0,0.01)
        y = np.arange(26,31,0.01)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        gauss_kernel = Gaussian2DKernel(0.8)
        smoothed_data_gauss = convolve(Z_linear, gauss_kernel)
        data_xr = xr.DataArray(smoothed_data_gauss, coords=[ y,x], dims=["lat", "lon"])
        cmaps ,norm = self.color_map()
        lat = data_xr.lat
        lon = data_xr.lon
        lons, lats = np.meshgrid(lon, lat)
        contour = plt.contourf(lons,lats,data_xr,cmap=cmaps,levels = norm)
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contour,
            ndigits=3,
            unit='mm'
        )
        plt.close()
        # 风力
        wind = self.wind_to_json()
        wind_json = wind.to_json(orient = "records", force_ascii=False)
        # 文字
        rain = data[data['Province']=="浙江省"]
        rain_max = rain.sort_values(by="PRE",ascending=False).head(3)
        raintop = rain_max.head(1)
        text_rain = ""
        if raintop['PRE'].values[0]>0: 
            text_rain = text_rain +"目前" + raintop['City'].values[0] + raintop['Cnty'].values[0] + raintop['Town'].values[0] +raintop['Station_Name'].values[0] +"出现"+ str(raintop['PRE'].values[0]) +"毫米的降水。"
        average = rain.groupby(['City'])['PRE'].mean().to_frame().sort_values(by="PRE",ascending=False)
        text_cnty_average = "各市面雨量较大的有："
        for index,row in average.iterrows():
            if row['PRE']>0:
                text_cnty_average = text_cnty_average + index + str(round(row['PRE'],2)) + "毫米,"  
        text_cnty_average = text_cnty_average[:-1] + "。"
        text_rain = text_rain + text_cnty_average
        wind = wind[wind['value']>10].sort_values(by="value",ascending=False)
        text = ""
        wind_text = ""
        if len(wind)>0:
            wind_text = wind_text + "全省出现8级以上大风，风力较大的有"
            text_maxwind = ""
            if len(wind)>3:
                wind = wind.head(3)
            else:
                wind = wind
            for index,row in wind.iterrows():
                text_maxwind = text_maxwind + row['Cnty'] + row['Town'] + "-"+ row['Station_Name'] + str(row['value']) +"m/s,"
            text_maxwind = text_maxwind[:-1] + "。"
            wind_text =  wind_text + text_maxwind  
        text = text + text_rain + wind_text
        return geojson,text,wind_json



# 日历
class clander:
    def __init__(self,city,click_type):
        self.city = city
        self.type = click_type
    def get_clander(self):
        data = pd.read_csv("static/data/downfile/clander.csv")
        data_json = data.to_json(orient = "values")
        return data_json
    def get_daily(self):
        data = pd.read_csv("static/data/downfile/daily.csv")
        data = data[(data['rain']<5000)&(data['Datetime']=='2023-10-01 23:00:00')]
        return data
    def plot_rain(self):
        data = self.get_daily()
        data['value'] = data['rain']
        lat = np.array(data['Lat'].to_list())
        lon = np.array(data['Lon'].to_list())
        Zi = np.array(data['value'].to_list())
        np.set_printoptions(precision = 2)
        x = np.arange(118.0,123.0,0.01)
        y = np.arange(26,31,0.01)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        gauss_kernel = Gaussian2DKernel(0.8)
        smoothed_data_gauss = convolve(Z_linear, gauss_kernel)
        data_xr = xr.DataArray(smoothed_data_gauss, coords=[ y,x], dims=["lat", "lon"])
        colorslist = ['#FFFFFF','#A3FAFD', '#29D3FD', '#29AAFF', '#2983FF', '#4EAB37', '#46FA35', '#F1F837', '#F1D139', '#F2A932', '#F13237', '#C4343A', '#A43237', '#A632B4', '#D032E1', '#E431FF']# 24降水
        levels = [-1,0.01, 5, 10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 250, 350, 500]
        cmaps = mpl.colors.ListedColormap(colorslist)  # 自定义颜色映射 color-map
        norm = mpl.colors.BoundaryNorm(levels, cmaps.N)  # 基于离散区间生成颜色映射索引
        lat = data_xr.lat
        lon = data_xr.lon
        lons, lats = np.meshgrid(lon, lat)
        contour = plt.contourf(lons,lats,data_xr,cmap = cmaps,norm = norm ,levels = levels)
        geojson = geojsoncontour.contourf_to_geojson(
            contourf=contour,
            ndigits=3,
            unit='mm'
        )
        plt.close()
        # 风力
        # wind = data
        wind = data[data['wind']>0]
        wind['WIN_S_Gust_Max'] = wind.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
        wind['WIN_D_Gust_Max'] = wind.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
        wind_json = wind.to_json(orient = "records", force_ascii=False)
        return geojson,wind_json
    def return_text(self):
        data = self.get_daily()
        rain_max = data.sort_values(by="rain",ascending=False).head(3)
        raintop = rain_max.head(1)
        text_rain = ""
        if raintop['rain'].values[0]>0: 
            text_rain = text_rain +"近24小时内" + raintop['City'].values[0] + raintop['Cnty'].values[0] + raintop['Town'].values[0] +raintop['Station_Name'].values[0] +"出现"+ str(raintop['rain'].values[0]) +"毫米的降水。"
        average = data.groupby(['Cnty'])['rain'].mean().to_frame().sort_values(by="rain",ascending=False)
        text_cnty_average = "各市面雨量较大的有："
        for index,row in average.iterrows():
            if row['rain']>0:
                text_cnty_average = text_cnty_average + index + str(round(row['rain'],2)) + "毫米,"  
        text_cnty_average = text_cnty_average[:-1] + "。"
        text_rain = text_rain + text_cnty_average
        wind = data[data['wind']>0]
        wind['WIN_S_Gust_Max'] = wind.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
        wind['WIN_D_Gust_Max'] = wind.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
        wind['value'] = wind['WIN_S_Gust_Max'] 
        wind = wind[wind['value']>10].sort_values(by="value",ascending=False)
        text = ""
        wind_text = ""
        if len(wind)>0:
            wind_text = wind_text + "全市出现8级以上大风，风力较大的有"
            text_maxwind = ""
            if len(wind)>3:
                wind = wind.head(3)
            else:
                wind = wind
            for index,row in wind.iterrows():
                text_maxwind = text_maxwind + row['Cnty'] + row['Town'] + "-"+ row['Station_Name'] + str(row['value']) +"m/s,"
            text_maxwind = text_maxwind[:-1] + "。"
            wind_text =  wind_text + text_maxwind  
        text = text + text_rain + wind_text
        return text


# 报警程序的模块
class warring_alert():
    def __init__(self,rain_type):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
#         self.rs = redis.Redis(host='127.0.0.1', port=6379,password="tzqxj58660")
#         self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="tzqxj58660",db="ZJSZDZDB")
#         self.con = pymssql.connect("172.21.158.201","down","downx","ZJSZDZDB")
        self.userId = "BEHZ_TZSJ_TZSERVICE" 
        self.pwd = "Liyuan3970!@" 
        self.dataFormat = "json"   
        self.rain_type = rain_type
        self.now = self.get_latest()
    def get_rain(self):
        rain_dir = {
            "10min":"warring_zdz",
            "rain01":"rain01",
            "rain03":"rain03",
            "rain12":"rain12",
            "rain24":"rain24"        
        }
        if self.rain_type =="10min":
            data = self.now
            rain = data[(data['PRE']>0)&(data['PRE']<5000)][['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','PRE']]
            rain_data = rain.groupby(['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti'])['PRE'].sum().reset_index()
        else:
            data = pd.read_csv("/workspace/liyuan3970/Data/My_Git/web_met/static/data/downfile/rain01.csv")
            rain_data = data
        return rain_data
    def get_latest(self):
        data = pd.read_csv("/workspace/liyuan3970/Data/My_Git/web_met/static/data/downfile/server.csv")
        data = data.astype({'Lat': 'float', 'Lon': 'float','PRE': 'float','WIN_S_Inst_Max': 'float', 'WIN_D_INST_Max': 'float','TEM':'float','VIS_HOR_1MI':'float'})
        return data         
    def get_wind(self):
        data = self.now
        wind = data[(data['WIN_S_Inst_Max']>17)&(data['WIN_S_Inst_Max']<5000)][['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','WIN_S_Inst_Max','WIN_D_INST_Max']]
        wind_data = wind.groupby(['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','WIN_D_INST_Max'])['WIN_S_Inst_Max'].max().reset_index().sort_values('WIN_S_Inst_Max', ascending=False).drop_duplicates(subset=['Station_Id_C'], keep='first')
        return wind_data
    def get_temp(self):
        data = self.now
        tmin = data[(data['TEM']<0)&(data['TEM']<5000)][['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','TEM']]
        tmax = data[(data['TEM']>10)&(data['TEM']<5000)][['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','TEM']]
        tmax_data = tmax.groupby(['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti'])['TEM'].max().reset_index().to_json(orient = "records", force_ascii=False)
        tmin_data = tmin.groupby(['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti'])['TEM'].min().reset_index().to_json(orient = "records", force_ascii=False)
        return tmax_data,tmin_data
    def get_view(self):
        data = self.now
        view = data[(data['VIS_HOR_1MI']<1000)&(data['VIS_HOR_1MI']<30000)][['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti','VIS_HOR_1MI']]
        view_data = view.groupby(['Cnty','Province','Town','Station_Name','City','Station_Id_C','Lat','Lon','Alti'])['VIS_HOR_1MI'].min().reset_index().to_json(orient = "records", force_ascii=False)
        return view_data
    def get_radar(self):
        '''获取雷达数据'''
        img = pickle.loads(self.rs.get("radar"))
        return img
    def warring_data(self):
        # 开始编写风雨数据模型
#         radar = self.get_radar()
        rain = self.get_rain()
        wind = self.get_wind()
        tmax_data,tmin_data = self.get_temp()
        view_data = self.get_view()
        rain_data = rain.to_json(orient = "records", force_ascii=False)
        wind_data = wind.to_json(orient = "records", force_ascii=False)
        return rain_data,wind_data,tmax_data,tmin_data,view_data
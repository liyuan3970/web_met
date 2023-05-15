import pymysql
import pymssql 
import numpy as np
import pandas as pd
import pickle
import redis
import datetime as dtt
from datetime import timezone
class station_zdz:
    def __init__(self, start, end):
        self.start = start
        self.end = end
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
        start = '2019-08-07 20:53:00'
        end = '2019-08-08 20:53:00' 
        yesday = start[0:10] + " 20:00:00"
        today = end[0:10] + " 20:00:00"
        hours = dtt.datetime.strptime(end,'%Y-%m-%d %H:%M:%S').hour
        # 数据库中读取单站数据并解析
        sql = """select tTime,ta.IIiii,station.StationName,station.Province,station.City,station.County,station.Town,station.lat,station.lon,Ri,T,V,fFy,dFy 
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIiii=station.IIiii and station.Province ='浙江' 
        where (ta.IIiii='{station}' and tTime between '{start_time}' and '{end_time}')  order by tTime  """
        rsql = sql.format(start_time=start,end_time=end,station=station)
        data = pd.read_sql(rsql , con=self.conn)
        if hours>=20:
            # 实时数据
            now_data = data[data['tTime']>=today]
            now_data = now_data[now_data['tTime']<=end]
            # 历史数据
            his_data = data[data['tTime']>start]
            his_data = his_data[his_data['tTime']<=today]   
        else:
            # 实时数据
            now_data = data[data['tTime']>=yesday]
            now_data = now_data[now_data['tTime']<=end]
            # 历史数据
            his_data = data[data['tTime']>start]
            his_data = his_data[his_data['tTime']<=yesday]
        # 开始筛选数据种类
        nul_now = [0 for i in range(len(now_data['T'].to_list()))]    
        nul_his =  [0 for i in range(len(his_data['T'].to_list()))]
        value_now =  now_data['T'].to_list()  + nul_his
        value_his = nul_now + his_data['T'].to_list() 
        # 解析数据成两个序列
        return data
    def upload2_redis_Minutes(self):
        '''根据date_type向redis中传输数据'''
        table = 'Tab_AM_M'
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        sql = """select tTime,ta.IIiii,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,Ri,T,V,fFy,dFy 
        from {table} as ta inner join TAB_StationInfo as station on ta.IIiii=station.IIiii and station.Province ='浙江' 
        where (tTime > '{time}') order by tTime """
        # 数据加载
        data_class = "table_minutes"
        redis_data = self.get_redis(data_class)
        if not redis_data:
            # 当redis无数据时
            #time = today.strftime('%Y-%m-%d %H:%M:%S')
            offset = dtt.timedelta(seconds=-120)
            time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
            # 测试
            time = '2019-08-09 19:30:00' 
            # 测试
            rsql = sql.format(time=time,table=table)
            station_all = pd.read_sql(rsql , con=self.conn)
            data = {
                'time':time,
                "data_class":"table_minutes",
                "data":station_all
            }
        else:
            # 当redis有数据时,存储数据并保留最近24小时的
            redis_df = redis_data['data']
            time = redis_df['tTime'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
            daydelay = dtt.timedelta(days=-1)
            daystar = (today + daydelay).strftime('%Y-%m-%d %H:%M:%S')
            redis_df['tTime'] = pd.to_datetime(redis_df['tTime'])
            # 测试
            time = '2019-08-09 19:30:00'
            daystar = '2019-08-08 19:30:00'
            # 测试
            remain_df = redis_df[redis_df['tTime']>=daystar]
            rsql = sql.format(time=time,table=table)
            station_all = pd.read_sql(rsql , con=self.conn)
            output = pd.concat([remain_df,station_all])
            output['tTime'] = pd.to_datetime(output['tTime'])
            output.drop_duplicates(keep='first',inplace=True)
            data = {
                'time':time,
                "data_class":"table_minutes",
                "data":output
            }
        # 将保留的数据重新存储到redis中
        self.rs.set("table_minutes", pickle.dumps(data))  
    def sql_now(self,decode_type,area):
        '''根据date_type向redis中获取数据'''
        return data
    def decode_time(self,data,times,area,boundary):
        date_type = "table_minutes"
        times = 60*3
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-times)
        timeindex = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        timeindex = '2019-08-09 19:00:00'
        data = pickle.loads(self.rs.get(date_type))
        if boundary!="all":
            lat0 =  boundary[0]
            lat1 =  boundary[1]
            lon0 =  boundary[2]
            lon1 =  boundary[3]
        else:
            lat0 = 25
            lat1 = 35
            lon0 = 110
            lon1 = 125
        if area=="nation":
            remain = data[(data['lat']>lat0) & (data['lat']<lat1)  &  (data['lon']<lon1) & (data['lon']>lon0) & (pd.isnull(data['Type']))&(data['ZoomLevel']<6)&(data['tTime']>timeindex)]
        if area=="regin":
            remain = data[(data['lat']>lat0) & (data['lat']<lat1)  &  (data['lon']<lon1) & (data['lon']>lon0) & (data['Type']=="区域站")&(data['tTime']>timeindex)]
        elif area=="all":
            remain = data[(data['lat']>lat0) & (data['lat']<lat1)  &  (data['lon']<lon1) & (data['lon']>lon0)&(data['tTime']>timeindex)]
        elif area=="main":
            remain = data[(data['lat']>lat0) & (data['lat']<lat1)  &  (data['lon']<lon1) & (data['lon']>lon0) & (data['ZoomLevel']<6)&(data['tTime']>timeindex)]
        return remain
    def decode_rain(self,data):
        data = data[data['Ri']>0]
        rain=data.groupby(by=['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['Ri'].sum().reset_index()
        table_list = rain.to_json(orient='records',force_ascii=False)
        return table_list
    def decode_tmin(self,data):
        data = data[data['T']!=-9999]
        tmin=data.groupby(by=['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['T'].min().reset_index()
        table_list = tmin.to_json(orient='records',force_ascii=False)
        return table_list
    def decode_tmax(self,data):
        data = data[data['T']!=-9999]
        tmax = data.groupby(by=['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['T'].max().reset_index()
        table_list = tmax.to_json(orient='records',force_ascii=False)
        return table_list
    def decode_view(self,data):
        data = data[data['V']>=0]
        view = data.groupby(by=['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['V'].min().reset_index()
        table_list = view.to_json(orient='records',force_ascii=False)
        return table_list
    def decode_wind(self,data):
        data = data[data['fFy']!=-9999]
        grouped_IIiii = remain.groupby('IIiii')
        table_list = []
        for i in grouped_IIiii.size().index:
            single = grouped_IIiii.get_group(i)
            wind = single_data['fFy'].max()
            index =  single_data[single_data['fFy'] == single_data['fFy'].max()].index.tolist()[0]
            deg = single_data['dFy'][index]
            single_dir = {
                "IIiii":str(single_data['IIiii'].iloc[0]),
                "StationName":str(single_data['StationName'].iloc[0]),
                "Province":str(single_data['Province'].iloc[0]),
                "City":str(single_data['City'].iloc[0]),
                "County":str(single_data['County'].iloc[0]),
                "Town":str(single_data['Town'].iloc[0]),
                "lat":str(single_data['lat'].iloc[0]),
                "lon":str(single_data['lon'].iloc[0]),
                "fFy":str(wind),
                "dFy":str(deg)          
            }
            table_list.append(single_dir)
        return table_list
    def decode_data(self,decode_type,times,area,boundary,data_type):
        '''根据前段参数处理数据
        返回列表数据
        decode_type--类型
        data_type--请求的数据类型'''
        date_type = "table_minutes"  
        data = pickle.loads(self.rs.get(date_type))['data']
        if decode_type=='now':
            print("解析当前时刻的降水数据")
            data = self.sql_now()
        elif decode_type=='station':
            print("解析单站的降水数据")
        elif decode_type=='aera':
            print("解析面上的降水数据")
            timedata = self.decode_time(data,times,area,boundary)
            if data_type == "rain":
                output = self.decode_rain(timedata)
            elif data_type == "wind":
                output = self.decode_wind(timedata)
            elif data_type == "tmax":
                output = self.decode_tmax(timedata)
            elif data_type == "tmin":
                output = self.decode_tmin(timedata)
            elif data_type == "view":
                output = self.decode_view(timedata)
        return output


start = '2019-08-08 06:00:00'
end = '2019-08-08 18:00:00'  
date_type = 'table_minutes'
station = '58653'
worker = station_zdz(start,end)
dataframe = None
decode_type = "aera"
times = 60*3
area="nation"
boundary = "all"
data_type = "rain"
# worker.decode_dataframe(dataframe,decode_type)
#data = worker.single_station(station)
#worker.upload2_redis_Minutes()
# data = worker.get_redis(date_type)
# data['data']

worker.decode_data(decode_type,times,area,boundary,data_type)
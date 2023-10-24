
import pymysql
import pymssql 
import numpy as np
import pandas as pd
import pickle
import redis
import datetime as dtt
from datetime import timezone
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
            # 测试 基本数据
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
    def decode_time(self):
        date_type = "table_minutes"
        data = self.get_redis(date_type)['data']
        times = [60*24,60*12,60*6,60*3]
        tables = ['24hours','12hours','6hours','3hours']
        # today = self.time_today()
        # 测试时间数据
        end = '2019-08-09 21:00:00' 
        today = dtt.datetime.strptime(end,'%Y-%m-%d %H:%M:%S')
        value_list = ['Ri','Tx','Tn','fFy','V']
        for i, value in enumerate(times):
            offset = dtt.timedelta(minutes=-value)         
            timeindex = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
            remain = data[data['tTime']>timeindex]
            table = tables[i]
            table_data_list = []
            for j in value_list:
                value_df = j
                df = self.return_data(remain,value_df)
                table_data_list.append(df)  
            data_redis = {
                "data_class":tables[i],
                "data":remain,
                "value_list":value_list,
                "table_data_list":table_data_list
            }
            # 将保留的数据重新存储到redis中
            self.rs.set(tables[i], pickle.dumps(data_redis))  
    def get_regin(self,boundary,table_type):
        '''解码单站数据'''
        lat0 = boundary[0]
        lat1 = boundary[1]
        lon0 = boundary[2]
        lon1 = boundary[3]
        boundary_data =  data[(data['lat']>lat0) & (data['lat']<lat1)  &  (data['lon']<lon1) & (data['lon']>lon0)]
        if table_type=="nation":
            remain = data[(pd.isnull(data['Type']))&(data['ZoomLevel']<6)]
        if table_type=="regin":
            remain = data[(data['Type']=="区域站")]
        elif table_type=="all":
            remain = data
        elif table_type=="main":
            remain = data[(data['ZoomLevel']<6)]
        return remain
    def return_data(self,remain,value):
        '''返回数据的类型
        boundary 边界
        table redis的表
        value 要素 Ri fFy Tx Tn V  
        table_type  nation regin main all 
        '''
        if value=="Ri":
            re = remain[remain['Ri']>0]
            df = re.groupby(['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['Ri'].sum().reset_index()
            df['value'] = df['Ri']
            output = df.to_json(orient='records',force_ascii=False)
        elif value=="fFy":
            re = remain[remain['fFy']>0]
            #output = re
            grouped_IIiii = re.groupby(['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])
            all_list = []
            for i in grouped_IIiii.size().index:
                single = grouped_IIiii.get_group(i)
                value = single[ single['fFy']== single['fFy'].max()].head(1)
                all_list.append(value)
            df = pd.concat(all_list) 
            df['value'] = df['fFy']
            output = df.to_json(orient='records',force_ascii=False)         
        elif value=="Tx":
            re = remain[remain['T']!=-9999]
            df = re.groupby(['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['T'].max().reset_index() 
            df['value'] = df['T']
            output = df.to_json(orient='records',force_ascii=False)
        elif value=="Tn":
            re = remain[remain['T']!=-9999]
            df = re.groupby(['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['T'].min().reset_index()
            df['value'] = df['T']
            output = df.to_json(orient='records',force_ascii=False)
        elif value=="V":
            re = remain[remain['V']!=-9999]
            df = re.groupby(['IIiii','StationName','Province','City','County','Town','Type','lat','lon'])['V'].min().reset_index() 
            df['value'] = df['V']
            output = df.to_json(orient='records',force_ascii=False)
        return df
        
        
            

start = '2019-08-08 06:00:00'
end = '2019-08-08 18:00:00'  
date_type = 'table_minutes'
station = '58653'
worker = station_zdz()
dataframe = None
decode_type = "aera"
times = 60*3
area="main" # nation region all main
boundary = "all"
data_type = "wind"
# worker.decode_dataframe(dataframe,decode_type)
#data = worker.single_station(station)
# worker.upload2_redis_Minutes()
# data = worker.get_redis(date_type)
# data['data']
worker.decode_time()
#len(worker.decode_data(decode_type,times,area,boundary,data_type))
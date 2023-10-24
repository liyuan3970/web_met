import datetime
import os

from django.db import connection

from tzdemo.settings import base
from zdz.common.utils import db_utils

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "tzdemo.settings.%s" % base.ENV)


def sync_station_data():
    # 表名
    station_table = "TAB_StationInfo"
    aws_table = "TAB_Aws2019"
    mws_table = "TAB_Mws2019"

    # 两站表名
    station_dict = {
        aws_table: {
            "station_type": 1,
            "criteria": ' st.Province = "浙江" ',
        },
        mws_table: {
            "station_type": 2,
            "criteria": ' st.Province = "浙江" and st.City = "台州" ',
        }
    }
    # 远程数据库
    remote_connector = db_utils.Connector(
        host="192.168.192.5",
        port=3306,
        user="root",
        password="lq9394",
        db="ZJSZDZDB"
    )

    # 获取当前日期
    now = datetime.datetime.now()
    one_day_delta = datetime.timedelta(days=1)
    date = now.date()
    start_time = (date - one_day_delta).strftime("%Y-%m-%d") + " 21:00:00"
    end_time = date.strftime("%Y-%m-%d") + " 20:59:59"
    if now.time().hour > 20:
        date = date + one_day_delta
        start_time = date.strftime("%Y-%m-%d") + " 21:00:00"
        end_time = (date + one_day_delta).strftime("%Y-%m-%d") + " 20:59:59"

    remote_connector.connect()
    # 执行sql
    for table in station_dict.keys():
        # 查询气象站数据
        query = f"""
        select r1.IIiii as station_no,
        r1.Province as station_province,
        r1.City as station_city,
        r1.County as station_county,
        r1.Town as station_town,
        r1.Village as station_village,
        r1.Country as station_country,
        r1.station_name as station_name, 
        r1.lon as lon, r1.lat as lat,
        r1.rain as p_total, r1.tmax as t_max, r1.tmin as t_min, r2.wind_max as w_max, r2.wind_dir as w_dir, r1.vis as vis from
        (select st.IIiii as IIiii, 
        max(st.Province) as Province,
        max(st.City) as City,
        max(st.County) as County,
        max(st.Town) as Town,
        max(st.Village) as Village,
        max(st.Country) as Country,
        max(st.StationName) as station_name,  
        max(st.lon) as lon, max(st.lat) as lat, 
        sum(if(sd.RR>0,sd.RR,0)) as rain, 
        max(if(sd.T>-2730,sd.T,null)) as tmax, 
        min(if(sd.T>-2730,sd.T,null)) as tmin, 
        min(if(sd.VV>=0,sd.VV,null)) as vis from {station_table} as st
        left join {table} as sd on sd.IIiii = st.IIiii
        where 1 = 1 and {station_dict.get(table)["criteria"]} and sd.tTime between "{start_time}" and "{end_time}"
        group by st.IIiii) as r1
        inner join
        (select wind.IIiii as IIiii, max(if(wind.fFy>=0,wind.fFy,null)) as wind_max, max(if(dFy>=0,dFy,null)) as wind_dir from
        (select st.IIiii as IIiii, sd.fFy as fFy, max(dFy) as dFy from {station_table} as st
        left join {table} as sd on sd.IIiii = st.IIiii
        where 1 = 1 and {station_dict.get(table)["criteria"]} and sd.tTime between "{start_time}" and "{end_time}"
        group by st.IIiii, sd.fFy) as wind
        group by wind.IIiii) as r2 on r2.IIiii = r1.IIiii
        """

        results = remote_connector.execute_query(query=query)

        if results:
            # 写入插入本地数据库，同一天之内更新，前一日21-后一日20
            station_type = station_dict.get(table)["station_type"]
            values_list = [(date, data["station_no"],data["station_province"],data["station_city"],data["station_county"],data["station_town"],data["station_village"],data["station_country"] ,data["station_name"], data["lon"], data["lat"], station_type,
                            data["p_total"], data["t_max"], data["t_min"], data["w_max"], data["w_dir"], data["vis"])
                           for data in results]

            # 处理数据
            upsert_sql = f"""
            INSERT INTO station_data(datetime, 
            station_no,
            station_province,
            station_city,
            station_county,
            station_town,
            station_village,
            station_country,
            station_name, 
            lon, lat, station_type, p_total, t_max, t_min, w_max, w_dir, vis) 
            VALUES (%s, %s,%s,%s,%s,%s,%s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
            ON DUPLICATE KEY UPDATE datetime = VALUES(datetime),
            station_no = VALUES(station_no),
            station_province = VALUES(station_province),
            station_city = VALUES(station_city),
            station_county = VALUES(station_county),
            station_town = VALUES(station_town),
            station_village = VALUES(station_village),
            station_country = VALUES(station_country),
            station_name = VALUES(station_name),
            lon = VALUES(lon), lat = VALUES(lat), station_type = VALUES(station_type),
            p_total = VALUES(p_total), t_max = VALUES(t_max), t_min = VALUES(t_min),
            w_max = VALUES(w_max), w_dir = VALUES(w_dir), vis = VALUES(vis)
            """
            with connection.cursor() as cursor:
                cursor.executemany(upsert_sql, values_list)

    # over
    remote_connector.close()


if __name__ == '__main__':
    sync_station_data()

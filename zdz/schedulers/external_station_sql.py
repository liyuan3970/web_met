import datetime
import os

from django.db import connection

from tzdemo.settings import base
from zdz.common.utils import db_utils

from zdz.common.utils import data_class

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "tzdemo.settings.%s" % base.ENV)


def rain_data(tables_name,timesdelay):
    # 开始
    worker = data_class.station_sql_data()
    worker.rain_sql(tables_name,timesdelay)

def wind_data(tables_name,timesdelay):
    # 开始
    worker = data_class.station_sql_data()
    worker.wind_sql(tables_name,timesdelay)

def tmax_data(tables_name,timesdelay):
    # 开始
    worker = data_class.station_sql_data()
    worker.tmax_sql(tables_name,timesdelay)

def tmin_data(tables_name,timesdelay):
    # 开始
    worker = data_class.station_sql_data()
    worker.tmin_sql(tables_name,timesdelay)

def view_data(tables_name,timesdelay):
    # 开始
    worker = data_class.station_sql_data()
    worker.view_sql(tables_name,timesdelay)

if __name__ == '__main__':
    print("这是自动站数据同步到redis的脚本")

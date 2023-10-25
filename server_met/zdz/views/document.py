# -*- coding: utf-8 -*-
import base64
import json
import os
from io import BytesIO

import matplotlib.pyplot as plt
# 序列化numpy的数字
import numpy
import pandas as pd
from django.core.cache import cache
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.clickjacking import xframe_options_exempt
from numpy import random

from zdz.common.utils import data_class, func
from ..models.doucument_model import *
from ..models.user_model import *
# pdf的插件
from weasyprint import HTML
from django.http import HttpResponse, Http404, StreamingHttpResponse
import datetime
from django.contrib.auth import authenticate
# 解决跨域问题
from django.views.decorators.csrf import csrf_exempt 
import pymysql
from django.conf import settings


# json数据的交互程序
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):  # add this line
            return obj.tolist()  # add this line
        return json.JSONEncoder.default(self, obj)


# 首页
def station_zdz(request):
    return render(request, 'station_zdz.html')


# 登录
def login_main(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        if request.method == 'POST':
            passwd = request.POST.get('passwd', '')
            user = request.POST.get('user', '')
            userinfo = authenticate(username=user, password=passwd)
            if userinfo is None:
                return redirect('/login')
            else:
                return redirect('/index')


# 数据交互接口
@csrf_exempt
def station_zdz_data(request):
    model = request.POST.get('model', '')
    click_type = request.POST.get('click_type', '')
    button_value = request.POST.get('button_value', '')
    # 编写数据查询的后端逻辑
    if model =="zdz":
        table_type = request.POST.get('table_type', '')
        value_index = int(request.POST.get('table_index', ''))
        boundary = json.loads(request.POST.get('boundary', ''))
        zoom = int(request.POST.get('zoom', ''))
        worker = data_class.station_zdz()
        tables_name = button_value
        data = worker.get_regin(boundary,table_type,tables_name,value_index,zoom)
        context = {
            'status': "ok",
            'click_type':click_type,
            'data':data 
        }
        return JsonResponse(context)
    elif model =="single":
        # print("环境测试",settings.ENV)
        station = button_value
        plot_type = click_type
        plot_time = request.POST.get('plot_time', '')
        worker = data_class.station_zdz()
        nowdata,history,windhis,windnow = worker.single_station(station)
        context = {
            'status': "ok",
            'now':nowdata,
            'his':history,
            'windhis':windhis,
            'windnow':windnow
        }
        return JsonResponse(context)
    elif model =="extra_download":
        start = request.POST.get('start', '')
        end = request.POST.get('end', '')
        city_type = button_value
        city_code = click_type
        start = '2023-07-01 20:52:00'
        end = '2023-07-02 20:52:00'
        worker = data_class.station_text(city_code,start,end)
        text,rain_json,wind_json,tmax_json,view_json = worker.main()
        # 以上为测试      
        context = {
            'status': "ok",
            'extra_text':text,
            'extra_rain':rain_json,
            'extra_wind':wind_json,
            'extra_tmax':tmax_json,
            'extra_view':view_json
        }
        return JsonResponse(context)
    elif model =="extra_geojosn_plot":
        start = request.POST.get('start', '')
        end = request.POST.get('end', '')
        # city = click_type # 绘图类型
        # country = button_value # 绘图数据
        city_code = request.POST.get('city_code', '')
        worker = data_class.station_text(city_code,start,end)
        plot_type = click_type
        plot_data =json.loads(button_value)
        geojson,hours = worker.plot_rain(plot_type,plot_data)
        # 以上为测试      
        context = {
            'status': "ok",
            'contour':geojson,
            'hours':hours
        }
        return JsonResponse(context)
    elif model =="extra_geojosn_remain":
        rain_data = json.loads(request.POST.get('rain', ''))
        wind_data = json.loads(request.POST.get('wind', ''))
        tmax_data = json.loads(request.POST.get('tmax', ''))
        view_data = json.loads(request.POST.get('view', ''))
        city_code = request.POST.get('city_code', '')
        start = '2019-08-08 08:00:00'  
        end = '2019-08-08 09:00:00'
        worker = data_class.station_text(city_code,start,end)
        text,rain_json,wind_json,tmax_json,view_json = worker.remain(rain_data,wind_data,tmax_data,view_data)
        # 以上为测试      
        context = {
            'status': "ok",
            'extra_text':text,
            'extra_rain':rain_json,
            'extra_wind':wind_json,
            'extra_tmax':tmax_json,
            'extra_view':view_json
        }
        return JsonResponse(context)
    elif model =="radar_sec":
        start = click_type.split(',')
        end = button_value.split(',')
        start_cor = [round(float(start[0]),2),round(float(start[1]),2)]
        end_cor = [round(float(end[0]),2),round(float(end[1]),2)]
        # 测试
        worker = data_class.radar_data()
        img = worker.plot_sec(start_cor,end_cor)
        # print("测试",start_cor)
        # 以上为测试      
        context = {
            'status': "ok",
            'img':img
        }
        return JsonResponse(context)
    elif model =="daily_clander":
        city = click_type 
        click_type = button_value
        worker = data_class.clander(city,click_type)
        data = worker.get_clander()      
        context = {
            'status': "ok",
            'data':data
        }
        return JsonResponse(context)
    elif model =="daily_single":
        city = click_type 
        click_type = button_value
        times = request.POST.get('times', '')
        worker = data_class.clander(city,click_type)
        geojson,wind_json = worker.plot_rain()
        text = worker.return_text()      
        context = {
            'status': "ok",
            'contour':geojson,
            'wind_json':wind_json,
            'text':text
        }
        return JsonResponse(context)
    elif model =="plot_image":
        plot_type = click_type
        city = int(button_value)
        js_status = request.POST.get('js_status', '')
        current_data = request.POST.get('current_data', '')
        time_hours = request.POST.get('time_hours', '')
        if js_status =="False":
            js_status = False
            if current_data =="none":
                # 此处为台州市的画图
                recv_data = "none"
                worker = data_class.server_plot(time_hours,city,plot_type,js_status,recv_data)
                contour,mark = worker.plot_rain()
                text = worker.text_wind_rain()
                context = {
                    'status': "ok",
                    'plot_type':plot_type,
                    'contour':contour,
                    'mark':mark,
                    'text':text
                }
            elif current_data =="province":
                worker = data_class.server_plot(time_hours,city,"rain",js_status,current_data)
                geojson,text,wind_json = worker.return_province()
                context = {
                    'status': "ok",
                    'plot_type':"rain",
                    'contour':geojson,
                    'mark':wind_json,
                    'text':text
                }
        return JsonResponse(context)

# 报警程序的接口
@csrf_exempt
def station_zdz_warring(request):
    name = request.POST.get('name', '')
    print("测试----",name)
    rain_type = request.POST.get('rain_type', '')
    worker = data_class.warring_alert(rain_type)
    img = worker.get_radar()
    # 开始编写风雨数据模型
    rain_data,wind_data,tmax_data,tmin_data,view_data = worker.warring_data()
    context = {
            'warring': "warring",
            'rain':rain_data,
            'wind':wind_data,
            'tmax':tmax_data,
            'tmin':tmin_data,
            'view':view_data,
            'radar':img
        }
    return JsonResponse(context)










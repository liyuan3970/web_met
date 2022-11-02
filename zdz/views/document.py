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


def kuaibao(request):
    # print(this is a index)
    return render(request, 'kuaibao.html', locals())


# demo_02是气象快报的核心代码主要用来统计数据
def index_kb(request):
    # print(this is a index)
    return render(request, 'post_data.html', locals())


def post_data(request):
    # 获取查询数据
    # print(request.POST)
    start = request.POST.get('start', '')
    end = request.POST.get('end', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    city = request.POST.get('city', '')
    dicr = {
        'start': start,
        'end': end,
        'crf': crf,
        'city': city,
    }
    print(dicr, "月份:", start[0:3])
    #

    sql = "test"
    sql_worker = data_class.sql_data(sql)
    RR_County, tmp_max_County, tmp_min_County = sql_worker.comput_county()
    sql_worker.comput_IIiii()
    imd, imd_tmax, imd_tmin, tz_json, RR_sum, RR_rx, level_rain, RR_station_rank, RR_station_bar, tmp_min_scatter, tmp_max_scatter, tmp_event_scatter, tmp_station_bar, VV_min_scatter, fFy_wind7up_scatter, vv_time, vv_value, data_fFy_list = sql_worker.data_output()
    print(RR_station_rank)
    context = {
        'img': imd,
        'img_tmax': imd_tmax,
        'img_tmin': imd_tmin,
        'taizhou': json.dumps(tz_json),
        'RR_County': json.dumps(RR_County),
        'RR_sum': RR_sum,
        'RR_rx': RR_rx,
        'level_rain': level_rain,
        'RR_rank': RR_station_rank,
        'RR_bar': RR_station_bar,
        'tn': json.dumps(tmp_min_County),
        'tn_scatter': tmp_min_scatter,
        'tx': json.dumps(tmp_max_County),
        'tx_scatter': tmp_max_scatter,
        'tmp_event': tmp_event_scatter,
        'tmp_bar': tmp_station_bar,
        'vv_scatter': VV_min_scatter,
        'fy_scatter': fFy_wind7up_scatter,
        'vv_time': vv_time,
        'vv_value': vv_value,
        'fy_list': data_fFy_list,
    }
    # 返回所需数据
    return render(request, 'post_data.html', context)


def url_data(request):
    # 处理点击数据时链接url显示单站数据
    print("链接url")
    return redirect('https://www.baidu.com/')



# 单站请求数据的URl
# http://127.0.0.1:8000/station/k8505/
def station_view(request, station_name):
    return HttpResponse("The station_name is : " + station_name)


# 决策服务操作平台
def index_main(request):
    return render(request, 'index.html')
    # return render(request,'main.html',context)


def login_main(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        if request.method == 'POST':
            passwd = request.POST.get('passwd', '')
            user = request.POST.get('user', '')
            if user == '1' and passwd == '1':
                print("成功")
                # return redirect('/index')
                return render(request, 'index.html')
            else:
                print("失败")
                print(request.POST)
                # return redirect('/index/')
                return render(request, 'login.html')
                # return HttpResponse("失败")
    # return render(request,'index.html')


def quick_look(request):
    data_list = request.POST.get('data_post', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    # data_list = request.POST['data_post']
    # 获取核心数据，保存版本、编写解析函数、保存文档为word、
    print("获取到的预览数据:", data_list)
    return render(request, 'index.html')


# canvas 三点数据自定义绘图
def plot_self_data(request):
    plot_self_data = request.POST.get('plot_self_data', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    data = json.loads(plot_self_data)
    lon = []
    lat = []
    value = []
    # (llcrnrlon=120.1,llcrnrlat=27.8,urcrnrlon=122,urcrnrlat=29.5)
    for i in range(len(data['station'])):
        x = data['station'][i][0]
        lon.append(x)
        y = data['station'][i][1]
        lat.append(y)
        value.append(data['station'][i][2] * 10)
    func.plot_image(lat, lon, value)
    buffer = BytesIO()
    plt.savefig(buffer, bbox_inches='tight')
    plot_img = buffer.getvalue()
    imb = base64.b64encode(plot_img)
    ims = imb.decode()
    imd = "data:image/png;base64," + ims
    # <img src="{{ img }}">
    context2 = {
        'data_test': 723.5,
        "img": imd,
    }
    return JsonResponse(context2)
    # return render(request,'index.html',context2)


def upload_select_taizhou_data(request):
    plot_type = request.POST.get('plot_type', '')
    plot_time = request.POST.get('plot_time', '')

    # 设置key
    key = f"imd_dict_{plot_type}_{plot_time}"
    imd_dict = cache.get(key)
    if not imd_dict:
        print('非缓存方法')
        # 常规方法获取列表
        imd_dict = get_imd_dict(request)
        # 设置缓存
        cache.set(key, imd_dict, timeout=60 * 60 * 24)

    context = {
        'data_test': 723.5,
        'imd_list': imd_dict.get("imd_list"),
        'time_list': imd_dict.get("time_list")
    }
    return JsonResponse(context)


def get_imd_dict(request):
    # 测试绘制等值线的图
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
    plot_type = request.POST.get('plot_type', '')
    plot_time = request.POST.get('plot_time', '')

    print(os.environ["HDF5_USE_FILE_LOCKING"])

    plot_worker = data_class.plot_tz_product(plot_type, plot_time)
    imd_list, time_list = plot_worker.multy_plot()

    return {
        "imd_list": imd_list,
        "time_list": time_list
    }


# 新建文档


def create_new_doc(request):
    writers = Writer.objects.all().values()
    unity = Unity.objects.all().values()
    publisher = Publisher.objects.all().values()
    documenttype = DocumentType.objects.all().values()
    data_publisher = [i['name'] for i in publisher]
    data_writers = [i['name'] for i in writers]
    data_unity = [i['name'] for i in unity]
    data_documenttype = [i['name'] for i in documenttype]
    # print("返回数据")
    context = {
        'data_publisher': data_publisher,
        'data_writers': data_writers,
        'data_unity': data_unity,
        'data_documenttype': data_documenttype
    }
    return JsonResponse(context)


# 创建新的一期文档


def create_new_doc_data(request):
    type_doc = request.POST.get('doc_type', '')
    doc_writer = request.POST.get('doc_writer', '')
    doc_publisher = request.POST.get('doc_publisher', '')
    doc_unity = request.POST.get('doc_unity', '')
    doc_date = request.POST.get('doc_date', '')
    year = doc_date[0:4]
    data = Document.objects.filter(year=year, types=type_doc).last()
    item = data.item + 1
    content = []
    obj = Document.objects.create(
        types=type_doc,
        writer=doc_writer,
        publisher=doc_publisher,
        unity=doc_unity,
        pub_date=doc_date,
        item=item,
        year=year,
        verson_content={
            'blank': content
        }
    )
    context = {
        'status': "ok",
        'type_doc': type_doc

    }
    return JsonResponse(context)


# 获取呈送发的数据
def leader_Data_post(request):
    versions = LeaderData.objects.all().values()

    names = Picture.objects.all().values('name')
    name_list = []
    for name in names:
        name_list.append(name['name'])
    version_list = []
    for version in versions:
        v = {}
        v['name'] = version['name']
        v['service_name'] = version['service_name']
        v['service_unity'] = version['service_unity']
        v['recive_unity'] = version['recive_unity']
        version_list.append(v)

    print("呈送", version_list, name_list)
    context = {
        'status': "ok",
        'version': version_list,
        'name': name_list
    }
    return JsonResponse(context)


# 设置全局变量用来存储EC数据的数据对象
ec_worker = None


def ec_single_data(request):
    # 数据的接收 
    village = request.POST.get('ec_village', '')
    lat = request.POST.get('ec_lat', '')
    lon = request.POST.get('ec_lon', '')
    ec_start_time = request.POST.get('ec_start_time', '')
    ec_end_time = request.POST.get('ec_end_time', '')
    ec_data_type = request.POST.get('ec_data_type', '')
    ec_time_item = request.POST.get('ec_time_item', '')
    ec_select_time = request.POST.get('ec_select_time', '')

    # 测试
    #select_time,select_type,select_county = '2022041700','rain','仙居'
    print("经纬度",lat,"经度:--",lon) 
    select_lon,select_lat,select_time = lon[3:-2],lat[3:-2],'2022041700'
    # 处理数据逻辑
    global ec_worker
    if ec_worker:
        img_rain =  ec_worker.plot_rain(select_lon,select_lat)
        img_wind =  ec_worker.plot_wind(select_lon,select_lat)
    else:
        ec_worker = data_class.ec_data_point(select_lon,select_lat,select_time) 
        img_rain =  ec_worker.plot_rain(select_lon,select_lat)
        img_wind =  ec_worker.plot_wind(select_lon,select_lat)

    # 数据的返回
    context = {
        'status': "ok",
        'img_rain':img_rain,
        'img_wind':img_wind
    }
    return JsonResponse(context)


def self_plot_download(request):
    self_plot_start_time = request.POST.get('self_plot_start_time', '')
    self_plot_end_time = request.POST.get('self_plot_end_time', '')
    # 编写数据查询的后端逻辑
    data = pd.read_csv('static/data/' + 'rect_station_info_tz.csv', encoding='ISO-8859-1')
    data_canvas = {
        "station_list": [],
        "station": []
    }
    length = data.shape[0]

    for i in range(length):
        station_data = []
        if i < 41:
            data_canvas['station_list'].append(data.iloc[i, 4])
            station_data.append(data.iloc[i, 3])
            station_data.append(data.iloc[i, 2])
            station_data.append(random.randint(100))
            data_canvas['station'].append(station_data)
        else:
            station_data.append(data.iloc[i, 3])
            station_data.append(data.iloc[i, 2])
            station_data.append(random.randint(100))
            data_canvas['station'].append(station_data)
    context = {
        'status': "ok",
        'data_canvas': data_canvas
    }
    return JsonResponse(context)


# 自动站历史数据的查询交互
# 设置自动站的全居变量
zdz_worker = None


def tool_zdz_date(request):
    start_time = request.POST.get('start_time', '')
    end_time = request.POST.get('end_time', '')
    # 用于测试
    start = '2022-01-25 20:00'
    end = '2022-02-10 06:00'
    global zdz_worker
    zdz_worker = data_class.zdz_data(start, end)
    context = {
        'status': "ok",
        'day_list': zdz_worker.day_list,
        'day_range': [zdz_worker.day_list[0][0], zdz_worker.day_list[-1][0]],
        'rain_line': zdz_worker.rain_line,
        'rain_scatter': json.dumps(zdz_worker.rain_scatter),
        'rain_img':zdz_worker.img
    }
    return JsonResponse(context)


# 自动站历史数据大风的查询
def tool_zdz_wind(request):
    # 用于测试
    start = '2022-01-25 20:00'
    end = '2022-02-10 06:00'
    global zdz_worker
    if zdz_worker:
        data_wind_list, sort_html = zdz_worker.wind_data()
    else:
        zdz_worker = data_class.zdz_data(start, end)
        data_wind_list, sort_html = zdz_worker.wind_data()
    # print(data_wind_list)
    context = {
        'status': "ok",
        'data_wind_list': json.dumps(data_wind_list, cls=NpEncoder),
        'sort_html': sort_html
    }

    return JsonResponse(context)


# 自动站历史数据能见度的查询
def tool_zdz_view(request):
    # 用于测试
    start = '2022-01-25 20:00'
    end = '2022-02-10 06:00'
    global zdz_worker
    if zdz_worker:
        data_view_list, sort_html = zdz_worker.view_data()
    else:
        zdz_worker = data_class.zdz_data(start, end)
        data_view_list, sort_html = zdz_worker.view_data()
    # print(data_wind_list)
    context = {
        'status': "ok",
        'data_view_list': json.dumps(data_view_list, cls=NpEncoder),
        'sort_html': sort_html
    }

    return JsonResponse(context)


# 自动站历史数据气温的查询
def tool_zdz_temp(request):
    # 用于测试
    start = '2022-01-25 20:00'
    end = '2022-02-10 06:00'
    global zdz_worker
    if zdz_worker:
        data_temp_max, data_temp_min = zdz_worker.temp_data()
    else:
        zdz_worker = data_class.zdz_data(start, end)
        data_temp_max, data_temp_min = zdz_worker.temp_data()
    context = {
        'status': "ok",
        'data_temp_max': data_temp_max,
        'data_temp_min': data_temp_min
    }

    return JsonResponse(context)


# ###################################################################
# 自动站日报daily 的数据查询
def tool_zdz_daily(request):
    # 用于测试
    start = '2022-01-25 20:00'
    end = '2022-02-10 06:00'
    date = request.POST.get('date', '')
    print('获取的时间', date)
    global zdz_worker
    if zdz_worker:
        daily_data = zdz_worker.pre_day(date)
    else:
        zdz_worker = data_class.zdz_data(start, end)
        daily_data = zdz_worker.pre_day(date)
    context = {
        'status': "ok",
        'date': str(date),
        'daily_data': json.dumps(daily_data, cls=NpEncoder)
    }
    return JsonResponse(context)


@xframe_options_exempt
def home(request):
    # return redirect('https://www.baidu.com/')
    # return redirect('http://192.168.192.2:9001/index')
    return redirect('http://www.tz121.com/index.php')
    # return render(request, 'www.baidu.com')

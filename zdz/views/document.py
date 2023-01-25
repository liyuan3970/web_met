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
    # #print(this is a index)
    return render(request, 'kuaibao.html', locals())



def test_demo(request):

    unitys = WebUnity.objects.all().values()
    #images = WebPicture.objects.all().values()
    class_types = WebClass.objects.all().values()
    class_dir = []
    view_list = []
    for unity in unitys:
        if unity['name']=='台州气象局':
            img_src = unity['img']
            check_name = unity['name']
            for single_class in class_types:
                single_list = {
                    'name':single_class['name'],
                    'src':single_class['img'],
                    'img_list':[]
                }            
                img_name = single_class['name']
                images = WebPicture.objects.filter(unity=check_name,webclass=img_name).all().values()
                for img in images:
                    img_dir = {
                        'name':img['name'],
                        'src':img['img']
                    }
                    single_list['img_list'].append(img_dir)
                class_dir.append(single_list)
    
    view_list=class_dir[0]['img_list']  
    #print("数据",class_dir,"shuju ",view_list)    
    content = {
        'img_src': img_src,
        'class_dir':class_dir,
        'first':view_list
    }  
        
    return render(request, 'test_demo.html',locals())


def canvas_plot(request):
    content = {
        'status': "ok"
    }        
    return render(request, 'canvas_plot.html',locals())

# 网站预览功能
def website(request):

    unitys = WebUnity.objects.all().values()
    #images = WebPicture.objects.all().values()
    class_types = WebClass.objects.all().values()
    class_dir = []
    view_list = []
    for unity in unitys:
        if unity['name']=='台州气象局':
            img_src = unity['img']
            check_name = unity['name']
            for single_class in class_types:
                single_list = {
                    'name':single_class['name'],
                    'src':single_class['img'],
                    'img_list':[]
                }            
                img_name = single_class['name']
                images = WebPicture.objects.filter(unity=check_name,webclass=img_name).all().values()
                for img in images:
                    img_dir = {
                        'name':img['name'],
                        'src':img['img']
                    }
                    single_list['img_list'].append(img_dir)
                class_dir.append(single_list)
    
    view_list=class_dir[0]['img_list']  
    #print("数据",class_dir,"shuju ",view_list)    
    content = {
        'img_src': img_src,
        'class_dir':class_dir,
        'first':view_list
    }  
        
    return JsonResponse(content)
# 历史数据查询
def history_file(request):
    unity = "台州市气象局"
    year = 2022
    doc_types = DocumentType.objects.filter(unity=unity).all().values()
    #images = WebPicture.objects.all().values()
    # class_types = WebClass.objects.all().values()
    type_list = []
    for doc_type in doc_types:
        doc_single = {
            'name':doc_type['name'],
            'filelist':[]
        }
        doc_all = Document.objects.filter(unity="台州市气象台",year=year,types=doc_type['name']).all().values()
        for doc in doc_all:
            doc_dir = {}
            doc_dir['type'] = doc['types']
            doc_dir['item'] = doc['item']
            doc_dir['unity'] = doc['unity']
            doc_dir['year'] = doc['year']
            doc_single['filelist'].append(doc_dir)
        type_list.append(doc_single)   
    content = {
        'doc_list': type_list
    }  
        
    return JsonResponse(content)

# demo_02是气象快报的核心代码主要用来统计数据
def index_kb(request):
    # #print(this is a index)
    
    return render(request, 'post_data.html', locals())


def post_data(request):
    # 获取查询数据
    # #print(request.POST)
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
    #print(dicr, "月份:", start[0:3])
    #

    sql = "test"
    sql_worker = data_class.sql_data(sql)
    RR_County, tmp_max_County, tmp_min_County = sql_worker.comput_county()
    sql_worker.comput_IIiii()
    imd, imd_tmax, imd_tmin, tz_json, RR_sum, RR_rx, level_rain, RR_station_rank, RR_station_bar, tmp_min_scatter, tmp_max_scatter, tmp_event_scatter, tmp_station_bar, VV_min_scatter, fFy_wind7up_scatter, vv_time, vv_value, data_fFy_list = sql_worker.data_output()
    #print(RR_station_rank)
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
    #print("链接url")
    return redirect('https://www.baidu.com/')



# 单站请求数据的URl
# http://127.0.0.1:8000/station/k8505/
def station_view(request, station_name):
    return HttpResponse("The station_name is : " + station_name)


# 决策服务操作平台
def index_main(request):
    return render(request, 'index.html')
    # return render(request,'main.html',context)

# 登录页面的完善
def login_main(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        if request.method == 'POST':
            passwd = request.POST.get('passwd', '')
            user = request.POST.get('user', '')
            userinfo=authenticate(username=user,password=passwd)
            if userinfo is None:
                return redirect('/login')             
            else:
                return redirect('/index')
                

def quick_look(request):
    data_list = request.POST.get('data_post', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    # data_list = request.POST['data_post']
    # 获取核心数据，保存版本、编写解析函数、保存文档为word、
    #print("获取到的预览数据:", data_list)
    return render(request, 'index.html')


# canvas 三点数据自定义绘图
def plot_self_data(request):
    plot_self_data = request.POST.get('plot_self_data', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    plot_title,plot_bar  = "降水预报测试画图" ,[0,1,2,3,4,5,6,7]  
    data = json.loads(plot_self_data)
    worker = data_class.canvas_plot(data,plot_title,plot_bar)
    imd = worker.plot_img()
    key,value = worker.village_data()
    pre = worker.shp_average()
    context2 = {
        "img": imd,
        'key':key,
        'value':value,
        'pre':pre
    }
    return JsonResponse(context2)


# 订正产品
def upload_select_taizhou_data(request):
    plot_type = request.POST.get('plot_type', '')
    plot_time = request.POST.get('plot_time', '')
    plot_item = request.POST.get('plot_item', '')
    ##print("订正时次",plot_item)
    if plot_item:
        product = data_class.plot_tz_product(plot_type, plot_time)
        back_data = product.return_data(int(plot_item))
        label = product.label_index(int(plot_item))
        context = {
            'click_data': back_data,
            'click_label': label
        }
        return JsonResponse(context)
    else:
        product = data_class.plot_tz_product(plot_type, plot_time)
        back_data = product.return_data(0)
        label = product.label_index(0)
        btn_index = product.btn_index()
        context = {
            'back_data': back_data,
            'label': label,
            'btn_index': btn_index
        }
        return JsonResponse(context)



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
    # #print("返回数据",writers)
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
    content = request.POST.get('doc_list', '')
    year = doc_date[0:4]
    data = Document.objects.filter(year=year, types=type_doc).last()
    if data==None:
        item = 0
    else:
        item = data.item + 1
    obj = Document.objects.create(
        types=type_doc,
        writer=doc_writer,
        publisher=doc_publisher,
        unity=doc_unity,
        pub_date=doc_date,
        item=item,
        year=year,
        verson_content = content,
        create_user = 0, 
        update_user = 0
    )
    context = {
        'status': "ok",
        'doc_type': type_doc,
        'doc_item':item,
        'doc_unity':doc_unity,
        'doc_year':year

    }
    return JsonResponse(context)


# 打开文档选项
def open_old_doc(request):
    unity = Unity.objects.all().values()
    documenttype = DocumentType.objects.all().values()
    year = Document.objects.all().values()
    today = datetime.datetime.today()
    year = today.year
    data_year = [year,year-1,year-2]
    data_unity = [i['name'] for i in unity]
    data_documenttype = [i['name'] for i in documenttype]
    context = {
        'type': data_documenttype,
        'year': data_year,
        'unity': data_unity
    }
    return JsonResponse(context)
# 打开所选文档的列表
def open_doc_data(request):
    doc_type = request.POST.get('type', '')
    doc_unity = request.POST.get('unity', '')
    doc_year = request.POST.get('year', '') 
    data = Document.objects.filter(year=doc_year,unity=doc_unity,types=doc_type).all().values()
    item_list = [i['item'] for i in data]
    writer_list = [i['writer'] for i in data]
    type_list = [i['types'] for i in data]
    context = {
        'type': type_list,
        'item':item_list,
        'writer':writer_list,
        'unity':doc_unity,
        'year':doc_year

    }
    return JsonResponse(context)

# 加载选中的文档数据
def open_load_object(request):
    info = request.POST.get('info', '')
    fields = info.split("-")
    item = int(fields[0])
    year = int(fields[1]) 
    unity = fields[2] 
    doc_type = fields[3] 
    data = Document.objects.filter(year=year,item =item,unity=unity,types=doc_type).all().values()
    verson_content = str(data[0]['verson_content']).split(",")[0:-1]
    content_list = []
    for i in range(len(verson_content)):
        name = verson_content[i]
        single_content = ""
        data_self = SelfDefine.objects.filter(year=year,name=name,item =item,unity=unity,types=doc_type).all().order_by('-create_time').values()
        for j in data_self:
            if int(j['data']['id'])==i:
                single_content = j['data']['data']
                content_list.append(single_content)
                #print("查找数据",j['name'],j['data']['id'],j['types'],j)
                break
    context = {
        'type_list':verson_content,
        'item':item,
        'year':year,
        'type':doc_type,
        'unity':unity,
        'content_list':content_list
    }
    return JsonResponse(context)

# tinymce_footer
def tinymce_footer(request):
    versions = LeaderData.objects.all().values()
    version_list = []
    for version in versions:
        v = {}
        v['name'] = version['name']
        v['service_name'] = version['service_name']
        v['service_unity'] = version['service_unity']
        v['recive_unity'] = version['recive_unity']
        version_list.append(v)
    datalist = func.decode_footer(version_list)
    images = Picture.objects.all().values()
    image_list = []
    for image in images:
        single = {
            'name':image['name'],
            'src':image['img']
        }
        image_list.append(single)
    html_png = func.footer_png(image_list)
    context = {
        'status': "ok",
        'datalist':datalist,
        'html_png':html_png
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

    #print("呈送", version_list, name_list)
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
    
    ec_start_time = request.POST.get('ec_start_time', '')
    ec_end_time = request.POST.get('ec_end_time', '')
    start_time,end_time = 0,24
    
    # 处理数据逻辑
    global ec_worker
    if ec_worker:
        data = ec_worker.comput_average(start_time,end_time)
    else:
        ec_worker = data_class.ec_data_point(start_time,end_time) 
        data = ec_worker.comput_average(start_time,end_time)
    #print("ec-->ok")
    # 数据的返回
    context = {
        'status': "ok",
        'ec_data':data
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
    text = zdz_worker.text_data()
    context = {
        'status': "ok",
        'day_list': zdz_worker.day_list,
        'day_range': [zdz_worker.day_list[0][0], zdz_worker.day_list[-1][0]],
        'rain_line': zdz_worker.rain_line,
        'rain_scatter': json.dumps(zdz_worker.rain_scatter),
        'rain_img':zdz_worker.img,
        'text':text
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
    # #print(data_wind_list)
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
    # #print(data_wind_list)
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
    #print('获取的时间', date)
    global zdz_worker
    if zdz_worker:
        daily_data = zdz_worker.pre_day(date)
    else:
        zdz_worker = data_class.zdz_data(start, end)
        daily_data = zdz_worker.pre_day(date)
    context = {
        'status': "ok",
        'date': str(date),
        # 'rain': daily_data['rain'],
        # 'tmax': daily_data['tmax'],
        # 'tmin': daily_data['tmin'],
        # 'view': daily_data['view'],
        # 'wind': daily_data['wind'],
        'daily_data': json.dumps(daily_data, cls=NpEncoder)
    }
    return JsonResponse(context)


@xframe_options_exempt
def home(request):
    # return redirect('https://www.baidu.com/')
    # return redirect('http://192.168.192.2:9001/index')
    return redirect('http://www.tz121.com/index.php')
    # return render(request, 'www.baidu.com')

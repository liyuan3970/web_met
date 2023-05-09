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

# 绘图的功能
def canvas_plot(request):
    content = {
        'status': "ok"
    }
    return render(request, 'canvas_plot.html', locals())

# 实况监测
def station_zdz(request):
    content = {
        'status': "ok"
    }
    return render(request, 'station_zdz.html', locals())


# 网站预览功能
def website(request):
    unitys = Unity.objects.all().values()
    # images = WebPicture.objects.all().values()
    class_types = WebClass.objects.all().values()
    class_dir = []
    view_list = []
    for unity in unitys:
        if unity['name'] == '台州市气象局':
            check_name = unity['name']
            for single_class in class_types:
                single_list = {
                    'name': single_class['name'],
                    'img_list': []
                }
                img_name = single_class['name']
                images = WebPicture.objects.filter(unity=check_name, web_class=img_name).all().values()
                for img in images:
                    img_dir = {
                        'name': img['name'],
                        'src': img['img'],
                        'url':img['url']
                    }
                    single_list['img_list'].append(img_dir)
                class_dir.append(single_list)
    view_list = class_dir[0]['img_list']
    # print("数据",class_dir,"shuju ",view_list)
    content = {
        'class_dir': class_dir,
        'first': view_list
    }

    return JsonResponse(content)


# 历史数据查询
def history_file(request):
    info = request.POST.get('info', '')
    if info =='none':        
        unity = "台州市气象局"
        year = 2023
        doc_types = DocumentType.objects.filter(unity=unity).all().values()
        title = list(SelfPlot.objects.values('document_type', 'id'))[-5:]
        type_list = []
        for doc_type in doc_types:
            doc_single = {
                'name': doc_type['name'],
                'filelist': []
            }
            doc_all = Document.objects.filter(unity="台州市气象局", year=year,document_type=doc_type['name']).all().values()
            for doc in doc_all:
                doc_dir = {}
                doc_dir['type'] = doc['document_type']
                doc_dir['item'] = doc['item']
                doc_dir['unity'] = doc['unity']
                doc_dir['year'] = doc['year']
                doc_single['filelist'].append(doc_dir)
            type_list.append(doc_single)
        content = {
            'doc_list': type_list,
            'self_title': title
        }
        return JsonResponse(content)
    else:
        fields = info.split("-")
        unity = "台州市气象局"
        item = int(fields[0])
        doc_type = str(fields[1])
        year = int(fields[2])
        data = Document.objects.filter(year=year, unity=unity, document_type=doc_type).all().values()
        version_content = str(data[0]['version_content']).split(",")[0:-1]
        content_list = []
        for i in range(len(version_content)):
            name = version_content[i]
            single_content = ""
            data_self = SelfModule.objects.filter(year=year, name=name, item=item, unity=unity,document_type=doc_type).all().order_by('-create_time').values()
            for j in data_self:
                if int(j['data']['id']) == i:
                    single_content = j['data']['data']
                    content_list.append(single_content)
                    # print("查找数据",j['name'],j['data']['id'],j['types'],j)
                    break
        content = {
            'status': "ok",
            'type_list': version_content,
            'item': item,
            'year': year,
            'type': doc_type,
            'unity': unity,
            'content_list': content_list
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
    # print(dicr, "月份:", start[0:3])
    #

    sql = "test"
    sql_worker = data_class.sql_data(sql)
    RR_County, tmp_max_County, tmp_min_County = sql_worker.comput_county()
    sql_worker.comput_IIiii()
    imd, imd_tmax, imd_tmin, tz_json, RR_sum, RR_rx, level_rain, RR_station_rank, RR_station_bar, tmp_min_scatter, tmp_max_scatter, tmp_event_scatter, tmp_station_bar, VV_min_scatter, fFy_wind7up_scatter, vv_time, vv_value, data_fFy_list = sql_worker.data_output()
    # print(RR_station_rank)
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
    # print("链接url")
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
            userinfo = authenticate(username=user, password=passwd)
            if userinfo is None:
                return redirect('/login')
            else:
                return redirect('/index')


def quick_look(request):
    data_list = request.POST.get('data_post', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    # data_list = request.POST['data_post']
    # 获取核心数据，保存版本、编写解析函数、保存文档为word、
    # print("获取到的预览数据:", data_list)
    return render(request, 'index.html')


# canvas 三点数据自定义绘图
def plot_self_data(request):
    self_data = request.POST.get('plot_self_data', '')
    time = request.POST.get('time', '')
    title = request.POST.get('title', '')
    crf = request.POST.get('csrfmiddlewaretoken', '')
    types = time + "----" + title
    obj = SelfPlot.objects.create(
        document_type=types,
        time=time,
        data=json.loads(self_data),
        create_user=0,
        update_user=0
    )
    # obj = SelfPlot(types=types,time=time,data =plot_self_data)
    # obj.save()
    context2 = {
        "status": "ok"
    }
    return JsonResponse(context2)


# 订正产品
def upload_select_taizhou_data(request):
    plot_type = request.POST.get('plot_type', '')
    plot_time = request.POST.get('plot_time', '')
    plot_item = request.POST.get('plot_item', '')
    if plot_item:
        product = data_class.plot_tz_product(plot_type, plot_time)
        back_data,data_min,data_max = product.return_data(int(plot_item))
        label = product.label_index(int(plot_item))
        context = {
            'click_data': back_data,
            'click_label': label
        }
        return JsonResponse(context)
    else:
        product = data_class.plot_tz_product(plot_type, plot_time)
        back_data,data_min,data_max = product.return_data(0)
        label = product.label_index(0)
        btn_index = product.btn_index()
        context = {
            'plot_type':plot_type,
            'back_data': back_data,
            'label': label,
            'btn_index': btn_index,
            'data_min':float(data_min),
            'data_max':float(data_max)
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
    data_item = []
    for doc_type in data_documenttype:
        ite = Document.objects.filter(document_type=doc_type).order_by('-item')
        if ite:
            # print("查找数据---",ite,ite[0].item)
            singleitem = {
                'docitem':ite[0].item,
                'doctype':doc_type
            }
        else:
            # print("查找数据---")
            singleitem = {
                'docitem':0,
                'doctype':doc_type
            }  
        data_item.append(singleitem)  
    context = {
        'data_publisher': data_publisher,
        'data_writers': data_writers,
        'data_unity': data_unity,
        'data_documenttype': data_documenttype,
        'data_item':data_item
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
    data = Document.objects.filter(year=year, document_type=type_doc).last()
    if data == None:
        item = 0
    else:
        item = data.item + 1
    obj = Document.objects.create(
        document_type=type_doc,
        writer=doc_writer,
        publisher=doc_publisher,
        unity=doc_unity,
        pub_date=doc_date,
        item=item,
        year=year,
        version_content=content,
        create_user=0,
        update_user=0
    )
    context = {
        'status': "ok",
        'doc_type': type_doc,
        'doc_item': item,
        'doc_unity': doc_unity,
        'doc_year': year

    }
    return JsonResponse(context)


# 打开文档选项
def open_old_doc(request):
    unity = Unity.objects.all().values()
    document_type = DocumentType.objects.all().values()
    year = Document.objects.all().values()
    today = datetime.datetime.today()
    year = today.year
    data_year = [year, year - 1, year - 2]
    data_unity = [i['name'] for i in unity]
    data_document_type = [i['name'] for i in document_type]
    context = {
        'type': data_document_type,
        'year': data_year,
        'unity': data_unity
    }
    return JsonResponse(context)


# 打开所选文档的列表
def open_doc_data(request):
    doc_type = request.POST.get('type', '')
    doc_unity = request.POST.get('unity', '')
    doc_year = request.POST.get('year', '')
    data = Document.objects.filter(year=doc_year, unity=doc_unity, document_type=doc_type).all().values()
    item_list = [i['item'] for i in data]
    writer_list = [i['writer'] for i in data]
    type_list = [i['document_type'] for i in data]
    context = {
        'type': type_list,
        'item': item_list,
        'writer': writer_list,
        'unity': doc_unity,
        'year': doc_year

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
    data = Document.objects.filter(year=year, item=item, unity=unity, document_type=doc_type).all().values()
    date = data[0]['pub_date']
    writer = data[0]['writer']
    publisher = data[0]['publisher']
    version_content = str(data[0]['version_content']).split(",")[0:-1]
    content_list = []
    for i in range(len(version_content)):
        name = version_content[i]
        single_content = ""
        data_self = SelfModule.objects.filter(year=year, name=name, item=item, unity=unity,
                                              document_type=doc_type).all().order_by('-create_time').values()
        for j in data_self:
            if int(j['data']['id']) == i:
                single_content = j['data']['data']
                content_list.append(single_content)
                # print("查找数据",j['name'],j['data']['id'],j['types'],j)
                break
    context = {
        'type_list': version_content,
        'item': item,
        'year': year,
        'type': doc_type,
        'unity': unity,
        'date':date,
        'writer':writer,
        'publisher':publisher,
        'content_list': content_list
    }
    return JsonResponse(context)


# 呈送发的代码
def tinymce_footer(request):
    versions = LeaderData.objects.all().values()
    version_list = []
    for version in versions:
        v = {}
        v['name'] = version['name']
        v['service_name'] = version['service_name']
        v['service_unity'] = version['service_unity']
        v['receive_unity'] = version['receive_unity']
        version_list.append(v)
    datalist = func.decode_footer(version_list)
    context = {
        'status': "ok",
        'datalist': datalist
    }
    return JsonResponse(context)

# 建议库的代码
def tinymce_advicemet(request):
    versions = DocumentAdvice.objects.all().values()
    version_list = []
    for version in versions:
        v = {}
        v['value'] = version['doc_content']
        v['title'] = version['doc_name']
        v['disabled'] = ""
        v['checked'] = ""
        v['label'] = version['doc_label']
        version_list.append(v)
    context = {
        'status': "ok",
        'datalist':version_list
    }
    return JsonResponse(context)

# 自定义模块的代码
def tinymce_selfmode(request):
    versions = DocumentSelfDefine.objects.all().values()
    version_list = []
    for version in versions:
        print("-----------------------",version)
        v = {}
        v['value'] = version['data']['data']
        v['text'] = version['name']
        version_list.append(v)
    # datalist = func.decode_footer(version_list)
    context = {
        'status': "ok",
        'datalist':version_list
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
        v['receive_unity'] = version['receive_unity']
        version_list.append(v)

    # print("呈送", version_list, name_list)
    context = {
        'status': "ok",
        'version': version_list,
        'name': name_list
    }
    return JsonResponse(context)




def ec_single_data(request):
    # 十天预测的数据加载 
    ec_start_time = request.POST.get('ec_start_time', '')
    ec_end_time = request.POST.get('ec_end_time', '')
    start_time, end_time = int(ec_start_time),int(ec_end_time)
    # 处理数据逻辑
    ec_worker = data_class.ec_data_point(start_time, end_time)
    line_data = ec_worker.read_sql()
    plot_data = ec_worker.rander_leaflet(start_time, end_time)
    context = {
        'status': "ok",
        'linedata':line_data,
        'plotdata':plot_data
    }
    return JsonResponse(context)


def self_plot_download(request):
    self_plot_start_time = request.POST.get('self_plot_start_time', '')
    self_plot_end_time = request.POST.get('self_plot_end_time', '')
    self_plot_select_type = request.POST.get('self_plot_select_type', '')
    # 编写数据查询的后端逻辑
    # zdz    
    start_time,end_time,select_type = self_plot_start_time,self_plot_end_time,self_plot_select_type 
    # ec
    # start_time,end_time,select_type = 1,25,"ec"
    sql_worker = data_class.sql_plot(start_time,end_time,select_type)
    data_canvas = sql_worker.return_data()
    context = {
        'status': "ok",
        'data_canvas': data_canvas
    }
    return JsonResponse(context)


# 自动站历史数据的查询交互
def tool_zdz_date(request):
    start_time = request.POST.get('start_time', '')
    end_time = request.POST.get('end_time', '')
    click_type = request.POST.get('click_type', '')
    # 用于测试
    # print("自动站查询的时间",start_time,end_time)
    # start = '2022-01-25 20:00'
    # end = '2022-02-10 06:00'
    # start = start_time
    # end = end_time
    if click_type=='zdz_index':
        start = "2019" + start_time[4:]
        end = "2019" + end_time[4:]
        zdz_worker = data_class.zdz_data(start, end)
        table_data,points,daily_btn_list = zdz_worker.sql_index()
        context = {
            'status': "ok",
            'table_data': json.dumps(table_data, cls=NpEncoder),
            'points': points,
            'daily_btn_list': daily_btn_list
        }
        return JsonResponse(context)
    elif click_type=='mark_click':
        station = request.POST.get('click_station', '')
        click_type = request.POST.get('click_class', '')
        start = "2019" + start_time[4:]
        end = "2019" + end_time[4:]
        zdz_worker = data_class.zdz_data(start, end)
        plot_time,plot_list,plot_type,plot_name = zdz_worker.sql_click(start,end,station,click_type)
        context = {
            'status': "ok",
            'plot_time': json.dumps(plot_time, cls=NpEncoder),
            'plot_list': json.dumps(plot_list, cls=NpEncoder),
            'plot_type': plot_type,
            'plot_name': plot_name
        }
        return JsonResponse(context)
    elif click_type=='btn_single':
        click_type = request.POST.get('click_class', '')
        start = "2019" + start_time[4:]
        end = "2019" + end_time[4:]
        zdz_worker = data_class.zdz_data(start, end)
        if click_type == "view":
            table_data,points = zdz_worker.sql_view(start,end)
            context = {
                'status': "ok",
                'single_data': json.dumps(table_data, cls=NpEncoder),
                'points': points
            }
            return JsonResponse(context)
        if click_type == "wind":
            table_data,points = zdz_worker.sql_wind(start,end)
            context = {
                'status': "ok",
                'single_data': json.dumps(table_data, cls=NpEncoder),
                'points': points
            }
            return JsonResponse(context)
        if click_type == "rain":
            table_data,points = zdz_worker.sql_rain(start,end)
            context = {
                'status': "ok",
                'single_data': json.dumps(table_data, cls=NpEncoder),
                'points': points
            }
            return JsonResponse(context)
        if click_type == "temp":
            table_data,points = zdz_worker.sql_temp(start,end)
            context = {
                'status': "ok",
                'single_data': json.dumps(table_data, cls=NpEncoder),
                'points': points
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
    # print('获取的时间', date)
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


# 自订正降水的查询
def select_self_plot(request):
    select_id = request.POST.get('select_id', '')
    data = list(SelfPlot.objects.filter(id=int(select_id)).all().values())
    # print("自订正降水:",data[0]["data"])
    context = {
        'status': "ok",
        'grid': data[0]["data"]

    }
    return JsonResponse(context)


# 十天数据
def longmet(request):
    # 调用接口
    #title,templist,daylist = func.get_long_data()
    # 解析后的数据
    title = "台州市主城区具体天气预报如下："
    day1 = "3日（周一）：晴到多云"
    day2 = "4日（周二）：晴转多云"
    day3 = "5日（周三）：多云到阴，下午转阴有时有小雨"
    day4 = "6日（周四）：阴局部小雨"
    day5 = "7日（周五）：阴局部小雨 "
    day6 = "8日（周六）：多云到阴局部小雨"
    day7 = "9日（周日）：多云"
    daylist = [day1, day2, day3, day4, day5, day6, day7]
    temp1 = "5～16度；"
    temp2 = "6～16度；"
    temp3 = "10～19度；"
    temp4 = "9～13度；"
    temp5 = "8～11度；"
    temp6 = "7～12度；"
    temp7 = "7～13度。"
    templist = [temp1, temp2, temp3, temp4, temp5, temp6, temp7]
    # 包装数据
    head_h = '''<p class="MsoNormal" style="text-align: left;"><span style="font-size: 20px; font-family: 宋体;">'''
    head_f = '''</span></p><table style="border-collapse: collapse; width: 100.433%; height: 235.438px; border-width: 1px; border-style: none;" border="1"><colgroup><col style="width: 19.0939%;"><col style="width: 67.3133%;"><col style="width: 13.5928%;"></colgroup><tbody>'''
    foot = '''</tbody></table><p class="MsoNormal">&nbsp;</p>'''
    html = ""
    html = html + head_h + title + head_f
    for i in range(len(daylist)):
        header = '''<tr style="height: 33.625px; text-align: left;"><td style="height: 33.625px; border-width: 1px; text-align: left;" colspan="2"><span style="font-family: 宋体;">'''
        middle = '''</span></td><td style="height: 33.625px; border-width: 1px; text-align: right;"><span style="font-family: 宋体;">'''
        footer = '''</span></td></tr>'''
        single = header + daylist[i] + middle + templist[i] + footer
        html = html + single
    html = html + foot
    context = {
        'data': html
    }
    return JsonResponse(context)


# 短期数据
def shortmet(request):
    # 调用接口
    #title,weather,temp1st,temp2st,wind = func.get_short_data()
    # 解析后的数据
    title = "一、天气预报"
    weather = "今天白天阴局部小雨转多云，夜里到明天阴有小雨；后天阴转多云。"
    temp1st = "今天白天最高温度：19～21度；"
    temp2st = "明天最低温度：7～9度。"
    wind = "沿海海面：部分海区有雾，南到西南风5-6级阵风7级。"
    # 包装数据
    head_h = '''<p style="line-height: 2;"><span style="font-size: 26px;"><strong><span style="font-family: 宋体;">'''
    head_f = '''</span></strong></span></p>'''
    content_h = '''<p class="MsoNormal"><span style="font-family: 宋体;">'''
    content_f = '''</span></p>'''
    html = head_h + title + head_f + content_h + weather + content_f + content_h + temp1st + content_f + content_h + temp2st + content_f + content_h + wind + content_f
    context = {
        'data': html
    }
    return JsonResponse(context)

# 网页链接的请求
@xframe_options_exempt
def home(request):
    url = request.GET.get('url')
    return redirect(url)

# 气象数据分析系统
import pymysql
def station_zdz_data(request):
    model = request.POST.get('model', '')
    click_type = request.POST.get('click_type', '')
    button_value = request.POST.get('button_value', '')
    # 编写数据查询的后端逻辑
    conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
    sql_location = """select lat ,lon ,p_total,station_no,station_name from station_data where datetime = '2019-05-01 00:00:00' """
    df_location = pd.read_sql(sql_location , con=conn) 
    datalist = []
    for i in range(len(df_location)):
        single = {
            "name":str(df_location.iloc[i,4]),
            "IIiii":str(df_location.iloc[i,3]),
            "value":str(df_location.iloc[i,2]),
            "lat":str(df_location.iloc[i,0]),
            "lon":str(df_location.iloc[i,1])
        }
        datalist.append(single)
    context = {
        'status': "ok",
        'data':datalist,
        'click_type':click_type
    }
    return JsonResponse(context)




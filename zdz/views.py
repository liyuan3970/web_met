from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from . import func
from . import data_class
# Create your views here.

def kuaibao(request):
    ## print(this is a index)
    return render(request,'kuaibao.html',locals())


# demo_02是气象快报的核心代码主要用来统计数据
def index_kb(request):
    ## print(this is a index)
    return render(request,'post_data.html',locals())


import matplotlib
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import json
import pandas as pd
import numpy as np





def post_data(request):

    #获取查询数据
    #print(request.POST)
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
    print(dicr,"月份:",start[0:3])
    #

    sql = "test"
    sql_worker = data_class.sql_data(sql)
    RR_County,tmp_max_County,tmp_min_County =  sql_worker.comput_county() 
    sql_worker.comput_IIiii()
    imd,imd_tmax,imd_tmin,tz_json,RR_sum ,RR_rx,level_rain,RR_station_rank,RR_station_bar,tmp_min_scatter,tmp_max_scatter,tmp_event_scatter,tmp_station_bar,VV_min_scatter,fFy_wind7up_scatter,vv_time,vv_value,data_fFy_list = sql_worker.data_output()
    print(RR_station_rank)
    context = {
        'img': imd,
        'img_tmax': imd_tmax,
        'img_tmin': imd_tmin,
        'taizhou':json.dumps(tz_json),
        'RR_County':json.dumps(RR_County),
        'RR_sum':RR_sum,
        'RR_rx':RR_rx,
        'level_rain':level_rain,
        'RR_rank':RR_station_rank,
        'RR_bar':RR_station_bar,
        'tn':json.dumps(tmp_min_County),
        'tn_scatter':tmp_min_scatter,
        'tx':json.dumps(tmp_max_County),
        'tx_scatter':tmp_max_scatter,
        'tmp_event':tmp_event_scatter,
        'tmp_bar':tmp_station_bar,
        'vv_scatter':VV_min_scatter,
        'fy_scatter':fFy_wind7up_scatter,
        'vv_time':vv_time,
        'vv_value':vv_value,
        'fy_list':data_fFy_list,

    }
    #返回所需数据
    return render(request,'post_data.html',context)

# 这里处理显示单站数据页面
from django.shortcuts import redirect
def url_data(request):
    # 处理点击数据时链接url显示单站数据
    print("链接url")
    return redirect('https://www.baidu.com/')

# 单站请求数据的URl
#http://127.0.0.1:8000/station/k8505/
def station_view(request,station_name):
    return HttpResponse("The station_name is : " + station_name)


## 决策服务操作平台
def index_main(request):
    context = {
        'data':[123]
    }
    return render(request,'index.html',context)
    # return render(request,'main.html',context)



from django.views.decorators.clickjacking import xframe_options_exempt

@xframe_options_exempt
def home(request):
    # return redirect('https://www.baidu.com/')
    # return redirect('http://192.168.192.2:9001/index')
    return redirect('http://www.tz121.com/index.php')
    # return render(request, 'www.baidu.com')












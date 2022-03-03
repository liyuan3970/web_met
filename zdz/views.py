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
    return render(request,'demo_02.html',locals())


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
    print(dicr)



    # matplotlib绘制降水分布数据
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = [1, 2, 3, 4]
    y = [10, 50, 20, 100]
    plt.plot(x, y, "r", marker='*', ms=10, label="a")
    buffer = BytesIO()
    plt.savefig(buffer)  
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64,"+ims
    # static数据
    # 读取台州的json数据
    with open('static/json/taizhou.json', encoding='utf-8') as f:
        line = f.readline()
        tz_json = json.loads(line)
        tz_json = json.dumps(tz_json)
        # print("json的类型：",type(tz_json))
        f.close()
    
    # 单站降水量的排序
    # RR_station_sum = [
    #     {"index":1,"IIiii":"K8515","name":"临海站","town":"食堂","value":125},
    #     {"index":1,"IIiii":"K8515","name":"临海站","town":"食堂","value":125}
    # ]

    sql = "test"
    RR_County,tmp_max_County,tmp_min_County =  data_class.sql_data(sql).comput_county() 
    level_rain,RR_rx ,RR_sum,RR_station_rank ,RR_station_bar,tmp_station_bar,tmp_min_scatter,tmp_max_scatter,tmp_event_scatter,data_vvmin,VV_min_scatter,VV_station_rank,data_fFy,fFy_wind7up_scatter = data_class.sql_data(sql).comput_IIiii()
    print("能见度排序：",data_vvmin.sort_values(by = 'tTime'))
    vv_time = data_vvmin['tTime'].tolist()
    vv_value = data_vvmin['VV'].tolist()
    context = {
        'img': imd,
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
    }
    #返回所需数据
    return render(request,'demo_02.html',context)

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

















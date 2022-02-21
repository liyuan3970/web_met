from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from . import func
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ## 读取数据
    station_Mws = pd.read_csv("static/data/Mws2022.csv", keep_default_na=-9999)
    station_Aws = pd.read_csv("static/data/Aws2022.csv", keep_default_na=-9999)
    station_all = pd.concat([station_Aws, station_Mws])
    station_58665 = station_all[station_all['IIiii'] == 58665]
    data_hj = station_58665['RR'].values.tolist()
    grouped = station_all.groupby('IIiii')
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 数据1 单站总时间的统计数据 station_dot_all {}
# 数据2 单站聚合时间数据数据 station_dot_comput {}
# 数据3 市县的排序数据 station_county_comput {}
# 数据4 统计站点分级数据 station_event_count {}


    station_IIiii = station_all['IIiii']

    station_name = station_IIiii.drop_duplicates().to_list()


    station_dot_all = {}
    for i in station_name:
        station_dot_all[str(i)] = grouped.get_group(i)
    station_dot_comput = {}
    for i in station_name:
        data = grouped.get_group(i)
        data['VV'].replace(-9999, np.nan, inplace=True)
        data['RR'].replace(-9999, np.nan, inplace=True)
        data['Tn'].replace(-9999, np.nan, inplace=True)
        dic = {}
        dic['IIiii'] = data['IIiii'].iloc[0]
        dic['StationName'] = data['StationName'].iloc[0]
        dic['county'] = data['county'].iloc[0]
        dic['lat'] = data['lat'].iloc[0]
        dic['lon'] = data['lon'].iloc[0]
        dic['Town'] = data['Town'].iloc[0]
        dic['fFy'] = data['fFy'].max()
        dic['dFy'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
        dic['T'] = data['T'].tolist()
        dic['Tx'] = data['Tx'].max()/10.0
        dic['Tn'] = data['Tn'].min()/10.0
        dic['VV'] = data['VV'].min()
        dic['RR'] = data['RR'].sum()
        # 整合
        station_dot_comput[str(i)] = dic

    # 按照市县的排序数据
    grouped_county = station_all.groupby('county')


    station_county = {}
    for i in grouped_county.size().index:
    #     print(i)
        station_county[i] = grouped_county.get_group(i)

    station_county_comput = []
    for i in grouped_county.size().index:
        data = grouped_county.get_group(i)
        data['VV'].replace(-9999, np.nan, inplace=True)
        data['RR'].replace(-9999, np.nan, inplace=True)
        data['Tn'].replace(-9999, np.nan, inplace=True)
        dic = {}
        dic['county'] = str(i)
        dic['RR'] = data['RR'].mean()
        dic['Tx'] = data['Tx'].max()
        dic['Tn'] = data['Tn'].min()
        dic['VV'] = data['VV'].min()
        dic['fFy'] = data['fFy'].max()
        dic['dFy'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
        print(dic)
        station_county_comput.append(dic)

    # 统计实际数据分级（降水、能见度）
    station_event_count = {"RR": [], "VV": []}
    count_vv_red = []
    count_vv_orange = []
    count_vv_yellow = []
    count_vv_blue = []
    vv_station_info = {"name": [], "lat": [], "lon": [], "value": []}
    for i in station_name:
        vv_data = station_dot_comput[str(i)]['VV']
        if vv_data > 500 and vv_data < 1000:
            count_vv_blue.append(1)
            vv_station_info['name'].append(station_dot_comput[str(i)]['IIiii'])
            vv_station_info['lat'].append(station_dot_comput[str(i)]['lat'])
            vv_station_info['lon'].append(station_dot_comput[str(i)]['lon'])
            vv_station_info['value'].append(station_dot_comput[str(i)]['VV'])
        elif vv_data < 500 and vv_data > 250:
            count_vv_yellow.append(1)
            vv_station_info['name'].append(station_dot_comput[str(i)]['IIiii'])
            vv_station_info['lat'].append(station_dot_comput[str(i)]['lat'])
            vv_station_info['lon'].append(station_dot_comput[str(i)]['lon'])
            vv_station_info['value'].append(station_dot_comput[str(i)]['VV'])
        elif vv_data < 250 and vv_data > 50:
            count_vv_orange.append(1)
            vv_station_info['name'].append(station_dot_comput[str(i)]['IIiii'])
            vv_station_info['lat'].append(station_dot_comput[str(i)]['lat'])
            vv_station_info['lon'].append(station_dot_comput[str(i)]['lon'])
            vv_station_info['value'].append(station_dot_comput[str(i)]['VV'])
        elif vv_data < 50 and vv_data > 0:
            count_vv_red.append(1)
            vv_station_info['name'].append(station_dot_comput[str(i)]['IIiii'])
            vv_station_info['lat'].append(station_dot_comput[str(i)]['lat'])
            vv_station_info['lon'].append(station_dot_comput[str(i)]['lon'])
            vv_station_info['value'].append(station_dot_comput[str(i)]['VV'])
        elif vv_data > 1000:
            vv_station_info['name'].append(station_dot_comput[str(i)]['IIiii'])
            vv_station_info['lat'].append(station_dot_comput[str(i)]['lat'])
            vv_station_info['lon'].append(station_dot_comput[str(i)]['lon'])
            vv_station_info['value'].append(station_dot_comput[str(i)]['VV'])

    station_event_count = {"RR": [], "VV": {"red": sum(count_vv_red), "blue": sum(count_vv_blue), "yellow": sum(count_vv_yellow), "orange": sum(count_vv_orange)}}

# 处理数据过程(分为降水模块、气温模块、能见度模块、风力模块)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. 降水模块的基本数据
    # sql计算面雨
    # matplotlib绘制降水分布数据
    # 单站降水量的排序（按照累计降水、按照降水强度）
    # 降水量级分级统计
    # 洪家站（其他市县）降水随时间变化规律
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 



    # 1.获取整合数据(将三个sql的数据进行整合保留数据的（站号、lon、lat、county、town、
    # tTime,dFy,fFy,RR,T,Tx,Tn,VV）)
    # pass

    # 2.sql计算面雨
    RR_County = [
        {"name":'仙居县',"value":data_hj[0]},
        {"name":'椒江区',"value":data_hj[0]},
        {"name":'黄岩区',"value":data_hj[0]},
        {"name":'路桥区',"value":data_hj[0]},
        {"name":'三门县',"value":data_hj[0]},
        {"name":'天台县',"value":900},
        {"name":'温岭市',"value":700},
        {"name":'临海市',"value":800},
        {"name":'玉环市',"value":-900}
    ]
    # RR_County = [
    #     {"name":'仙居县',"value":100},
    #     {"name":'椒江区',"value":200},
    #     {"name":'黄岩区',"value":300},
    #     {"name":'路桥区',"value":400},
    #     {"name":'三门县',"value":500},
    #     {"name":'天台县',"value":600},
    #     {"name":'温岭市',"value":700},
    #     {"name":'临海市',"value":800},
    #     {"name":'玉环市',"value":900}
    # ]


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
    RR_station_sum = [
        {"index":1,"IIiii":"K8515","name":"临海站","town":"食堂","value":125},
        {"index":1,"IIiii":"K8515","name":"临海站","town":"食堂","value":125}
    ]
    # 单站降水量级分布
    RR_station_rank = [
        { "value": 855, "name": '暴雨' },
        { "value": 274, "name": '大雨' },
        { "value": 310, "name": '中雨' },
        { "value": 335, "name": '小-中雨' },
        { "value": 400, "name": '小雨' }
    ]
    # 降水柱状图
    RR_station_bar = [
        ['product', '市县降水', '流域降水'],
        ['市局', 43.3, 85.8],
        ['天台', 83.1, 73.4],
        ['仙居', 86.4, 65.2],
        ['三门', 72.4, 53.9],
        ['临海', 72.4, 53.9],
        ['椒江', 72.4, 53.9],
        ['黄岩', 72.4, 53.9],
        ['路桥', 72.4, 53.9],
        ['温岭', 72.4, 53.9],
        ['玉环', 72.4, 53.9]
]


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. 气温模块
    # 最低气温的面分布和散点分布
    # 最高气温的面分布和散点分布
    # 指标站的时序图
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # 最低气温
    nation_station = ['58660','58666','K8505','K8206','58665','58559','58655','K8271','58662','58653']
    temp_scatter_list = ['58559','K8705','K8706','58652','K8903','58568','K8818','58662','K8821','58660','58653','K8609','K8505','58667','58664','K8413','58655','K8282','K8217','K8201','K8301','58665']
    # 散点数据
    tmp_min_scatter = []
    tmp_max_scatter = []
    for i in temp_scatter_list:
        dic_temp_max = {"value":[],"url":"",}
        dic_temp_max['value'].append(station_dot_comput[i]['lon'])
        dic_temp_max['value'].append(station_dot_comput[i]['lat'])
        dic_temp_max['value'].append(station_dot_comput[i]['Tx'])
        dic_temp_max['url'] ="station/"+str(station_dot_comput[i]['IIiii'])
        dic_temp_max['name'] = str(station_dot_comput[i]['StationName'])
        dic_temp_min = {"value":[],"url":""}
        dic_temp_min['value'].append(station_dot_comput[i]['lon'])
        dic_temp_min['value'].append(station_dot_comput[i]['lat'])
        dic_temp_min['value'].append(station_dot_comput[i]['Tn'])
        dic_temp_min['url'] ="station/"+str(station_dot_comput[i]['IIiii'])
        dic_temp_min['name'] = str(station_dot_comput[i]['StationName'])
        tmp_max_scatter.append(dic_temp_max)
        tmp_min_scatter.append(dic_temp_min)
        #print(tmp_max_scatter)
    # 柱状图数据
    tmp_station_bar = []
    tmp_station_bar.append(['product', '最高气温','最低气温'])
    for i in nation_station:
        tmp_station_bar.append([station_dot_comput[i]['StationName'], station_dot_comput[i]['Tx'],station_dot_comput[i]['Tn']])
    
    # 面数据
    tmp_max_County = []
    tmp_min_County = []
    for i in station_county_comput:
        tmp_max_County.append({"name":i['county'],"value":str(i['Tx'])})
        tmp_min_County.append({"name":i['county'],"value":str(i['Tn'])})
    # tmp_min_County = [
    #     {"name": '仙居县', "value": 100},
    #     {"name": '椒江区', "value": 200},
    #     {"name": '黄岩区', "value": 300},
    #     {"name": '路桥区', "value": 400},
    #     {"name": '三门县', "value": 500},
    #     {"name": '天台县', "value": 600},
    #     {"name": '温岭市', "value": 700},
    #     {"name": '临海市', "value": 800},
    #     {"name": '玉环市', "value": 900}
    # ]
    # 散点最低气温
    # tmp_min_scatter = [{ "value": [121.5, 28.5, 80], "url": "station/58667" ,"name":"58667"}]

    # 最低气温
    # tmp_max_County = [
    #     {"name": '仙居县', "value": 100},
    #     {"name": '椒江区', "value": 200},
    #     {"name": '黄岩区', "value": 300},
    #     {"name": '路桥区', "value": 400},
    #     {"name": '三门县', "value": 500},
    #     {"name": '天台县', "value": 600},
    #     {"name": '温岭市', "value": 700},
    #     {"name": '临海市', "value": 800},
    #     {"name": '玉环市', "value": 900}
    # ]
    # 散点最低气温
    # tmp_max_scatter = [{ "value": [121.5, 28.5, 80], "url": "url_data" }]

    # 指标站的时序图
    # tmp_station_bar = [
    #     ['product', '最高气温','最低气温'],
    #     ['市局', 43.3, 43.3],
    #     ['天台', 83.1, 43.3],
    #     ['仙居', 86.4, 43.3],
    #     ['三门', 72.4, 43.3],
    #     ['临海', 72.4, 43.3],
    #     ['椒江', 72.4, 43.3],
    #     ['黄岩', 72.4, 43.3],
    #     ['路桥', 72.4, 43.3],
    #     ['温岭', 72.4, 43.3],
    #     ['玉环', 72.4, 43.3],
    # ]

    context = {
        'img': imd,
        'taizhou':json.dumps(tz_json),
        'RR_County':json.dumps(RR_County),
        'RR_sum':RR_station_sum,
        'RR_rank':RR_station_rank,
        'RR_bar':RR_station_bar,
        'tn':json.dumps(tmp_min_County),
        'tn_scatter':tmp_min_scatter,
        'tx':json.dumps(tmp_max_County),
        'tx_scatter':tmp_min_scatter,
        'tmp_bar':tmp_station_bar,
        'testdata':json.dumps(data_hj),

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

















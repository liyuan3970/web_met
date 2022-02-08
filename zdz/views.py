from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
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
def post_data(request):


    


    #获取查询数据
    #print(request.POST)
    start = request.POST.get('start','')
    end = request.POST.get('end','')
    crf = request.POST.get('csrfmiddlewaretoken','')
    city = request.POST.get('city','')
    dicr = {
        'start':start,
        'end':end,
        'crf':crf,
        'city':city,
    }
    print(dicr)


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
        {"name":'仙居县',"value":100},
        {"name":'椒江区',"value":200},
        {"name":'黄岩区',"value":300},
        {"name":'路桥区',"value":400},
        {"name":'三门县',"value":500},
        {"name":'天台县',"value":600},
        {"name":'温岭市',"value":700},
        {"name":'临海市',"value":800},
        {"name":'玉环市',"value":900}
    ]



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
        ['玉环', 72.4, 53.9],
]


    context = {
        'img': imd,
        'taizhou':json.dumps(tz_json),
        'RR_County':json.dumps(RR_County),
        'RR_sum':RR_station_sum,
        'RR_rank':RR_station_rank,
        'RR_bar':RR_station_bar,
    }

    #返回所需数据



    return render(request,'demo_02.html',context)


from django.shortcuts import redirect
def url_data(request):
    # 处理点击数据时链接url显示单站数据
    print("链接url")
    return redirect('https://www.baidu.com/')

















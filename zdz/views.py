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


    # 处理数据过程
    # 测试绘图数据
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    x = [1, 2, 3, 4]
    y = [10, 50, 20, 100]
    # fig,ax=plt.subplots(figsize=(8.9,6.5))
    plt.plot(x, y, "r", marker='*', ms=10, label="a")
    #fig = plt.figure()
    # axes=fig.add_axes([0,0,1,1])
    # axes.set_axis_off()
    buffer = BytesIO()
    plt.savefig(buffer)
    
    plot_data = buffer.getvalue()
    imb = base64.b64encode(plot_data)  # 对plot_data进行编码
    ims = imb.decode()
    imd = "data:image/png;base64,"+ims
    ## 所需数据
    # 读取台州的json数据
    with open('static/json/taizhou.json', encoding='utf-8') as f:
        line = f.readline()
        tz_json = json.loads(line)
        f.close()
    sum_RR = {
        "data":[]
    }
    _sum_station = {
        "data":[]
    }
    context = {
        'img': imd,
        'taizhou':tz_json,
    }

    #返回所需数据



    return render(request,'demo_02.html',context)


from django.shortcuts import redirect
def url_data(request):
    # 处理点击数据时链接url显示单站数据
    print("链接url")
    return redirect('https://www.baidu.com/')

















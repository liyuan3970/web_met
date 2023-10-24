import datetime

# import netCDF4 as nc
# from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
# from osgeo import gdal
import shapefile
import xarray as xr
from affine import Affine
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from ncmaps import Cmaps
from rasterio import features
from scipy.interpolate import griddata
import requests as rq

def transform_from_latlon(lat, lon):
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale


def rasterize(shapes, coords, latitude='lat', longitude='lon',
              fill=np.nan, **kwargs):
    transform = transform_from_latlon(coords[latitude], coords[longitude])
    out_shape = (len(coords[latitude]), len(coords[longitude]))
    raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
    spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
    return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))


def add_shape_coord_from_data_array(xr_da, shp_path, coord_name):
    # 1. read in shapefile
    shp_gpd = gpd.read_file(shp_path)

    # 2. create a list of tuples (shapely.geometry, id)
    #    this allows for many different polygons within a .shp file (e.g. States of US)
    shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]

    # 3. create a new coord in the xr_da which will be set to the id in `shapes`
    xr_da[coord_name] = rasterize(shapes, xr_da.coords,
                                  longitude='lon', latitude='lat')

    return xr_da


def basemask(cs, ax, map, shpfile):
    sf = shapefile.Reader(shpfile)
    vertices = []
    codes = []
    for shape_rec in sf.shapeRecords():
        if shape_rec.record[0] >= 0:
            pts = shape_rec.shape.points
            prt = list(shape_rec.shape.parts) + [len(pts)]
            for i in range(len(prt) - 1):
                for j in range(prt[i], prt[i + 1]):
                    vertices.append(map(pts[j][0], pts[j][1]))
                codes += [Path.MOVETO]
                codes += [Path.LINETO] * (prt[i + 1] - prt[i] - 2)
                codes += [Path.CLOSEPOLY]
            clip = Path(vertices, codes)
            clip = PathPatch(clip, transform=ax.transData)
    for contour in cs.collections:
        contour.set_clip_path(clip)


def makedegreelabel(degreelist):
    labels = [str(x) + u'°E' for x in degreelist]
    return labels


# 用来出定时任务
def contrab_data():
    # 将数据存储到数据库中的基本操作
    time_now = datetime.datetime.now().strftime('%Y-%m-%d')
    print("测试时间:", time_now)


def plot_image(lat, lon, value):
    # 线性插值
    N = len(value)
    a = []
    b = []
    z = []
    for i in range(N):
        if value[i] != -9999.0:
            a.append(round(lon[i], 2))
            b.append(round(lat[i], 2))
            z.append(round(value[i], 2))
    lat = np.array(b)
    lon = np.array(a)
    Zi = np.array(z)
    data_max = max(Zi)
    data_min = min(Zi)
    np.set_printoptions(precision=2)
    x = np.arange(120.0, 122.0, 0.05)
    # print(x)
    y = np.arange(27.8, 29.5, 0.05)
    nx0 = len(x)
    ny0 = len(y)
    X, Y = np.meshgrid(x, y)  # 100*100
    P = np.array([X.flatten(), Y.flatten()]).transpose()
    Pi = np.array([lon, lat]).transpose()
    Z_linear = griddata(Pi, Zi, P, method="nearest").reshape([ny0, nx0])
    # 绘图
    levels = np.linspace(start=data_min / 10.0, stop=data_max / 10.0, num=7)
    self_define_list = [130, 144, 155, 170, 185, 200, 225, 235, 240, 244]
    rgb_file = 'ncl_default'
    # 以下是核心api,实质为调用Cmaps基类的listmap()方法
    cmaps = Cmaps('ncl_default', self_define_list).listmap()
    # plt.rcParams.update({'font.size': 20})
    fig = plt.figure(figsize=[10, 8])

    ax = fig.add_subplot(111)
    # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    filepath = "static/data/shpfile/"
    data_xr = xr.DataArray(Z_linear / 10.0, coords=[y, x],
                           dims=["lat", "lon"])
    shp_da = add_shape_coord_from_data_array(data_xr, filepath + "taizhou.shp", "test")
    awash_da = shp_da.where(shp_da.test < 7, other=np.nan)
    m = Basemap(llcrnrlon=120.0, llcrnrlat=27.8, urcrnrlon=122, urcrnrlat=29.5, resolution=None, projection='cyl')
    # 设置colorbar
    cbar_kwargs = {'shrink': 0.5}
    # cs = data_xr.plot.contourf(ax=ax, cmap='Spectral_r',levels =levels,cbar_kwargs=cbar_kwargs,add_labels=False)
    cs = data_xr.plot.contourf(ax=ax, cmap='Spectral_r', add_labels=False)
    m.readshapefile(filepath + 'taizhou', 'taizhou', color='k', linewidth=1.2)
    parallels = np.arange(27.8, 29.5, 0.2)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels, labels=[True, False, True, False], color='dimgrey', dashes=[2, 3],
                    fontsize=12)  # ha= 'right'
    meridians = np.arange(120.0, 122.0, 0.2)
    m.drawmeridians(meridians, labels=[False, True, False, True], color='dimgrey', dashes=[2, 3], fontsize=12)
    len_lat = len(data_xr.lat.data)
    len_lon = len(data_xr.lon.data)
    # for i in range(len_lon-1):
    #     for j in range(len_lat-1):
    #         y0 = round(27.8+j*0.05,2)
    #         x0 = round(120.0+i*0.05,2)
    #         if not isnan(awash_da.data[j,i]):
    #             plt.text(x0,y0,str(int(awash_da.data[j,i])),fontsize= 7,fontweight = 800 ,color ="black")
    basemask(cs, ax, m, filepath + 'taizhou')

# 解析tinymce的footer 模块
def decode_footer(version_list):
    lineh = '''<div><hr style="background-color: :#000000; size:3px"></div>'''
    linep = '''<div style="line-height: 1.5;"><div><span style="font-size: 10pt;"><span style="font-family: 仿宋_GB2312;">'''
    lineend = '''</div></div></div>'''
    linef = '''</span></div>'''
    datalist = []
    for data in version_list:
        single = {}
        single['value'] = lineh + linep + "呈</span>:" + data['service_name'] + linef +linep + "送</span>:" + data['service_unity'] + linef +linep + "发</span>:" 
        single['value'] = single['value'] + data['receive_unity'] + linef + lineend
        single['text'] = data['name']
        datalist.append(single)
    return datalist

# 爬取短期和十天
def get_short_data():
    url = '''http://10.137.4.30:6001/integration/main/ssd-product-publish/getLatestPublish?productCode=dqyb&loginAreaId=34197a647f8111eaae330221860e9b7e&loginOrgId=fd1411e47c3f4dcb87beaa110a21a979&loginUserId=dc739e110dbe4911bb7438da035b5d59'''
    req =  rq.get(url,timeout=3000)
    req_json = req.json()
    req_data = req_json['data']['content'].split("\n") 
    for i in range(len(req_data)):
        if req_data[i]=='天气预报：':
            title = req_data[i]
            weather = req_data[i+1]
            temp1st = req_data[i+2] 
            temp2st = req_data[i+3] 
            wind = req_data[i+4]  
    #print(title,weather,temp1st,temp2st,wind)
    return title,weather,temp1st,temp2st,wind

def get_long_data():
    url = '''http://10.137.4.30:6001/integration/main/ssd-product-publish/getLatestPublish?productCode=sttq&loginAreaId=34197a647f8111eaae330221860e9b7e&loginOrgId=fd1411e47c3f4dcb87beaa110a21a979&loginUserId=dc739e110dbe4911bb7438da035b5d59'''
    req =  rq.get(url,timeout=3000)
    req_json = req.json()
    req_data = req_json['data']['content'].split("\n") 
    return_list = []
    for i in range(len(req_data)):
        if req_data[i]=='台州主城区十天预报：':
            title = req_data[i]
            day1= req_data[i+1].split("，")[-1]
            temp1 = req_data[i+1].replace(day1,"")
            day2= req_data[i+2].split("，")[-1]
            temp2 = req_data[i+2].replace(day2,"")
            day3= req_data[i+3].split("，")[-1]
            temp3 = req_data[i+3].replace(day3,"")
            day4= req_data[i+4].split("，")[-1]
            temp4 = req_data[i+4].replace(day4,"")
            day5= req_data[i+5].split("，")[-1]
            temp5 = req_data[i+5].replace(day5,"")
            day6= req_data[i+6].split("，")[-1]
            temp6 = req_data[i+6].replace(day6,"")
            day7= req_data[i+7].split("，")[-1]
            temp7 = req_data[i+7].replace(day7,"")
            daylist = [day1, day2, day3, day4, day5, day6, day7]
            templist = [temp1, temp2, temp3, temp4, temp5, temp6, temp7]   
    #print(title,templist,daylist)  
    return title,templist,daylist
import base64
import json
import os
from io import BytesIO
from math import isnan

import geopandas as gpd
import matplotlib.pyplot as plt
import netCDF4
import numpy as np
# import modin.pandas as pd
import pandas as pd
import shapefile
import xarray as xr
from affine import Affine
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from ncmaps import Cmaps
from rasterio import features

from . import func

import xesmf as xe
import redis
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
# import h5netcdf.legacyapi as netCDF4

os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import datetime as dtt
from datetime import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from pylab import *
from matplotlib.font_manager import FontProperties
# 查询历史数据的calss
class sql_data:
    def __init__(self, sql):
        self.sql = sql  # 传进来的参数
        station_Mws = pd.read_csv("static/data/Mws_15.csv")
        station_Aws = pd.read_csv("static/data/Aws_15.csv")
        self.station_all = pd.concat([station_Aws, station_Mws])
        # 数据
        self.grouped_county = self.station_all.groupby('county')
        self.grouped_IIiii = self.station_all.groupby('IIiii')
        self.timecounts = len(self.grouped_IIiii.get_group(58660)['tTime'])
        # 统计变量.station_RR_small = 0.0
        self.station_RR_small = 0.0
        self.station_RR_mid = 0.0
        self.station_RR_big = 0.0
        self.station_RR_huge = 0.0
        self.station_RR_bighuge = 0.0
        self.station_RR_more = 0.0

        self.station_wind7 = 0.0
        self.station_wind8 = 0.0
        self.station_wind9 = 0.0
        self.station_wind10 = 0.0
        self.station_wind11 = 0.0
        self.station_wind12 = 0.0
        self.station_wind13 = 0.0
        self.station_wind14 = 0.0
        self.station_wind15 = 0.0
        self.station_wind16 = 0.0
        self.station_wind17 = 0.0

        self.station_VV_small = 0.0
        self.station_VV_mid = 0.0
        self.station_VV_big = 0.0
        self.station_VV_huge = 0.0
        self.station_VV_more = 0.0
        # 分级
        self.RR_rank = []
        self.fFy_rank = []
        # 散点
        self.data_station = []
        # 单站
        self.station_list = {}
        self.symbol_ffy = ['path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z']
        self.vv_min_name = ""
        self.vv_min_value = 9999
        # 绘图和排序
        self.plot_data = {
            'IIiii': [],
            'lat': [],
            'lon': [],
            'county': [],
            'town': [],
            'name': [],
            'fFy': [],
            'dFy': [],
            'rsum': [],
            'tmax': [],
            'tmin': [],
            'rmax': []
        }

    def comput_county(self):
        '计算面最大雨强、累计降水、最高、最低气温'
        self.station_county_comput = []
        for i in self.grouped_county.size().index:
            data = self.grouped_county.get_group(i)
            data['VV'].replace(-9999, np.nan, inplace=True)
            data['RR'].replace(-9999, np.nan, inplace=True)
            data['Tn'].replace(-9999, np.nan, inplace=True)
            data['Tx'].replace(-9999, np.nan, inplace=True)
            dic = {}
            dic['county'] = str(i)
            dic['RR'] = data['RR'].mean() * self.timecounts / 10.0
            dic['RMax'] = data['RR'].max() / 10.0
            dic['Tx'] = data['Tx'].max() / 10.0
            dic['Tn'] = data['Tn'].min() / 10.0
            #             print(dic)
            self.station_county_comput.append(dic)
        tmp_max_County = []
        tmp_min_County = []
        RR_County = []
        for i in self.station_county_comput:
            tmp_max_County.append({"name": i['county'], "value": i['Tx']})
            tmp_min_County.append({"name": i['county'], "value": i['Tn']})
            RR_County.append({"name": i['county'], "value": i['RR']})
        return RR_County, tmp_max_County, tmp_min_County

    def data_gevent(self, data):
        # print("data_gevent")

        nan_del = lambda x: 9999 if np.isnan(x) else x
        value_rsum = data['RR'].sum() / 10.0
        value_rmax = data['RR'].sum() / 10.0
        fFy_data = data['fFy'].max() / 10.0
        dFy_data = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
        value_VV = nan_del(data['VV'].min())
        value_tmax = nan_del(data['Tx'].max() / 10.0)
        value_tmin = nan_del(data['Tn'].min() / 10.0)
        # print("查看数据：",print(type(value_VV)),value_VV)
        if value_rsum >= 0 and value_rsum < 10:
            self.station_RR_small = self.station_RR_small + 0
        elif value_rsum >= 9 and value_rsum < 25:
            self.station_RR_mid = self.station_RR_mid + 0
        elif value_rsum >= 24 and value_rsum < 50:
            self.station_RR_big = self.station_RR_big + 0
        elif value_rsum >= 49 and value_rsum < 100:
            self.station_RR_huge = self.station_RR_huge + 0
        elif value_rsum >= 99 and value_rsum < 250:
            self.station_RR_RR_bighuge = self.station_RR_bighuge + 0
        else:
            self.station_RR_more = self.station_RR_more + 0
        # 大风

        if fFy_data > 12.8 and fFy_data <= 17.1:
            self.station_wind6 = self.station_wind7 + 1
        elif fFy_data > 16.1 and fFy_data <= 20.7:
            self.station_wind7 = self.station_wind8 + 1
        elif fFy_data > 19.7 and fFy_data <= 24.4:
            self.station_wind8 = self.station_wind9 + 1
        elif fFy_data > 23.4 and fFy_data <= 28.4:
            self.station_wind9 = self.station_wind10 + 1
        elif fFy_data > 27.4 and fFy_data <= 32.6:
            self.station_wind10 = self.station_wind11 + 1
        elif fFy_data > 31.6 and fFy_data <= 36.9:
            self.station_wind11 = self.station_wind12 + 1
        elif fFy_data > 35.9 and fFy_data <= 41.4:
            self.station_wind12 = self.station_wind13 + 1
        elif fFy_data > 40.4 and fFy_data <= 46.1:
            self.station_wind13 = self.station_wind14 + 1
        elif fFy_data > 45.1 and fFy_data <= 51.0:
            self.station_wind14 = self.station_wind15 + 1
        elif fFy_data > 50.0 and fFy_data <= 56.1:
            self.station_wind15 = self.station_wind16 + 1
        else:
            self.station_wind16 = self.station_wind17 + 1

        # 能见度
        if value_VV < self.vv_min_value:
            self.vv_min_value = value_VV
            self.vv_min_name = str(data['IIiii'].iloc[0])
        if value_VV >= 0 and value_VV < 50:
            self.station_VV_small = self.station_VV_small + 0
        elif value_VV >= 49 and value_VV < 200:
            self.station_VV_mid = self.station_VV_mid + 0
        elif value_VV >= 199 and value_VV < 500:
            self.station_VV_big = self.station_VV_big + 0
        elif value_VV >= 499 and value_VV < 1000:
            self.station_VV_huge = self.station_VV_huge + 0
        else:
            self.station_VV_more = self.station_VV_more + 0
        # 绘图、排序数据
        self.plot_data['IIiii'].append(data['IIiii'].iloc[0])
        self.plot_data['lat'].append(data['lat'].iloc[0])
        self.plot_data['lon'].append(data['lon'].iloc[0])
        self.plot_data['county'].append(data['county'].iloc[0])
        self.plot_data['town'].append(data['Town'].iloc[0])
        self.plot_data['name'].append(data['StationName'].iloc[0])
        self.plot_data['fFy'].append(fFy_data)
        self.plot_data['dFy'].append(dFy_data)
        self.plot_data['rsum'].append(value_rsum / 10.0)
        self.plot_data['rmax'].append(data['RR'].max() / 10.0)
        self.plot_data['tmax'].append(value_tmax)
        self.plot_data['tmin'].append(value_tmin)

        self.dic_station = {'IIiii': data['IIiii'].iloc[0],
                            'StationName': data['StationName'].iloc[0],
                            'County': data['county'].iloc[0],
                            'Town': data['Town'].iloc[0],
                            'lat': data['lat'].iloc[0],
                            'lon': data['lon'].iloc[0],
                            'rsum': value_rsum / 10.0,
                            'rmax': data['RR'].max() / 10.0,
                            'tmax': value_tmax,
                            'tmin': value_tmin,
                            'vmin': value_VV,
                            'fmax': fFy_data,
                            'dfx': data[data['fFy'] == data['fFy'].min()]['dFy'].iloc[0],
                            #    'label_rsm': value_rsum/10.0,
                            #    'label_tx': data['RR'].max()/10.0,
                            #    'label_tn': data['Tn'].min()/10.0,
                            #    'label_tworn': data['StationName'].iloc[0],
                            #    'label_v': data['VV'].min(),
                            #    'label_fy': data['StationName'].iloc[0],
                            'url_r': "station/" + str(data['IIiii'].iloc[0]) + "/rain/",
                            'url_t': "station/" + str(data['IIiii'].iloc[0]) + "/temp/",
                            'url_v': "station/" + str(data['IIiii'].iloc[0]) + "/vv/",
                            'url_fy': "station/" + str(data['IIiii'].iloc[0]) + "/fFy/",
                            'name': data['StationName'].iloc[0],

                            'value': [
                                data['lon'].iloc[0],
                                data['lat'].iloc[0],
                                value_rsum,
                                value_rmax,
                                value_VV,
                                fFy_data,
                                value_tmax,
                                value_tmin
                            ]
                            #    'symbol':str(self.symbol_ffy[0]) ,
                            #    'symbolRotate':data[data['fFy'] == data['fFy'].min()]['dFy'].iloc[0],

                            }
        # self.data_station = self.data_station.append(self.dic_station, ignore_index=True)
        self.data_station.append(self.dic_station)
        self.station_list_dir = {
            'name': data['StationName'].iloc[0],
            'rmax': data['RR'].max() / 10.0,
            'rsum': value_rsum / 10.0,
            # 'tmax':[],
            # 'tmin':[],
            'time': data['tTime'].tolist(),
            'T': data['T'].tolist(),
            'V': data['VV'].tolist(),
            'fFy': data['fFy'].tolist(),
            'dFy': data['dFy'].tolist()
        }
        self.station_list[str(data['IIiii'].iloc[0])] = self.station_list_dir
        # print(self.data_station)

        # return station_list,data_station

    def read_shp_json(self):
        with open('static/json/taizhou.json', encoding='utf-8') as f:
            line = f.readline()
            tz_json = json.loads(line)
            tz_json = json.dumps(tz_json)
            f.close()
        return tz_json

    def comput_IIiii(self):
        '返回pandas、字典串、列表'

        # print("运行commput")

        for i in self.grouped_IIiii.size().index:
            # print("i:",i)
            data = self.grouped_IIiii.get_group(i)
            data['VV'].replace(-9999, np.nan, inplace=True)
            data['RR'].replace(-9999, np.nan, inplace=True)
            data['Tn'].replace(-9999, np.nan, inplace=True)
            data['Tx'].replace(-9999, np.nan, inplace=True)
            self.data_gevent(data)
        self.RR_rank = [
            self.station_RR_small,
            self.station_RR_mid,
            self.station_RR_big,
            self.station_RR_huge,
            self.station_RR_bighuge,
            self.station_RR_more
        ]
        self.fFy_rank = [
            self.station_wind7,
            self.station_wind8,
            self.station_wind9,
            self.station_wind10,
            self.station_wind11,
            self.station_wind12,
            self.station_wind13,
            self.station_wind14,
            self.station_wind15,
            self.station_wind16,
            self.station_wind17
        ]
        self.vv_rank = [
            self.station_VV_small,
            self.station_VV_mid,
            self.station_VV_big,
            self.station_VV_huge
        ]

        # g = gevent.spawn(data_gevent, data)

    # 数据排序
    def return_data_sort(self, sort_data, value_str):
        max_sort = max(sort_data[value_str])
        level_sort = np.linspace(start=0.0, stop=max_sort, num=9)
        sort_da = pd.DataFrame(sort_data)
        if value_str == "rsum" or 'rmax':
            sort_da['index'] = sort_da[value_str].rank(ascending=0, method='dense')
            sort_out = sort_da.sort_values(by=[value_str], ascending=[False])
            list_data = []
            for row in sort_out.itertuples():
                dic_iter = {'index': int(getattr(row, 'index')), 'IIiii': str(getattr(row, 'name')),
                            'county': getattr(row, 'county'), 'town': getattr(row, 'town'),
                            'data': getattr(row, value_str),
                            'value': [getattr(row, 'lon'), getattr(row, 'lat'), getattr(row, value_str)]}
                list_data.append(dic_iter)
        if value_str == "fFy":
            sort_da['index'] = sort_da[value_str].rank(ascending=0, method='dense')
            sort_out = sort_da.sort_values(by=[value_str], ascending=[False])
            list_data = []
            for row in sort_out.itertuples():
                dic_iter = {'index': int(getattr(row, 'index')), 'IIiii': str(getattr(row, 'name')),
                            'county': getattr(row, 'county'), 'town': getattr(row, 'town'),
                            'data': getattr(row, value_str),
                            'value': [getattr(row, 'lon'), getattr(row, 'lat'), getattr(row, value_str)],
                            'symbol': str(self.symbol_ffy[0]),
                            'symbolRotate': getattr(row, 'dFy')
                            }
                list_data.append(dic_iter)
        return list_data, level_sort

    # 数据绘图

    def plot_imd(self, plot_data, value_str):
        # value_str = 'rsum'
        lat = plot_data['lat']
        lon = plot_data['lon']
        value = plot_data[value_str]
        func.plot_image(lat, lon, value)
        buffer = BytesIO()
        plt.savefig(buffer, bbox_inches='tight')
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img)
        ims = imb.decode()
        imd = "data:image/png;base64," + ims
        return imd

    # 返回第一个页面的回调数据
    def data_output(self):
        # 排序
        RR_sum, level_rain = self.return_data_sort(self.plot_data, 'rsum')
        RR_rx, level_rmax = self.return_data_sort(self.plot_data, 'rmax')
        data_fFy_list, level_fFy = self.return_data_sort(self.plot_data, 'fFy')
        # 柱状图
        RR_station_rank = self.RR_rank
        nation_station = ['58660', '58666', 'K8505', 'K8206', '58665', '58559', '58655', 'K8271', '58662', '58653']
        tmp_station_bar = []
        tmp_station_bar.append(['product', '最高气温', '最低气温'])
        RR_station_bar = []
        RR_station_bar.append(['product', '累计降水', '最大雨强'])
        # 计算指标站nation_station的要素值 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in nation_station:
            # print(i)
            var_name = str(self.station_list[i]['name'])
            var_tmax = max(self.station_list[i]['T']) / 10.0
            var_tmin = min(self.station_list[i]['T']) / 10.0
            var_rsum = self.station_list[i]['rsum']
            var_rmax = self.station_list[i]['rmax']
            tmp_station_bar.append([var_name, var_tmax, var_tmin])
            RR_station_bar.append([var_name, var_rsum, var_rmax])
        # 散点图
        tmp_min_scatter = self.data_station
        tmp_max_scatter = self.data_station
        tmp_event_scatter = self.data_station
        VV_min_scatter = self.data_station
        fFy_wind7up_scatter = data_fFy_list
        # print(fFy_wind7up_scatter)
        # fFy_wind7up_scatter = self.data_station
        # 要素极值站点序列数据
        vv_time = self.station_list[self.vv_min_name]['time']
        vv_value = self.station_list[self.vv_min_name]['V']
        data_vvmin = pd.DataFrame()
        data_vvmin['tTime'] = vv_time
        data_vvmin['VV'] = vv_value
        data_vvmin.sort_values(by='tTime')
        tz_json = self.read_shp_json()
        imd = self.plot_imd(self.plot_data, 'rsum')
        imd_tmax = self.plot_imd(self.plot_data, 'tmax')
        imd_tmin = self.plot_imd(self.plot_data, 'tmin')

        return imd, imd_tmax, imd_tmin, tz_json, RR_sum, RR_rx, level_rain, RR_station_rank, RR_station_bar, tmp_min_scatter, tmp_max_scatter, tmp_event_scatter, tmp_station_bar, VV_min_scatter, fFy_wind7up_scatter, vv_time, vv_value, data_fFy_list

# 自定义画图类
class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = asarray(levels, dtype='float64')
        self._x = self.levels/ self.levels.max()
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._y = linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)

class canvas_plot:
    def __init__(self, plot_self_data,plot_title,plot_bar):
        self.plot_self_data = plot_self_data
        self.plot_title = plot_title
        self.plot_bar = plot_bar
        self.data = self.read_data()
    # 外部函数
    def transform_from_latlon(self,lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale    
    def rasterize(self,shapes, coords, latitude='lat', longitude='lon',fill=np.nan, **kwargs):
        transform = self.transform_from_latlon(coords[latitude], coords[longitude])
        out_shape = (len(coords[latitude]), len(coords[longitude]))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
        spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
        return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))
    def add_shape_coord_from_data_array(self,xr_da, shp_path, coord_name):   
        shp_gpd = gpd.read_file(shp_path)
        shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]
        xr_da[coord_name] = self.rasterize(shapes, xr_da.coords, longitude='lon', latitude='lat')
        return xr_da
    def basemask(self,cs, ax, map, shpfile):
        sf = shapefile.Reader(shpfile)
        vertices = []
        codes = []
        for shape_rec in sf.shapeRecords():
            if shape_rec.record[0] >= 0:  
                pts = shape_rec.shape.points
                prt = list(shape_rec.shape.parts) + [len(pts)]
                for i in range(len(prt) - 1):
                    for j in range(prt[i], prt[i+1]):
                        vertices.append(map(pts[j][0], pts[j][1]))
                    codes += [Path.MOVETO]
                    codes += [Path.LINETO] * (prt[i+1] - prt[i] -2)
                    codes += [Path.CLOSEPOLY]
                clip = Path(vertices, codes)
                clip = PathPatch(clip, transform = ax.transData)    
        for contour in cs.collections:
            contour.set_clip_path(clip)    
    def makedegreelabel(self,degreelist):
        labels=[str(x)+u'°E' for x in degreelist]
        return labels
    def colormap(self,levels):
        '''色标的自定义'''
        plt.rcParams['axes.facecolor']='snow'
        colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 降水
        levels = [0,1,10,15,25,50,100,250]
        cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
        cmap_nonlin = nlcmap(cmaps, levels)
        #  
        #levels = [10,20,30,40,50,60,70,80] 
        return cmap_nonlin ,levels
    def read_data(self):
        data = self.plot_self_data
        a = []
        b = []
        z = []
        for i in range(len(data['station'])):
            if data['station'][i][2]!=-9999.0:
                a.append(data['station'][i][0])
                b.append(data['station'][i][1])
                z.append(data['station'][i][2])
        lat = np.array(b)
        lon = np.array(a)
        Zi = np.array(z)
        data_max = max(Zi)
        data_min = min(Zi)
        np.set_printoptions(precision = 2)
        x = np.arange(120.0,122.0,0.05)
        y = np.arange(27.8,29.5,0.05)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        data_xr = xr.DataArray(Z_linear, coords=[ y,x], dims=["lat", "lon"])
        return data_xr    
    def village_data(self):
        data_xr = self.data
        shp_path = "static/data/shpfile/"
        shp_data = gpd.read_file("static/data/shpfile/xiangzhen.shp", encoding='utf8')
        village_list = shp_data['NAME'].values
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"xiangzhen.shp", "country")
        data_dir = {}
        for i in range(len(village_list)):
            awash_da = shp_da.where(shp_da.country==i, other=np.nan)
            name = village_list[i]
            if np.isnan(awash_da.mean().values.tolist()):
                data_dir[name]  = 0.0
            else:
                data_dir[name]  = round(awash_da.mean().values.tolist(), 2)
        key = []
        value = []
        for item in data_dir.items():
            key.append(item[0])
            value.append(item[1])     
        return key,value    
    def shp_average(self):
        data_xr = self.data
        shp_path = "static/data/shpfile/"
        shp_data = gpd.read_file("static/data/shpfile/xian.shp", encoding='utf8')
        county_list = shp_data['NAME'].values
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"xian.shp", "country")
        data_list = []
        for i in range(len(county_list)):
            awash_da = shp_da.where(shp_da.country==i, other=np.nan)
            if np.isnan(awash_da.mean().values.tolist()):
                data_list  = 0.0
            else:
                data_list.append(awash_da.mean().values.tolist())
            
        return data_list
    def plot_img(self):
        '''绘制自定义图'''
        data_xr = self.data
        # ##########色标和大小#############################
        cmaps ,levels = self.colormap(self.plot_bar)
        fig = plt.figure(figsize=[10,10]) 
        ax = fig.add_subplot(111)
        # shp_path = "static/data/shpfile/"
        shp_path = "static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"taizhou.shp", "country")
        awash_da = shp_da.where(shp_da.country<7, other=np.nan)
        lat = data_xr.lat
        lon = data_xr.lon
        m = Basemap(llcrnrlon=120.0,
            llcrnrlat=27.8,
            urcrnrlon=122,
            urcrnrlat=29.5,
            resolution = None, 
            projection = 'cyl')
        lons, lats = np.meshgrid(lon, lat)
        cs =m.contourf(lons,lats,data_xr,ax=ax, cmap=cmaps,levels =levels)
        ##########标题#############################
        font = FontProperties(fname="static/data/simkai.ttf", size=14)
        # 为matplotlib中文无法显示设置字体
        #plt.rcParams['font.sans-serif'] = 'SimHei' # 黑体
        label  = self.plot_title
        plt.text(120.2,29.4, label,fontsize=15, fontproperties=font)
        ##########标题#############################
        m.readshapefile(shp_path+'taizhou','taizhou',color='k',linewidth=1.2)
        plt.axis('off') 
#         parallels = np.arange(27.8,29.5,0.2)
#         m.drawparallels(parallels,labels=[True,False,True,False],color='dimgrey',dashes=[2, 3],fontsize= 12)  # ha= 'right'
#         meridians = np.arange(120.0,122.0,0.2)
#         m.drawmeridians(meridians,labels=[False,True,False,True],color='dimgrey',dashes=[2, 3],fontsize= 12)
        len_lat = len(data_xr.lat.data)
        len_lon = len(data_xr.lon.data)
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                y0 = round(27.80+j*0.05,2)
                x0 = round(120.00+i*0.05,2)
                if not isnan(awash_da.data[j,i]):
                    plt.text(x0,y0,str(int(awash_da.data[j,i])),fontsize= 7,fontweight = 800 ,color ="black")            
        # 在图上绘制色标
        rect1 = [0.35, 0.25, 0.03, 0.12]         
        ax2 = plt.axes(rect1,frameon='False')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        m.colorbar(cs, location='right', size='30%', pad="-100%",ax = ax2)
        self.basemask(cs, ax, m, shp_path+'taizhou') 
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight')  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        return imd



# 画图的核心类
class plot_tz_product:
    def __init__(self, plot_type, plot_time):
        self.plot_time = plot_time
        self.plot_type = plot_type
        self.time_len = 0
        self.lat, self.lon, self.time, self.data_xr_nc,self.date = self.read_data()
    # 外部函数
    def transform_from_latlon(self, lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale

    def rasterize(self, shapes, coords, latitude='lat', longitude='lon', fill=np.nan, **kwargs):
        transform = self.transform_from_latlon(coords[latitude], coords[longitude])
        out_shape = (len(coords[latitude]), len(coords[longitude]))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                    fill=fill, transform=transform,
                                    dtype=float, **kwargs)
        spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
        return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))

    def add_shape_coord_from_data_array(self, xr_da, shp_path, coord_name):
        shp_gpd = gpd.read_file(shp_path)
        shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]
        xr_da[coord_name] = self.rasterize(shapes, xr_da.coords, longitude='lon', latitude='lat')
        return xr_da

    def basemask(self, cs, ax, map, shpfile):
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

    def makedegreelabel(self, degreelist):
        labels = [str(x) + u'°E' for x in degreelist]
        return labels

    def read_data(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
        file_path = "static/data/TZ_self/"
        # file_name = file_path +"20220401/"+'I20220401080000.'+self.plot_type+'.nc'
        file_name = file_path + "20220402/" + 'I20220402080000.' + self.plot_type + '.nc'
        # f = xr.open_dataset(file_name)
        f = netCDF4.Dataset(file_name, "r", format="NETCDF4")
        data_xr_nc = f.variables[str(self.plot_type)]
        lat = f.variables['lat'][:]
        lon = f.variables['lon'][:]
        time = f.variables['time'][:]
        date = "20220402"
        self.time_len = len(time)
        return lat, lon, time, data_xr_nc,date
    def return_data(self,item):
        lat = self.lat
        lon = self.lon
        time = self.time
        data_xr_nc = self.data_xr_nc
        #print("计算time",time)
        data_xr = xr.DataArray(data_xr_nc[item,:,:],coords=[lat,lon], dims=["lat", "lon"])       
        #以下是核心api,实质为调用Cmaps基类的listmap()方法
        basicfile = '/home/liyuan3970/Data/My_Git/web_met/'
        shp_path = basicfile+"/static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"taizhou.shp", "test")
        awash_da = shp_da.where(shp_da.test<7, other=0)
        len_lat = len(data_xr.lat.data)
        len_lon = len(data_xr.lon.data)
        data = []
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                y0 = round(27.50+j*0.05,2)
                x0 = round(119.80+i*0.05,2)
                single = {
                    "type": "Feature",
                    "properties": {
                        "value": str(data_xr.data[j, i])
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [x0, y0]
                    }
                }
                data.append(single)
        return data        
    def multy_plot(self):
        '''返回图片列表'''
        imd_list = []
        for i in range(self.time_len):
            imd = self.return_data(i)
            imd_list.append(imd)
        return imd_list
    def label_index(self,item):
        hours_list = self.time.data.tolist()
        year = int(self.date[0:4])
        month = int(self.date[4:6])
        day = int(self.date[6:8]) 
        hour = int(self.plot_time)
        inittime = dtt.datetime(year, month, day, hour)
        if item==0:
            start = inittime 
            end = start + dtt.timedelta(hours = hours_list[item]) 
        else: 
            start = inittime + dtt.timedelta(hours = int(hours_list[item-1])) #dtt.datetime(year, month, day, hour)
            end = start + dtt.timedelta(hours = hours_list[item-1]) 
        label = str(start)[0:16] +" - "+str(end)[0:16]+" "+self.plot_type
        return label
    def btn_index(self):
        buttn_list = []
        hours_list = self.time.data.tolist()
        year = int(self.date[0:4])
        month = int(self.date[4:6])
        day = int(self.date[6:8]) 
        hour = int(self.plot_time)
        inittime = dtt.datetime(year, month, day, hour)
        for i in hours_list:
            start = inittime + dtt.timedelta(hours = int(i))
            label = str(start)[5:7] +"月"+str(start)[8:10] +"日"+str(start)[11:13]+"时"
            buttn_list.append(label)
        return buttn_list
# 自动站数据查询的class

class zdz_data:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        # 基础数据
        self.station_data = None
        self.day_list = None
        self.rain_line = None
        self.rain_scatter = None
        self.station_all = self.read_csv()
        self.img = None
        self.event_data()
        self.rain_data()
    def plot_img(self,lat,lon,value):
        #线性插值
        N = len(value)
        a = []
        b = []
        z = []
        for i in range(N):
            if value[i]!=-9999.0:
                a.append(round(lon[i],2))
                b.append(round(lat[i],2))
                z.append(round(value[i],2))
        lat = np.array(b)
        lon = np.array(a)
        Zi = np.array(z)
        data_max = max(Zi)
        data_min = min(Zi)
        np.set_printoptions(precision = 2)
        x = np.arange(120.0,122.0,0.05)
        #print(x)
        y = np.arange(27.8,29.5,0.05)
        nx0 =len(x)
        ny0 =len(y)
        X, Y = np.meshgrid(x, y)#100*100
        P = np.array([X.flatten(), Y.flatten() ]).transpose()    
        Pi =  np.array([lon, lat ]).transpose()
        Z_linear = griddata(Pi, Zi, P, method = "nearest").reshape([ny0,nx0])
        #绘图
        colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 降水
        cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
        levels = [0,10,25,50,70,90,110,230]
        # plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=[12,16]) 
    
        ax = fig.add_subplot(111)
        # plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        filepath = "/home/liyuan3970/Data/My_Git/web_met/static/data/shpfile/"
        data_xr = xr.DataArray(Z_linear/10.0, coords=[ y,x], 
                        dims=["lat", "lon"])
        shp_da = func.add_shape_coord_from_data_array(data_xr, filepath+"taizhou.shp", "test")
        awash_da = shp_da.where(shp_da.test<7, other=np.nan)
        m = Basemap(llcrnrlon=120.0,llcrnrlat=27.8,urcrnrlon=122,urcrnrlat=29.5,resolution = None, projection = 'cyl')
        # 设置colorbar
        cbar_kwargs = {'shrink': 0.5}  
        lons, lats = np.meshgrid(x, y)
        cs =m.contourf(lons,lats,data_xr,ax=ax, cmap=cmaps)
        #cs = data_xr.plot.contourf(ax=ax, cmap=cmaps,levels =levels,cbar_kwargs=cbar_kwargs,add_labels=False)
        m.readshapefile(filepath+'taizhou','taizhou',color='k',linewidth=1.2)
        plt.axis('off')
        len_lat = len(data_xr.lat.data)
        len_lon = len(data_xr.lon.data)
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                y0 = round(27.8+j*0.05,2)
                x0 = round(120.0+i*0.05,2)
                if not isnan(awash_da.data[j,i]):
                    plt.text(x0,y0,str(int(awash_da.data[j,i])),fontsize= 7,fontweight = 800 ,color ="black")
        # 在图上绘制色标
        rect1 = [0.35, 0.29, 0.05, 0.12]         
        ax2 = plt.axes(rect1,frameon='False' )
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        m.colorbar(cs, location='right', size='30%', pad="-100%",ax = ax2)
        func.basemask(cs, ax, m, filepath+'taizhou')  
        # 保存为base64数据
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight')  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd_rain = "data:image/png;base64,"+ims
        return imd_rain
    def sql_date(self):
        '''数据库读取sql数据'''
        print('读取数据库数据')

    def read_csv(self):
        '''
        1.数据读取---此方法为测试方法
        2.数据分段---每天
        3.每天统计
        '''
        #         station_Mws = pd.read_csv("Mws_15.csv")
        #         station_Aws = pd.read_csv("Aws_15.csv")
        #         station_all = pd.concat([station_Aws, station_Mws])
        # station_all = pd.read_csv("data_zdz_height.csv")
        station_all = pd.read_csv("static/data/data_zdz_height.csv")
        return station_all

    def event_data(self):
        '''
        1.数据库查询
        2.数据分段---每天
        3.每天统计
        ['2022-04-14', 200,'高温 浓雾', '降水 大风', 200],
        '''
        dateList = []
        # 多少天
        dates = self.return_daylist()
        for i in dates:
            # 解析每天的数据
            # grouped_county = self.station_all.groupby('county')
            # grouped_IIiii = self.station_all.groupby('IIiii')
            # for i in self.grouped_county.size().index: 
            s_date = i + ' ' + '00:00'
            e_date = i + ' ' + '23:00'
            df_preday = self.station_all[(self.station_all['tTime'] >= s_date) & (self.station_all['tTime'] <= e_date)]
            #             df_RR= df_preday[df_preday['RR'] == -9999 ].count()
            #             grouped_county =
            date_preday = []
            # 日期
            day_date = str(i)
            # 面雨量
            rainfall = 200
            # 气温
            temp_min = df_preday[(df_preday['Height'] < 6000) & (df_preday['T'] > -400)]['T'].min()
            temp_min = temp_min / 10.0
            temp_max = df_preday[(df_preday['Height'] < 6000) & (df_preday['T'] > -400)]['T'].max()
            temp_max = temp_max / 10.0
            temp_label = ''
            if temp_min < 3.0:
                temp_label = '低温'
            elif temp_max > 35.0:
                temp_label = '高温'
            else:
                temp_label = ' '
            # 能见度
            # df_VV= df_preday[(df_preday['VV']>0.1)&(df_preday['VV']<500)]['VV'].count()
            vv = df_preday[(df_preday['VV'] > 0.1) & (df_preday['VV'] < 500)]['VV'].count()
            if vv > 0:
                view_label = '浓雾'
            else:
                view_label = ' '
                # 降水
            df_RR = df_preday[(df_preday['RR'] > 0.3) & (df_preday['RR'] < 8888)]['RR'].count()

            if df_RR > 0:
                pre_label = '降水'
            else:
                pre_label = ' '
            # 大风
            wind_count = df_preday[df_preday['fFy'] > 187]['fFy'].count()
            if wind_count > 0:
                wind_label = '大风'
            else:
                wind_label = ' '
            date_preday = [day_date, rainfall, temp_label + ' ' + view_label, pre_label + ' ' + wind_label, rainfall]
            dateList.append(date_preday)
        self.day_list = dateList

    #         print('event_data:返回一个每天的数组和日历所需的数据','气温',dateList)
    def return_daylist(self):
        '''
        返回每天的起始列表
        '''
        dates = []
        #         dt = datetime.datetime.strptime(self.start, "%Y-%m-%d")
        dt = dtt.datetime.strptime(self.start[0:10], "%Y-%m-%d")
        date = self.start[:10]
        while date <= self.end[:10]:
            dates.append(date)
            dt = dt + dtt.timedelta(1)
            date = dt.strftime("%Y-%m-%d")
        return dates

    def rain_data(self):
        '''
        1.根据起始时间计算面雨量的sql语句
        2.根据初始语句返回每站的总降水量
        '''
        station_all = self.station_all

        data_rain = {
            'rain_sum': {
                'time': [],
                'data': []
            },
            'IIiii_data': {}

        }
        # 县市区单站降水
        time_index = []
        station_k8734 = []
        station_k8748 = []
        grouped_tTime = station_all.groupby('tTime')
        for i in grouped_tTime.size().index:
            data = grouped_tTime.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            rain_mean = data['RR'].mean() / 10.0
            rain_time = i
            data_rain['rain_sum']['time'].append(rain_time)
            data_rain['rain_sum']['data'].append(rain_mean)
            time_index.append(rain_time)
            station_k8748.append(data[data['IIiii']=='K8748']['RR'].values[0]/1.0 if len(data[data['IIiii']=='K8748']['RR'].values)==1 else 0.0)
            station_k8734.append(data[data['IIiii']=='K8734']['RR'].values[0]/1.0 if len(data[data['IIiii']=='K8734']['RR'].values)==1 else 0.0)
        self.rain_line = [time_index,station_k8734,station_k8748]
        # 导出单站数据
        grouped_IIiii = station_all.groupby('IIiii')
        rain_scatter = []
        #         {
        #          name: "K8515", value: [121.2, 28.6, 110],
        #          symbol: 'circle'
        #         }
        lat = []
        lon = []
        value = []
        station_list = ['K8116','K8748']
        station_line = []
        pre_county_list = {}
        for i in grouped_IIiii.size().index:
            data = grouped_IIiii.get_group(i)
            data['RR'].replace(-9999,np.nan, inplace=True)# np.nan
            station_name = str(i)
            data_rain['IIiii_data'][station_name] = data
            single_data = {}
            single_data['name'] = station_name
            single_data['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['RR'].sum() / 10.0]
            lat.append(data['lat'].iloc[0])
            lon.append(data['lon'].iloc[0])
            value.append(data['RR'].sum())
            single_data['symble'] = 'circle'
            rain_scatter.append(single_data)  
        self.station_data = data_rain['IIiii_data']
        self.rain_scatter = rain_scatter
        self.img = self.plot_img(lat,lon,value)
    def wind_data(self):
        '''
        1.根据sql语句计算8及以上大风的分布和排序
        '''
        data_wind_list = []
        station_all = self.station_all
        sort_data = {
            'IIiii': [],
            'county': [],
            'town': [],
            'value': []
        }
        # wind_max = station_all['fFy'].idxmax(axis=0)
        # station_max_name = station_all.iloc[wind_max,7]
        # station_max_data = self.station_data[station_max_name]
        grouped_IIiii = station_all.groupby('IIiii')
        for i in grouped_IIiii.size().index:
            data = grouped_IIiii.get_group(i)
            if data['fFy'].max() > 187:
                data_single = {}
                data_single['IIiii'] = data['IIiii'].iloc[0]
                data_single['county'] = data['county'].iloc[0]
                data_single['town'] = data['Town'].iloc[0]
                data_single['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['fFy'].max() / 10.0]
                data_single['symbol'] = 'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'
                data_single['symbolRotate'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
                sort_data['IIiii'].append(data['IIiii'].iloc[0])
                sort_data['county'].append(data['county'].iloc[0])
                sort_data['town'].append(data['Town'].iloc[0])
                sort_data['value'].append(data['fFy'].max() / 10.0)
                data_wind_list.append(data_single)
        # 对数据进行排序
        max_sort = max(sort_data['value'])
        level_sort = np.linspace(start=0.0, stop=max_sort, num=9)
        sort_data = pd.DataFrame(sort_data)
        sort_data['index'] = sort_data['value'].rank(ascending=0, method='dense')
        sort_out = sort_data.sort_values(by=['value'], ascending=[False])
        sort_html = ''
        # <tr><th>排序</th><th>乡镇</th><th>站点</th><th>能见度</th></tr>
        for i in sort_out['index']:
            table_html = '<tr><th>' + str(int(i)) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['town'].iloc[0]) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['IIiii'].iloc[0]) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['value'].iloc[0]) + '</th></tr>'
            sort_html = sort_html + table_html
        return data_wind_list, sort_html

    def view_data(self):
        '''
        1.根据sql语句计算低能见度的分布和排序
        '''
        '''
        1.根据sql语句计算8及以上大风的分布和排序
        '''
        data_view_list = []
        station_all = self.station_all
        sort_data = {
            'IIiii': [],
            'county': [],
            'town': [],
            'value': []
        }
        # wind_max = station_all['fFy'].idxmax(axis=0)
        # station_max_name = station_all.iloc[wind_max,7]
        # station_max_data = self.station_data[station_max_name]
        grouped_IIiii = station_all.groupby('IIiii')
        for i in grouped_IIiii.size().index:
            data = grouped_IIiii.get_group(i)
            data['VV'].replace(-9999, 10000, inplace=True)
            if data['VV'].min() < 500 :
                data_single = {}
                data_single['IIiii'] = data['IIiii'].iloc[0]
                data_single['county'] = data['county'].iloc[0]
                data_single['town'] = data['Town'].iloc[0]
                data_single['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['VV'].min()/1.0]
                sort_data['IIiii'].append(data['IIiii'].iloc[0])
                sort_data['county'].append(data['county'].iloc[0])
                sort_data['town'].append(data['Town'].iloc[0])
                sort_data['value'].append(data['VV'].min()/1.0 )
                data_view_list.append(data_single)
        # 对数据进行排序
        max_sort = max(sort_data['value'])
        level_sort = np.linspace(start=0.0, stop=max_sort, num=9)
        sort_data = pd.DataFrame(sort_data)
        sort_data['index'] = sort_data['value'].rank(ascending=0, method='dense')
        sort_out = sort_data.sort_values(by=['value'], ascending=[False])
        sort_html = ''
        # <tr><th>排序</th><th>乡镇</th><th>站点</th><th>能见度</th></tr>
        for i in sort_out['index']:
            table_html = '<tr><th>' + str(int(i)) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['town'].iloc[0]) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['IIiii'].iloc[0]) + '</th><th>' + \
                         str(sort_out[sort_out['index'] == i]['value'].iloc[0]) + '</th></tr>'
            sort_html = sort_html + table_html
        return data_view_list, sort_html

    def temp_data(self):
        '''
        1.根据sql语句计算高低温的分布和排序
        '''
        temp_station_list =[
            "K8705","K8706","K8903","K8818","K8821",
            "K8609","K8282","K8217","K8201","K8301",
            "K8413","K8611","K8505"
        ]
        data_temp_max = []
        data_temp_min = []
        for i in temp_station_list:
            single_data_max = {}
            single_data_min = {}
            station_name = str(i)
            single_data_max['name'] = str(i)
            single_data_max['value'] = [self.station_data[i]['lon'].iloc[0], self.station_data[i]['lat'].iloc[0],
                                        self.station_data[i]['T'].max() / 10.0]
            single_data_min['name'] = i
            single_data_min['value'] = [self.station_data[i]['lon'].iloc[0], self.station_data[i]['lat'].iloc[0],
                                        self.station_data[i]['T'].min() / 10.0]
            data_temp_max.append(single_data_max)
            data_temp_min.append(single_data_min)
        return data_temp_max, data_temp_min

    def pre_day(self, date):
        '''
        日报的响应时间、触发后统计对应天的灾情
        '2022-04-14', 200,'高温 浓雾', '降水 大风', 
        '''
        # date = ['2022-04-14', 200,'低温 浓雾', '降水 大风', 200]
        # 计算时间
        today = dtt.datetime.strptime(date, '%Y-%m-%d')
        start_time = str(today + dtt.timedelta(days=-1))[0:10] + ' 20:00'
        end_time = str(today + dtt.timedelta(days=1))[0:10] + ' 08:00'
        data_all = self.station_all
        data_time = data_all[(data_all['tTime'] >= start_time) & (data_all['tTime'] <= end_time)]
        time_index  = pd.date_range(start=start_time,end=end_time,freq='1H')
        timeindex =  [str(i)[5:16].replace('T' , '-') for i in time_index.values]
        grouped_IIiii = data_time.groupby('IIiii')
        # 指标站名称
        station_plot = [
            "K8705","K8706","K8903","K8818","K8821",
            "K8609","K8282","K8217","K8201","K8301",
            "K8413","K8611","K8505"
        ]
        # 所需数据库
        pre_list = []
        wind_list = []
        view_list = []
        tmax_list = []
        tmin_list = []
        

        for i in grouped_IIiii.size().index:
            data_rr = grouped_IIiii.get_group(i)
            data_rr['RR'].replace(-9999, 0.0, inplace=True)
            data_rr['VV'].replace(-9999, 9999, inplace=True)
            data = data_rr.sort_values('tTime') 
            ## 指标站的排序
            # 低温 
            if data['IIiii'].iloc[0] in station_plot:
                tmin_dir = {}
                tmin_dir['name'] = data['IIiii'].iloc[0]
                tmin_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], timeindex, data['T'].tolist(), data['T'].min()]
                tmin_list.append(tmin_dir)
            # 高温   
                tmax_dir = {}
                tmax_dir['name'] = data['IIiii'].iloc[0]
                tmax_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0],timeindex, data['T'].tolist(), data['T'].max()]
                tmax_list.append(tmax_dir)
            # 降水
                pre_dir = {}
                pre_dir['name'] = data['IIiii'].iloc[0]
                pre_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], timeindex,data['RR'].tolist(), data['RR'].sum()]
                pre_dir['symble'] = "circle"
                pre_list.append(pre_dir)
            # 大风  
            if data['fFy'].max()>187:
                wind_dir = {}
                wind_line = []
                wind_dir['name'] = data['IIiii'].iloc[0]
                wind_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0],
                                     timeindex,data['dFy'].tolist(),data['fFy'].tolist()]
                wind_dir['symbol'] =  'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'    
                wind_dir['symbolSize'] = 15
                wind_dir['symbolRotate'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]    
                for j in range(len(data['dFy'].tolist())):
                    wind_line_dir = {}
                    wind_line_dir['symbol'] = 'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'
                    wind_line_dir['symbolRotate'] = data['dFy'].tolist()[j]
                    wind_line_dir['value'] = data['fFy'].tolist()[j]
                    wind_line.append(wind_line_dir)
                wind_dir['value'].append(wind_line)
                wind_dir['value'].append(data['fFy'].max())

                wind_list.append(wind_dir)
            # 能见度  ok 
            if data['VV'].min()<500:
                view_dir = {}
                view_dir['name'] = data['IIiii'].iloc[0]
                view_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0],timeindex, data['VV'].tolist(), data['VV'].min()]
                view_list.append(view_dir)
        pre_data = {
            'rain':pre_list,
            'tmax':tmax_list,
            'tmin':tmin_list,
            'view':view_list,
            'wind':wind_list,
        }
        return pre_data

    def text_data(self):
        '''用来处理风雨情统计数据'''
        start_time = self.start 
        end_time = self.end
        time_len =len(pd.date_range(start=start_time,end=end_time,freq='1H')) 
        station_all = self.station_all 
        # 计算面雨量
        grouped_county = station_all.groupby('county')        
        # 所需数据库
        pre_county = { }
        ## 
        iii_pre = {}
        iii_pre_data = {}
        iii_tmin_data = {}
        iii_tmax_data = {}
        iii_vvmin_data = {}
        iii_fmax_data = {}
        ##
        town_pre_data = {}
        # 面雨量   
        for i in grouped_county.size().index:
            data = grouped_county.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            ave = data['RR'].mean() * time_len /10.0
            pre_county[data['county'].iloc[0]] = ave
        #逐站统计    
        grouped_iii = station_all.groupby('IIiii') 
        for i in grouped_iii.size().index:
            data = grouped_iii.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            data['VV'].replace(-9999, np.nan, inplace=True)
            #data['fFy'].replace(-9999, np.nan, inplace=True)
            data['T'].replace(-9999, np.nan, inplace=True)
            # 存储变量
            iii_pre_data[data['IIiii'].iloc[0]] =  data['RR'].sum()/10.0 # 累计降水
            iii_tmin_data[data['IIiii'].iloc[0]] =  data['T'].min()      # 高温
            iii_tmax_data[data['IIiii'].iloc[0]] =  data['T'].max()      # 低温
            iii_vvmin_data[data['IIiii'].iloc[0]] =  data['VV'].min()    # 能见度
            iii_fmax_data[data['IIiii'].iloc[0]] =  np.nanmax(data['fFy'])# data['fFy'].max()    # 风力
        # 乡镇排序
        grouped_town = station_all.groupby('Town') 
        for i in grouped_town.size().index:
            data = grouped_town.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            town_pre_data[data['Town'].iloc[0]] =  data['RR'].sum()/10.0    
        # 文字排序的基本逻辑
        # 1. 降水
        # 2. 高低温
        # 3. 能见度
        # 4. 风
        # 5. 小时雨强
        ## 乡镇降水排序
        town_pre_sort = sorted(town_pre_data.items(), key=lambda x: x[1])

        ## 降水排序
        iii_pre_sort = sorted(iii_pre_data.items(), key=lambda x: x[1])

        ## 降水站点个数
        iii_pre_count = {
            '大于250毫米':len({k:v for k, v in iii_pre_data.items() if v>=51}.items()),
            '大于100毫米':len({k:v for k, v in iii_pre_data.items() if v>=100}.items()),
            '大于50毫米':len({k:v for k, v in iii_pre_data.items() if v>=50}.items()),
            '大于25毫米':len({k:v for k, v in iii_pre_data.items() if v>=25}.items()),
            '大于10毫米':len({k:v for k, v in iii_pre_data.items() if v>=10}.items()),
            '大于5毫米':len({k:v for k, v in iii_pre_data.items() if v>=5}.items()),
            '大于1毫米':len({k:v for k, v in iii_pre_data.items() if v>=0.1}.items())
        } 
        ## 风力
        iii_wind_sort = sorted(iii_fmax_data.items(), key=lambda x: x[1], reverse=True)
        iii_wind_count = {
            'wind_rank':[16,15,14,13,12,11,10,9,8],
            'wind_list':[
                len({k:v for k, v in iii_fmax_data.items() if v>=51 and v<56}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=47 and v<51}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=42 and v<47}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=37 and v<42}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=32 and v<37}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=27 and v<32}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=22 and v<27}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=17 and v<22}.items()),
                len({k:v for k, v in iii_fmax_data.items() if v>=12 and v<17}.items())      
            ] 
        } 
        ## 能见度
        iii_vv_sort = sorted(iii_vvmin_data.items(), key=lambda x: x[1])
        iii_vv_count = {
            'vv_rank':["小于50米的强浓雾","小于200米浓雾","小于500米的浓雾"],
            'vv_list':[
                len({k:v for k, v in iii_vvmin_data.items() if v>=0 and v<50}.items()),
                len({k:v for k, v in iii_vvmin_data.items() if v>=50 and v<200}.items()),
                len({k:v for k, v in iii_vvmin_data.items() if v>=200 and v<500}.items()),
    
            ] 
        } 
        ## 低温
        iii_tmin_sort = sorted(iii_tmin_data.items(), key=lambda x: x[1])
        iii_tmin_count = {
            'tmin_rank':["小于-3度","小于0度","小于3度"],
            'tmin_list':[
                len({k:v for k, v in iii_tmin_data.items() if v>=0 and v<3}.items()),
                len({k:v for k, v in iii_tmin_data.items() if v>=-3 and v<0}.items()),
                len({k:v for k, v in iii_tmin_data.items() if v>=-20 and v<-3}.items()),
    
            ] 
        } 
        ## 高温
        iii_tmax_sort = sorted(iii_tmax_data.items(), key=lambda x: x[1])
        iii_tmax_count = {
            'tmax_rank':["大于35度","小于38度","小于40度"],
            'tmax_list':[
                len({k:v for k, v in iii_tmax_data.items() if v>=0 and v<3}.items()),
                len({k:v for k, v in iii_tmax_data.items() if v>=-3 and v<0}.items()),
                len({k:v for k, v in iii_tmax_data.items() if v>=-20 and v<-3}.items()),
    
            ] 
        } 
        
        
        # 输出文档内容
        ## 时间开头
        text_all = "【风雨情通报】:"
        time_text = self.start + "到" + self.end
        ## 面雨量
        county_text = "各县市面雨量分别为:"
        for single in sorted(pre_county.items(), key=lambda x: x[1], reverse=True):
            single_text = single[0]+ "" + str(round(single[1],2)) + "毫米;"
            county_text = county_text + single_text  
        ## 雨量前五
        iii_pre_text = "单站前五：" + iii_pre_sort[-1][0] + "：" + str(iii_pre_sort[-1][1]) + "毫米,"\
            + iii_pre_sort[-2][0]+ "：" + str(iii_pre_sort[-2][1]) + "毫米,"\
            + iii_pre_sort[-3][0]+ "：" + str(iii_pre_sort[-3][1]) + "毫米,"\
            + iii_pre_sort[-4][0]+ "：" + str(iii_pre_sort[-4][1]) + "毫米,"\
            + iii_pre_sort[-5][0]+ "：" + str(iii_pre_sort[-5][1]) + "毫米."
        ## 乡镇降水前五
        town_pre_text = "乡镇前五：" + town_pre_sort[-1][0] + "：" + str(town_pre_sort[-1][1]) + "毫米,"\
            + town_pre_sort[-2][0]+ "：" + str(town_pre_sort[-2][1]) + "毫米,"\
            + town_pre_sort[-3][0]+ "：" + str(town_pre_sort[-3][1]) + "毫米,"\
            + town_pre_sort[-4][0]+ "：" + str(town_pre_sort[-4][1]) + "毫米,"\
            + town_pre_sort[-5][0]+ "：" + str(town_pre_sort[-5][1]) + "毫米."        
        ## 雨强前三
        ri_pre_text = ""
        ## 降水站点个数
        count_pre_text = "其中:"
        for key, value in iii_pre_count.items():
            if value!=0:
                count_pre_text = count_pre_text + key + "有" + str(value) + "站,"
        ## 大风
        wind_text = "沿海出现"
        wind_max = "16级"
        wind_min = "15"
        for i in range(len(iii_wind_count['wind_list'])):
            if iii_wind_count['wind_list'][i]>0:
                wind_max = str(iii_wind_count['wind_rank'][i]) + "级"
                break
        wind_reve = iii_wind_count['wind_list'][::-1]
        for i in wind_reve:
            if i>0:
                index = wind_reve.index(i)
                wind_min = str(iii_wind_count['wind_rank'][9-index])
                break  
        wind_text = wind_text + wind_min + "~" + wind_max + "大风." 
        wind_text = wind_text + "风力较大的有：" + iii_wind_sort[0][0] + "：" + str(iii_wind_sort[0][1]/10.0) + "米/秒,"\
            + iii_wind_sort[1][0]+ "：" + str(iii_wind_sort[1][1]/10.0) + "米/秒,"\
            + iii_wind_sort[2][0]+ "：" + str(iii_wind_sort[2][1]/10.0) + "米/秒,"\
            + iii_wind_sort[3][0]+ "：" + str(iii_wind_sort[3][1]/10.0) + "米/秒,"\
            + iii_wind_sort[4][0]+ "：" + str(iii_wind_sort[4][1]/10.0) + "米/秒."    
        ## 能见度
        vv_text = ""
        if iii_vv_count['vv_list'][0]>0:
            vv_text = vv_text + "沿海出现小于50米的强浓雾。"
        elif iii_vv_count['vv_list'][1]>0:
            vv_text = vv_text + "沿海出现小于200米的强浓雾。"
        else:
            vv_text = vv_text + "沿海出现小于500米的浓雾。"
        ## 温度
        temp_text = "主城区最高温度："
        temp_text = temp_text + str(iii_tmax_data['K8505']) + "度。" + "主城区最低温度：" + str(iii_tmin_data['K8505']) + "度。"
        temp_text = temp_text + "其中全市" 
        for i in range(len(iii_tmin_count['tmin_list'])):
            if iii_tmin_count['tmin_list'][i]:
                temp_text = temp_text + iii_tmin_count['tmin_rank'][i] + "的有" + str(iii_tmin_count['tmin_list'][i]) + "站。"
        for i in range(len(iii_tmax_count['tmax_list'])):
            if iii_tmax_count['tmax_list'][i]:
                temp_text = temp_text + iii_tmax_count['tmax_rank'][i] + "的有" + str(iii_tmax_count['tmax_list'][i]) + "站。"      
        text_all = text_all + time_text+county_text+iii_pre_text+town_pre_text+count_pre_text+wind_text+vv_text+temp_text
        #print("风雨情通报:",time_text,county_text,iii_pre_text,town_pre_text,count_pre_text,wind_text+vv_text,temp_text)
        return text_all




# ec数据的处理和对接
class ec_data_point:
    def __init__(self,start_time,end_time):
        self.timelist = [0,2,4,6,8,10,12,14,16,18,20,22,24,25,
                         26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                        41,42,43,44,45,46,47,48,49,50,51,52]
        self.test_time = '2022041700'
        self.file_path = "/workspace/liyuan3970/Data/My_Git/" + self.test_time + "/" 
        self.lat_list = [27.6, 28.1, 28.4, 29.1, 29.1, 29.8, 28.7, 28.5, 28.5, 28.6]
        self.lon_list = [120.7, 121.1, 121.3, 121.2, 121.0, 120.7, 121.1, 121.4, 121.4, 121.2]
        self.name = ["台州", "玉环", "温岭", "三门", "天台", "仙居", "临海", "路桥", "椒江", "黄岩"]
        self.name_en = ['taizhou','yuhuan','wenling','sanmen','tiantai','xianju','linhai','luqiao','jiaojiang','huangyan']
        self.cp,self.t2,self.lsp = self.read_data()
    # 外部函数
    def transform_from_latlon(self,lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale    
    def rasterize(self,shapes, coords, latitude='lat', longitude='lon',fill=np.nan, **kwargs):
        transform = self.transform_from_latlon(coords[latitude], coords[longitude])
        out_shape = (len(coords[latitude]), len(coords[longitude]))
        raster = features.rasterize(shapes, out_shape=out_shape,
                                fill=fill, transform=transform,
                                dtype=float, **kwargs)
        spatial_coords = {latitude: coords[latitude], longitude: coords[longitude]}
        return xr.DataArray(raster, coords=spatial_coords, dims=(latitude, longitude))
    def add_shape_coord_from_data_array(self,xr_da, shp_path, coord_name):   
        shp_gpd = gpd.read_file(shp_path)
        shapes = [(shape, n) for n, shape in enumerate(shp_gpd.geometry)]
        xr_da[coord_name] = self.rasterize(shapes, xr_da.coords, longitude='lon', latitude='lat')
        return xr_da
    def regrid(self,data):
        # 插值
        ds_out = xr.Dataset(
            {   
                
                "lat": (["lat"], np.arange(27.0, 31, 0.05)),
                "lon": (["lon"], np.arange(120, 122.9, 0.05)),
            }
        )
        regridder = xe.Regridder(data, ds_out, "bilinear")
        dr_out = regridder(data)
        return dr_out
    # 读取数据
    def read_data(self):
        '''读取数据'''
        files = os.listdir(self.file_path)
        lsp_list = [] 
        cp_list = [] 
        t2_list = [] 
        for fileitem in self.timelist:
            f=xr.open_dataset(self.file_path +files[fileitem],decode_times=False)
            lsp = f.lsp.sel(lonS=slice(118,123),latS=slice(32,26))
            cp = f.cp.sel(lonS=slice(118,123),latS=slice(32,26))
            t2 = f.t2.sel(lonS=slice(118,123),latS=slice(32,26)) 
            lsp_list.append(lsp)  
            cp_list.append(cp)  
            t2_list.append(t2)
            del f,lsp,cp,t2  
        cp_all = xr.concat(cp_list,dim="time")
        cp_all = self.regrid(cp_all)
        t2_all = xr.concat(t2_list,dim="time")
        t2_all = self.regrid(t2_all)
        lsp_all = xr.concat(lsp_list,dim="time")
        lsp_all = self.regrid(lsp_all)
        return cp_all,t2_all,lsp_all
    # 曲线的读取
    def accum_data(self,list_data):
        '''处理累计降水'''
        out_list = []
        for i in range(len(list_data)):
            if i==0:
                out_list.append(0)
            elif i==1:
                out_list.append(round(list_data[i],1))
            else:
                out_list.append(round(list_data[i]-list_data[i-1],1))
        return out_list
    def rain_data(self,lat,lon,data):
        list_data= data.sel(lon=lon, lat=lat,method='nearest').to_pandas().tolist()
        out_list = self.accum_data(list_data)
        return out_list
    def plot_line(self,lat,lon):
        '''返回单点的降水气温曲线图'''
        cp_line = self.rain_data(lat,lon,self.cp)
        totle_line = self.rain_data(lat,lon,self.lsp)
        lsp_line = []
        for i in range(len(totle_line)):
            lsp_line.append(totle_line[i] - cp_line[i])
        t2_line = self.t2.sel(lon=lon, lat=lat,method='nearest').to_pandas().tolist()
        return cp_line,lsp_line,t2_line
    def conuty_data(self):
        '''将数据整理成json并存储到MySQL'''
        data_list = []
        for i in range(len(self.lat_list)):
            lat = self.lat_list[i]
            lon = self.lon_list[i]
            cp_line,lsp_line,t2_line = self.plot_line(lat,lon)
            data_single = {
                "name":self.name_en[i],
                "cp":cp_line,
                "lsp":lsp_line,
                "t2":t2_line
            }
            data_list.append(data_single)
        return data_list
    # 定时任务
    def to_sql(self):
        '''将数据传到sql中'''
        pass
    # 截面数据的读取
    def rander_leaflet(self,start_time,end_time):
        '''返回数据'''
        if start_time==0:
            rain = end_rain 
        else:
            start_rain = self.lsp[start_time,:,:]
            end_rain = self.lsp[end_time,:,:]
            rain = end_rain - start_rain 
        filepath = "static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(rain, filepath+"taizhou.shp", "remain") 
        taizhou = shp_da.where(shp_da.remain<7, other=99999)
        len_lat = len(taizhou.lat.data)
        len_lon = len(taizhou.lon.data)
        data = []
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                #y0 = round(27.00+j*0.05,2)
                #x0 = round(120.00+i*0.05,2)
                y0 = taizhou.lat.data[j]
                x0 = taizhou.lon.data[i]
                if taizhou.data[j, i]!=99999:
                    # single = {
                    #     "Lat":y0,
                    #     "Lon":x0,
                    #     "value":str(taizhou.data[j, i])
                    # }
                    single = {
                        "type": "Feature",
                        "properties": {
                            "value": str(taizhou.data[j, i])
                        },
                        "geometry": {
                            "type": "Point",
                            "coordinates": [x0, y0]
                        }
                    }
                    data.append(single)  
        return data
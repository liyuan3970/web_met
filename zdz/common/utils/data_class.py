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
import pymysql
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
    def sql_data(self):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        start = self.start 
        end = self.end 
        #print(start,end)
        sql_location = "select b.StationName,b.county,b.Town,b.lat,b.lon,b.IIiii,b.Height,tTime,dFy,fFy,T,Tx,Tn,VV,RR from\
            TAB_Mws2019 as a left join TAB_StationInfo as b on a.IIiii=b.IIiii where\
            (b.IIiii in (select IIiii from TAB_StationInfo where(City = '台州') and tTime between"
        sql_location = sql_location + " " + "'" +  start + "'" +  "  " + "and" + " " + "'" +  end + "'" + "))"
        df_location = pd.read_sql(sql_location , con=conn)
        return df_location
    def return_daylist(self):
        '''
        返回日期列表
        '''
        dates = []
        #         dt = datetime.datetime.strptime(self.start, "%Y-%m-%d")
        dt = dtt.datetime.strptime(self.start[0:10], "%Y-%m-%d")
        date = self.start[:10]
        yesday = dt+dtt.timedelta(-1)
        dates.append(yesday.strftime("%Y-%m-%d")[:10])
        while date <= self.end[:10]:
            dates.append(date)
            dt = dt + dtt.timedelta(1)
            date = dt.strftime("%Y-%m-%d")
        return dates
    def day_button(self,city,station_data):
        '''返回按钮组'''
        station_all = station_data
        dates = self.return_daylist()
        dailylist = []
        for i in range(len(dates)):
            daily = {
                "time":None,
                "rain":False,
                "wind":False,
                "tmax":False,
                "tmin":False,
                "view":False,
            }
            if i!=0:
                s_date = dates[i-1] + ' ' + '20:00'
                e_date = dates[i] + ' ' + '20:00'
                daily["time"] = dates[i]
                df_preday = station_all[(station_all['tTime'] >= s_date) & (station_all['tTime'] <= e_date)]
                # 统计标量
                temp_min = df_preday[(df_preday['Height'] < 6000) & (df_preday['T'] > -400)]['T'].min()
                temp_max = df_preday[(df_preday['Height'] < 6000) & (df_preday['T'] > -400)]['T'].max()
                vv = df_preday[(df_preday['VV'] > 0.1) & (df_preday['VV'] < 500)]['VV'].count()
                RR = df_preday[(df_preday['RR'] > 0.1) & (df_preday['RR'] < 8888)]['RR'].count()
                wind = df_preday[df_preday['fFy'] > 187]['fFy'].count()    
                if temp_min<0:
                    daily["tmin"] = True
                if temp_max>350:
                    daily["tmax"] = True
                if vv >0:
                    daily["view"] = True
                if RR >0:
                    daily["rain"] = True
                if wind >0:
                    daily["wind"] = True
                dailylist.append(daily)
        return dailylist
    def rain_data(self,start,end,station_data):
        '''返回点击绘图所需的数据'''
        # 采集数据 可以是sql
        station_all = station_data
        grouped_tTime = station_all.groupby('IIiii')
        points = []
        table_data = []  
        for i in grouped_tTime.size().index:
            data = grouped_tTime.get_group(i)
            RR = data[(data['RR'] > 0.0) & (data['RR'] < 8888)]['RR'].sum()/10.0
            if RR > 0:
                single = {
                    "type": "Feature",
                    "properties": {
                        "value": str(RR)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [data['lon'].iloc[0], data['lat'].iloc[0]]
                    }
                }
                points.append(single)  
                # 采集marker和表格的数据
                rain = data.sort_values(by="tTime")["RR"].to_list()
                time = data.sort_values(by="tTime")["tTime"].to_list()
                ## 自己定义表格的数据形式
                single_rain = {
                    "IIiii":data['IIiii'].iloc[0],
                    "county":data['county'].iloc[0],
                    "town":data['Town'].iloc[0],
                    "StationName":data['StationName'].iloc[0],
                    "RR":RR,
                    "lat":data['lat'].iloc[0],
                    "lon":data['lon'].iloc[0],
                    "height":data['Height'].iloc[0],
                    "rain_list":rain,
                    "time_list":time
                }
                table_data.append(single_rain)  
        return points
    def index_data(self):
        '''返回所有数据'''
        city = "taizhou"
        #station_all = self.read_csv()
        station_all = self.sql_data()
        daily_btn_list = self.day_button(city,station_all)
        grouped_tTime = station_all.groupby('IIiii')
        table_data = []  
        points = []
        for i in grouped_tTime.size().index:
            data = grouped_tTime.get_group(i)
            time = data.sort_values(by="tTime")["tTime"].to_list()
            RR = data[(data['RR'] > 0.0) & (data['RR'] < 8888)]['RR'].sum()/10.0
            tmin = data[(data['T'] >-400) & (data['T'] < 8888)]['T'].min()/10.0
            tmax = data[(data['T'] >-400) & (data['T'] < 8888)]['T'].max()/10.0
            vv = data[(data['VV'] >0) & (data['VV'] < 1000)]['VV'].min()
            wind = data[data['fFy'] > 180]['fFy'].max()/10.0
            index =  data[data['fFy'] == data['fFy'].max()].index.tolist()[0]
            #print(index,"---",wind,"--",data['dFy'][index])
            #deg = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
            deg = data['dFy'][index]
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(RR)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [data['lon'].iloc[0], data['lat'].iloc[0]]
                }
            }        
            # 雨
            if np.isnan(RR):
                RR=-9999.0
                rain = False
            else:
                # rain_data = {
                #     "type": "Feature",
                #     "properties": {
                #         "value": str(RR)
                #     },
                #     "geometry": {
                #         "type": "Point",
                #         "coordinates": [data['lon'].iloc[0], data['lat'].iloc[0]]
                #     }
                # }
                # points.append(rain_data)
                rain = data.sort_values(by="tTime")["RR"].to_list()
            # 风
            if np.isnan(wind):
                wind=-9999.0
                fFy = False
                dFy = False
            else:
                fFy = data.sort_values(by="tTime")["fFy"].to_list()
                dFy = data.sort_values(by="tTime")["dFy"].to_list()
            # 温
            if np.isnan(tmax):
                tmax=-9999.0
                tx = False
            else:
                tx = data.sort_values(by="tTime")["Tx"].to_list()
            if np.isnan(tmin):
                tmin=-9999.0
                tn = False
            else:
                tn = data.sort_values(by="tTime")["Tn"].to_list()
            # 雾
            if np.isnan(vv):
                vv=-9999.0
                view = False
            else:
                view = data.sort_values(by="tTime")["VV"].to_list()
            single = {
                "IIiii":str(data['IIiii'].iloc[0]),
                "county":str(data['county'].iloc[0]),
                "town":str(data['Town'].iloc[0]),
                "StationName":str(data['StationName'].iloc[0]),
                "fFy":str(wind),
                "dFy":str(deg),
                "RR":str(RR),
                "Tx":str(tmax),
                "Tn":str(tmin),
                "VV":str(vv),
                "lat":str(data['lat'].iloc[0]),
                "lon":str(data['lon'].iloc[0]),
                "rain_list":rain,
                "tmax_list":tx,
                "tmin_list":tn,
                "fFy_list":fFy,
                "dFy_list":dFy,
                "view_list":view,
                "time_list":time
                }
            table_data.append(single)
        return table_data,points,daily_btn_list




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
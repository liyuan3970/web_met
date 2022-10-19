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

    def plot_img(self,item):
        '''绘制逐小时的气温'''
        lat = self.lat
        lon = self.lon
        time = self.time
        data_xr_nc = self.data_xr_nc
        data_xr = xr.DataArray(data_xr_nc[item,:,:],coords=[lat,lon], dims=["lat", "lon"])
        # ##########色标和大小#############################
        cmaps ,levels = self.colormap(self.plot_type)
        fig = plt.figure(figsize=[10,10]) 
        ax = fig.add_subplot(111)
        shp_path = "static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"taizhou.shp", "country")
        awash_da = shp_da.where(shp_da.country<7, other=np.nan)
        m = Basemap(llcrnrlon=120.0,
            llcrnrlat=27.8,
            urcrnrlon=122,
            urcrnrlat=29.5,
            resolution = None, 
            projection = 'cyl')
        lons, lats = np.meshgrid(lon, lat)
        cs =m.contourf(lons,lats,data_xr,ax=ax, cmap=cmaps,levels =levels,add_labels=True)
        ##########标题#############################
        label, start_time  = self.label_text(self.plot_type,item)
        plt.text(120.2,29.4, label,fontsize=15)
        ##########标题#############################
        m.readshapefile(shp_path+'taizhou','taizhou',color='k',linewidth=1.2)
        parallels = np.arange(27.8,29.5,0.2)
        m.drawparallels(parallels,labels=[True,False,True,False],color='dimgrey',dashes=[2, 3],fontsize= 12)  # ha= 'right'
        meridians = np.arange(120.0,122.0,0.2)
        m.drawmeridians(meridians,labels=[False,True,False,True],color='dimgrey',dashes=[2, 3],fontsize= 12)
        len_lat = len(data_xr.lat.data)
        len_lon = len(data_xr.lon.data)
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                y0 = round(27.50+j*0.05,2)
                x0 = round(119.80+i*0.05,2)
                if not isnan(awash_da.data[j,i]):
                    plt.text(x0,y0,str(int(awash_da.data[j,i])),fontsize= 7,fontweight = 800 ,color ="black")            
        # 在图上绘制色标
        rect1 = [0.35, 0.25, 0.03, 0.12]         
        ax2 = plt.axes(rect1,frameon='False' )
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
        return imd,str(start_time)[:16]

    def multy_plot(self):
        '''返回图片列表'''
        imd_list = []
        time_list = []
        for i in range(self.time_len):
            imd,time = self.plot_img(i)
            imd_list.append(imd)
            time_list.append(time)
        return imd_list,time_list
    def colormap(self,plot_type):
        '''色标的自定义'''
        # 可选用的绘图类型  
        ## 降水
#         colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 降水
#         cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
#         levels = [0,1,2,3,4,5,6,7]
        ## 云量
#         colorslist = ['#FFFFFF',"#F0F0F0","#E6E6E6","#D2D2D2","#BEBEBE","#AAAAAA","#969696","#828282","#6E6E6E","#5A5A5A"]# CLOUND
#         cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=10)
#         levels = [0,1,2,3,4,5,6,7,8,9,10]
        ## 气温
#         colorslist = ["#264FC7","#286BD9","#2B87EB","#2EA4FD","#48BBF0","#62D3E3","#7DEBD7","#9CEFC0","#BBF3A9","#DBF792","#E7E07C","#F3CB66","#FFB551","#FFBB6A","#FFC184","#FFC89E","#FFDABE","#FFECDE","#FFFFFF"]# CLOUND
#         cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=19)
#         levels = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]    
        if plot_type =="T":
            # 短期气温
            colorslist = ["#264FC7","#286BD9","#2B87EB","#2EA4FD","#48BBF0","#62D3E3","#7DEBD7","#9CEFC0","#BBF3A9","#DBF792","#E7E07C","#F3CB66","#FFB551","#FFBB6A","#FFC184","#FFC89E","#FFDABE","#FFECDE","#FFFFFF"]# CLOUND
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=19)
            levels = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] 
        elif plot_type =="Pr01":
            # 短期降水
            colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 降水
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
            levels = [0,1,2,3,4,5,6,7]
        elif plot_type =="Pr12":
            # 降水
            colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 降水
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
            levels = [0,1,2,3,4,5,6,7]
        elif plot_type =="TMax24":
            # 高温
            colorslist = ["#264FC7","#286BD9","#2B87EB","#2EA4FD","#48BBF0","#62D3E3","#7DEBD7","#9CEFC0","#BBF3A9","#DBF792","#E7E07C","#F3CB66","#FFB551","#FFBB6A","#FFC184","#FFC89E","#FFDABE","#FFECDE","#FFFFFF"]# CLOUND
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=19)
            levels = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] 
        elif plot_type =="TMin24":
            # 低温
            colorslist = ["#264FC7","#286BD9","#2B87EB","#2EA4FD","#48BBF0","#62D3E3","#7DEBD7","#9CEFC0","#BBF3A9","#DBF792","#E7E07C","#F3CB66","#FFB551","#FFBB6A","#FFC184","#FFC89E","#FFDABE","#FFECDE","#FFFFFF"]# CLOUND
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=19)
            levels = [6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40] 
        elif plot_type =="Cloud":
            colorslist = ['#FFFFFF',"#F0F0F0","#E6E6E6","#D2D2D2","#BEBEBE","#AAAAAA","#969696","#828282","#6E6E6E","#5A5A5A"]# CLOUND
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=10)
            levels = [0,1,2,3,4,5,6,7,8,9,10]
        elif plot_type =="Special12":
            colorslist = ['#FFFFFF',"#F0F0F0","#E6E6E6","#D2D2D2","#BEBEBE","#AAAAAA","#969696","#828282","#6E6E6E","#5A5A5A"]# CLOUND
            cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=10)
            levels = [0,1,2,3,4,5,6,7,8,9,10]
        return cmaps,levels
    def label_text(self,plot_type,item):
        '''图题'''
        if plot_type in ["T","Pr01","Cloud"]:
            # 逐小时
            print("查看数据",item)
            start_year = int(self.date[0:4])
            start_month = int(self.date[4:6])
            start_day = int(self.date[6:8])   
            init_time = datetime(start_year, start_month, start_day, int(self.plot_time), 0, 0)
            start_hours = int(self.time[item]-1)
            start_time = init_time + timedelta(hours = start_hours)
            end_time =  start_time + timedelta(hours = 1)
            label = str(start_time)[:16] + "---" + str(end_time)[10:16]
        else:
            # 每天
            start_year = int(self.date[0:4])
            start_month = int(self.date[4:6])
            start_day = int(self.date[6:8])   
            init_time = datetime(start_year, start_month, start_day, int(self.plot_time), 0, 0)
            start_hours = int(self.time[item]-1)
            start_time = init_time + timedelta(hours = start_hours)
            end_time =  start_time + timedelta(hours = 24)
            label = str(start_time)[:16] + "---" + str(end_time)[10:16]
        return label, start_time    

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
        self.event_data()
        self.rain_data()
        self.img = self.plot_img()
    def plot_img(self):
        x = [1,2,3,4,5,6]
        y = [1,2,3,4,5,6]
        plt.plot(y)
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight')  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        return imd
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
        
        grouped_tTime = station_all.groupby('tTime')
        for i in grouped_tTime.size().index:
            data = grouped_tTime.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            rain_mean = data['RR'].mean() / 10.0
            rain_time = i
            data_rain['rain_sum']['time'].append(rain_time)
            data_rain['rain_sum']['data'].append(rain_mean)
        # 导出单站数据
        grouped_IIiii = station_all.groupby('IIiii')
        rain_scatter = []
        #         {
        #          name: "K8515", value: [121.2, 28.6, 110],
        #          symbol: 'circle'
        #         }
        for i in grouped_IIiii.size().index:
            data = grouped_IIiii.get_group(i)
            data['RR'].replace(-9999, np.nan, inplace=True)
            station_name = str(i)
            data_rain['IIiii_data'][station_name] = data
            single_data = {}
            single_data['name'] = station_name
            single_data['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['RR'].sum() / 10.0]
            single_data['symble'] = 'circle'
            rain_scatter.append(single_data)
        self.station_data = data_rain['IIiii_data']
        self.rain_line = [data_rain['rain_sum']['time'], data_rain['rain_sum']['data']]
        self.rain_scatter = rain_scatter
        #self.plot_img()

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

        grouped_IIiii = data_time.groupby('IIiii')
        # 所需数据库
        pre_list = []
        wind_list = []
        view_list = []
        tmax_list = []
        tmin_list = []

        for i in grouped_IIiii.size().index:
            data = grouped_IIiii.get_group(i)
            # print(data)   
            # 低温
            if data[data['T'] > -999]['T'].min() < 30:
                tmin_dir = {}
                tmin_dir['name'] = data['IIiii'].iloc[0]
                tmin_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['T'].tolist(), data['T'].min()]
                tmin_list.append(tmin_dir)
            # 高温   
            if data['T'].max() > 350:
                tmax_dir = {}
                tmax_dir['name'] = data['IIiii'].iloc[0]
                tmax_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['T'].tolist(), data['T'].max()]
                tmax_list.append(tmax_dir)
            # 大风
            if data[data['fFy'] > 187]['fFy'].max():
                wind_dir = {}
                wind_dir['name'] = data['IIiii'].iloc[0]
                wind_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['fFy'].tolist(), data['fFy'].max()]
                wind_dir['symbol'] = 'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'
                wind_dir['symbolRotate'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
                wind_list.append(wind_dir)
            # 能见度
            if data[(data['VV'] < 500) & (data['VV'] > 0)]['VV'].min():
                view_dir = {}
                view_dir['name'] = data['IIiii'].iloc[0]
                view_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['VV'].tolist(), data['VV'].min()]
                view_list.append(view_dir)
                # 降水
            if data['RR'].max() > 0:
                pre_dir = {}
                data['RR'].replace(-9999, np.nan, inplace=True)
                pre_dir['name'] = data['IIiii'].iloc[0]
                pre_dir['value'] = [data['lon'].iloc[0], data['lat'].iloc[0], data['tTime'].tolist(),
                                    data['RR'].tolist(), data['RR'].sum()]
                pre_dir['symble'] = "circle"
                pre_list.append(pre_dir)
        return pre_list

    def text_data(self):
        '''用来处理风雨情统计数据'''
        print('用来处理风雨情统计数据')




# ec数据的处理和对接
class ec_data_point:
    def __init__(self, select_time,select_type,select_lat,select_lon): 
        self.var_list = ['u10','v10','tcc','t2','skt','lsp','cp','tp','r'] # 风向风速、云量、温度、tp（总降水）、相对湿度
        self.data = { }
        self.read_data()
    # 创建对象读取io的核心代码####################################################################
    def regrid_data(self,data):
        ds_out = xr.Dataset(
            {   
                
                "lat": (["lat"], np.arange(27.8, 29.5, 0.05)),
                "lon": (["lon"], np.arange(120, 122, 0.05)),
            }
        )
        regridder = xe.Regridder(data, ds_out, "bilinear")
        dr_out = regridder(data)
        return dr_out
    def read_data(self):
        file_path = "/home/liyuan3970/Data/My_Git/2022041700/*.nc" 
        f = xr.open_mfdataset(file_path, parallel=False)
        # 列表数据--
        u10 = f.u10.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        u10 =  u10.swap_dims({'latS':'lat','lonS':'lon'})
        v10 = f.v10.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        v10 =  v10.swap_dims({'latS':'lat','lonS':'lon'})
        tcc = f.tcc.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        tcc =  tcc.swap_dims({'latS':'lat','lonS':'lon'})
        t2 = f.t2.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        t2 =  t2.swap_dims({'latS':'lat','lonS':'lon'})
        skt = f.skt.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        skt =  skt.swap_dims({'latS':'lat','lonS':'lon'})
        lsp = f.lsp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8))
        lsp =  lsp.swap_dims({'latS':'lat','lonS':'lon'})
        cp = f.cp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8)) 
        cp =  cp.swap_dims({'latS':'lat','lonS':'lon'})    
        tp = f.tp.sel(lev=1000,lonS=slice(120,122),latS=slice(29.5,27.8)) 
        tp =  tp.swap_dims({'latS':'lat','lonS':'lon'}) 
        r = f.r.sel(lev=1000,lonP=slice(120,122),latP=slice(29.5,27.8)) # ！！！如果经纬度放大就会报错。。。
        r =  r.swap_dims({'latP':'lat','lonP':'lon'})
        r['lon'] = r['lonP']
        r['lat'] = r['latP']  
        # print(tp,r)
        # 空间插值----------------------------------
        # ----------------------------------------
        # ----------------------------------------
        # 此处为空间插值函数0.05
        grid_u10 = self.regrid_data(u10)
        grid_v10 = self.regrid_data(v10)
        grid_tcc = self.regrid_data(tcc)
        grid_t2 = self.regrid_data(t2)
        grid_skt = self.regrid_data(skt)
        grid_lsp = self.regrid_data(lsp)
        grid_cp = self.regrid_data(cp)
        grid_tp = self.regrid_data(tp)
        grid_r = r#self.regrid_data(r)
        # 数据的重新加载
        
        self.data['u10'] = grid_u10
        self.data['v10'] = grid_v10
        self.data['tcc'] = grid_tcc
        self.data['t2'] = grid_t2
        self.data['skt'] = grid_skt
        self.data['lsp'] = grid_lsp
        self.data['cp'] = grid_cp    
        self.data['tp'] = grid_tp 
        self.data['r'] = grid_r 
        # print(grid_vis.sel(lon=121.5, lat=28.5,method='nearest'))    
    # 以下为处理html表格的核心代码######################################################################################################################
    #  1. 计算超出指定日期的时间段数
    def time_point_len(self,times,step):
        start = str(times[0:10] + ' 00:00:00')
        t_start =dtt.datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end = str(times+':00')
        t_end =dtt.datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        timedelta = (t_end - t_start).seconds
        hours = timedelta/3600
        if step == 'hours':
            if hours <= 6:
                step_len = 0
            elif hours >6 and hours <= 12:
                step_len = 1
            elif hours >12 and hours <= 18:
                step_len = 2
            elif hours >18 and hours <= 24:
                step_len = 3
            return step_len
        elif step == '3hours':
            if hours <= 3:
                step_len = 0
            elif hours >3 and hours <= 6:
                step_len = 1
            elif hours >6 and hours <= 9:
                step_len = 2
            elif hours >9 and hours <= 12:
                step_len = 3
            elif hours >12 and hours <= 15:
                step_len = 4
            elif hours >15 and hours <= 18:
                step_len = 5
            elif hours >18 and hours <= 21:
                step_len = 6
            elif hours >21 and hours <= 24:
                step_len = 7   
            return step_len
        else:
            if hours <= 6:
                step_len = 0
            elif hours >6 and hours <= 12:
                step_len = 1
            elif hours >12 and hours <= 18:
                step_len = 2
            elif hours >18 and hours <= 24:
                step_len = 3
            return step_len  
    def return_dates_step(self):
        '''用于返回dates 和start_len,end_len'''
        # 起始时间
        start_day = r'2022-04-18 00:00'
        end_day = r'2022-04-22 13:00'
        # EC数据
        single_point_data = {
            'wind_speed':[i for i in range(241)],
            'wind_dir':[i for i in range(241)],
            'tcc':[i for i in range(241)],
            'skt':[i for i in range(241)],
            't2':[i for i in range(241)],
            'lsp':[i for i in range(241)],
            'cp':[i for i in range(241)],
            'tp':[i for i in range(241)],
            'r':[i for i in range(241)]
            
        }
        step = 'hours'
        time_data = pd.date_range(start='2022-04-17 00:00:00',end='2022-04-27 00:00:00',freq='1H')
        data_index = [i for i in range(241) ]
        ts = pd.Series(data_index, time_data)
        start_len = self.time_point_len(start_day,step)
        end_len = self.time_point_len(end_day,step)
        # 计算日期
        dates = []
        dt = dtt.datetime.strptime(start_day[0:10], "%Y-%m-%d")
        date = start_day[0:10] + ' 00:00'
        while date <= end_day:
            dates.append(date)
            dt = dt + dtt.timedelta(1)
            date = dt.strftime("%Y-%m-%d") + ' 00:00'
        return dates,start_len,end_len,ts  
    def return_timestep(self,dates,step,start_len,end_len,ts,single_point_data):
        step_data = []
        if step == 'hours':
            # 6是每段时间的个数
            day_step = ['凌晨','上午','下午','晚上']
            step_data_num = 6
            step_list = [
                ['00:00~01:00','01:00~02:00','02:00~03:00','03:00~04:00','04:00~05:00','05:00~06:00'],
                ['06:00~07:00','07:00~08:00','08:00~09:00','09:00~10:00','10:00~11:00','11:00~12:00'],
                ['12:00~13:00','13:00~14:00','14:00~15:00','15:00~16:00','16:00~17:00','17:00~18:00'],
                ['18:00~19:00','19:00~20:00','20:00~21:00','21:00~22:00','22:00~23:00','23:00~24:00']
            ]
        elif step == '3hours':
            step_data_num = 1
            step_list = [
                ['00:00~03:00'],['03:00~06:00'],['06:00~09:00'],['09:00~12:00'],['12:00~15:00'],['15:00~18:00'],['18:00~21:00'],['21:00~24:00']   
            ]
        else:
            step_data_num = 1
            step_list = [
                ['凌晨'],['上午'],['下午'],['晚上']   
            ]
        # day_list
        if step != 'hours':
            day_list = [(i[5:7] + '月' + i[8:10] + '日') for i in dates]
        else:
            day_list = []
        for i in range(len(dates)):
            start_date , start_index = dates[i],ts[dates[i]]
            if i == 0: 
                # 第一天的数据         
                if step == 'hours':
                    # 计算天的段数
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    start_day_list = [day_str + i for i in day_step ]
                    day_list.extend(start_day_list[start_len:]) 
                    len_max = 4
                    for start_num in range(len_max-start_len):
                        single_step_data = {}   
                        single_step_data['step_list'] = step_list[(start_num+start_len)]
                        #  6是每段时间的个数
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                elif step == '3hours':
                    len_max = 8
                    for start_num in range(len_max-start_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[(start_num+start_len)]   
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*3  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                else:
                    len_max = 4  
                    for start_num in range(len_max-start_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[(start_num+start_len)]   
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (start_num+start_len)*6  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (start_num+start_len)*6  + step_num ])    
                            appen_data.append(single_point_data['r'][start_index + (start_num+start_len)*6  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)      
            elif i ==len(dates)-1:    
                # 最后一天的数据
                if step == 'hours':
                    # 最后一天的day
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    end_day_list = [day_str + i for i in day_step ]
                    day_list.extend(end_day_list[:end_len]) 
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
                elif step == '3hours':
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*3  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*3  + step_num])
                            single_step_data['step_data'].append(appen_data)                                    
                        step_data.append(single_step_data)
                else:
                    for end_num in range(end_len):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[end_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (end_num)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (end_num)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
            else:
                # 中间天
                if step == 'hours':
                    # 添加 数据
                    day_str = dates[i][5:7] + '月' + dates[i][8:10] + '日'
                    middle_day_list = [day_str + i for i in day_step ]
                    day_list.extend(middle_day_list[:]) 
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)         
                        step_data.append(single_step_data)
                elif step == '3hours':
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num):   
                            #print(step_num,start_index + (start_num+start_len)*6  + step_num,start_date , start_index)
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*3  + step_num ])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*3  + step_num ])
                            single_step_data['step_data'].append(appen_data)
                        step_data.append(single_step_data)
                else:
                    for middle_num in range(len_max):
                        single_step_data = {}
                        single_step_data['step_list'] = step_list[middle_num]
                        single_step_data['step_data'] = []
                        for step_num in range(step_data_num): 
                            appen_data = []
                            appen_data.append(single_point_data['tcc'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['t2'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['wind_speed'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['tp'][start_index + (middle_num+start_len)*6  + step_num])
                            appen_data.append(single_point_data['r'][start_index + (middle_num+start_len)*6  + step_num])
                            single_step_data['step_data'].append(appen_data)        
                        step_data.append(single_step_data)
        data = {
            'day':day_list,
            'time_step':step_data
        }
        return data   
    def decode_html_table(self):
        # 用来解析数据并返回表格的html数据
        # 日期、间隔
        step = 'prehours'# 'sixhours';'thrdhours'
        data = {
            'day':['29日','30日'],
            'time_step':[
                {
                'step_list':['08:00~09:00','08:00~09:00','08:00~09:00'],
                'step_data':[
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%'], 
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%'], 
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%']
                    ]
                },
                {
                'step_list':['08:00~09:00','08:00~09:00','08:00~09:00'],
                'step_data':[
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%'], 
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%'], 
                    ['多云转阴','25~27°C','东南风6~8级','15mm','80%']
                    ]
                }
            ],
            'blank':[
                ['','08:00~09:00','多云转阴','25~27°C','东南风6~8级','15mm','80%']
            ]
        }
        len_day = len(data['day'])
        len_setp = 3#len(data['time_step']['step_list'])
        html_table = ""
        for i in data['day']: 
            day_index = 0
            if i == 'blank':
                k = 0
                html_table = " <tr ><td>" + data['blank'][k][0] + "</td><td>" +  \
                    "<td>" + data['blank'][k][1] + "</td>" + \
                    "<td>" + data['blank'][k][2] + "</td>" + \
                    "<td>" + data['blank'][k][3] + "</td>" + \
                    "<td>" + data['blank'][k][4] + "</td>" + \
                    "<td>" + data['blank'][k][5] + "</td>" + \
                    "</tr>"
                html_table = html_table + table
                k = k+1
            else:
                for j in range(len_setp):
                    if j==0:
                        table =" <tr ><td colspan='1' rowspan= "+"'" + str(len_setp) +  "'" + ">" + str(i) + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_list'][j] +  "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][0] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][1] + "</td>" + \
                        "<td>" + data['time_step'][day_index]['step_data'][j][2] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][3] + "</td>" +\
                        "<td>" + data['time_step'][day_index]['step_data'][j][4] + "</td></tr>"                        
                        html_table = html_table + table
                    else:
                        table = "<tr >"+"<td>"+ data['time_step'][day_index]['step_list'][j] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][0] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][1] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][2] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][3] + "</td>"+\
                        "<td>" + data['time_step'][day_index]['step_data'][j][4] + "</td>" + "</tr>"
                
                        html_table = html_table + table
                day_index = day_index + 1
        return html_table
    # 实例创建后和用户交互的核心代码###############################################################################################
    def interp1d_data(self,data):
        ''' 将get_data和get_single_data的数据进行时间插值'''
        date = data.index.tolist()
        value = data.tolist()
        d = pd.DataFrame()
        d['date'] = pd.to_datetime(date)
        d['val'] = value
        helper = pd.DataFrame({'date': pd.date_range(d['date'].min(), d['date'].max(), freq='H')})
        d = pd.merge(helper, d, on='date', how='left').sort_values('date')
        d['val'] = d['val'].interpolate(method='linear')        
        return d['val'].tolist() 
    def accum_data(self,data):
        date = data.index.tolist()
        value = data.tolist()
        time_data = pd.date_range(start='2022-04-17 00:00:00',end='2022-04-27 00:00:00',freq='1H')
        data_index =[i for i in range(241) ]
        ts = pd.Series(data_index, time_data)
        # 赋值        
        ts[0:73:3] = value[0:25]
        ts[78:241:6] = value[25:53]
        ts_list = []
        for i in range(len(ts)):
            if i >= 0 and i<3:
                ts_list.append(ts[3]/3) 
            elif i < 72 and i>=3:
                index = i % 3  # 余数
                mod = i// 3 # 商
                ts_list.append(ts[(mod+1)*3]/3 - ts[(mod)*3]/3  )
            elif i >=72 and i < 78:
                ts_list.append((ts[78] - ts[72])/6) 
            elif i>=78 and i <=234:
                index = i % 6  # 余数
                mod = i// 6 # 商
                ts_list.append(ts[(mod+1)*6]/6 - ts[(mod)*6]/6  )
            if i>=235:
                ts_list.append((ts[240] - ts[234])/6) 
        return ts_list 
    def get_single(self,select_lat,select_lon):
        '''用于处理特定经纬度数据'''
        # single_point_data = {
        #     'vis':vis
        # }
        single_point_data = {}
        single_point_inter1d_data = {}
        for var in self.var_list:
            select_data = self.data[var]
            single_point_data[var] = select_data.sel(lon=select_lon, lat=select_lat,method='nearest').to_pandas()
            if var in ['u10','v10','tcc','t2','skt','r']:
                single_point_inter1d_data[var] = self.interp1d_data(single_point_data[var])
            else:
                single_point_inter1d_data[var] = self.accum_data(single_point_data[var])
        return single_point_inter1d_data 
    # 处理所有数据的核心data #############################################################################################################
    def comput_all_data(self):
        '''用于计算所有数据的核心代码
        1.依据step 返回single_data
        2.依据step 返回曲线图的data 4个list 一个字典list
        '''
        # 计算单点数据
        # single_point_data = self.get_single(27.5,125.7)
        single_point_data = {
            'wind_speed':[i for i in range(241)],
            'wind_dir':[i for i in range(241)],
            'tcc':[i for i in range(241)],
            'skt':[i for i in range(241)],
            't2':[i for i in range(241)],
            'lsp':[i for i in range(241)],
            'cp':[i for i in range(241)],
            'tp':[i for i in range(241)],
            'r':[i for i in range(241)]
            
        }

        dates,start_len,end_len,ts  = self.return_dates_step()
 
        step = 'hours'
        self.return_timestep(dates,step,start_len,end_len,ts,single_point_data)
        self.decode_html_table()
        
        # 计算datalist
        wind_list = single_point_data['r']
        r_list = single_point_data['r']
        temp_list = single_point_data['r']
        pre_list = single_point_data['r']
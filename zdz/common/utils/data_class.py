import gevent
from math import isnan
import numpy as np
#import modin.pandas as pd 
import pandas as pd
import time
from . import func
import netCDF4 
# import h5netcdf.legacyapi as netCDF4
import h5py

from mpl_toolkits.basemap import Basemap
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapefile
import matplotlib as mpl
import xarray as xr
from matplotlib.font_manager import FontProperties
import netCDF4 as nc
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib
import geopandas as gpd
from ncmaps import Cmaps
from io import BytesIO
import base64
import json
import math
from scipy.interpolate import griddata
from math import ceil, floor
from rasterio import features
from affine import Affine
import os
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
import datetime






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
            'IIiii':[],
            'lat':[],
            'lon':[],
            'county':[],
            'town':[],
            'name':[],
            'fFy':[],
            'dFy':[],
            'rsum':[],
            'tmax':[],
            'tmin':[],
            'rmax':[]
            }
    def comput_county(self):
        '计算面最大雨强、累计降水、最高、最低气温'
        self.station_county_comput = []
        for i in self.grouped_county.size().index:  
            data= self.grouped_county.get_group(i)
            data['VV'].replace(-9999,np.nan,inplace=True)
            data['RR'].replace(-9999,np.nan,inplace=True)
            data['Tn'].replace(-9999,np.nan,inplace=True)
            data['Tx'].replace(-9999,np.nan,inplace=True)
            dic = {}
            dic['county'] = str(i)
            dic['RR'] = data['RR'].mean()*self.timecounts/10.0
            dic['RMax'] = data['RR'].max()/10.0
            dic['Tx'] = data['Tx'].max()/10.0
            dic['Tn'] = data['Tn'].min()/10.0 
#             print(dic)
            self.station_county_comput.append(dic)
        tmp_max_County = []
        tmp_min_County = []
        RR_County = []
        for i in self.station_county_comput:
            tmp_max_County.append({"name":i['county'],"value":i['Tx']})
            tmp_min_County.append({"name":i['county'],"value":i['Tn']})
            RR_County.append({"name":i['county'],"value":i['RR']})
        return RR_County,tmp_max_County,tmp_min_County
    def data_gevent(self,data):
        # print("data_gevent")

        nan_del = lambda x: 9999 if np.isnan(x) else x 
        value_rsum = data['RR'].sum()/10.0
        value_rmax = data['RR'].sum()/10.0
        fFy_data = data['fFy'].max()/10.0
        dFy_data = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
        value_VV = nan_del(data['VV'].min())
        value_tmax = nan_del(data['Tx'].max()/10.0)
        value_tmin = nan_del(data['Tn'].min()/10.0)
        # print("查看数据：",print(type(value_VV)),value_VV)
        if value_rsum >= 0 and value_rsum < 10:
            self.station_RR_small = self.station_RR_small + 0
        elif value_rsum >= 9 and value_rsum < 25:
            self.station_RR_mid = self.station_RR_mid+0
        elif value_rsum >= 24 and value_rsum < 50:
            self.station_RR_big = self.station_RR_big + 0
        elif value_rsum >= 49 and value_rsum < 100:
            self.station_RR_huge = self.station_RR_huge+0
        elif value_rsum >= 99 and value_rsum < 250:
            self.station_RR_RR_bighuge = self.station_RR_bighuge+0
        else:
            self.station_RR_more = self.station_RR_more+0
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
        if value_VV< self.vv_min_value:
            self.vv_min_value = value_VV
            self.vv_min_name = str(data['IIiii'].iloc[0])
        if value_VV >= 0 and value_VV < 50:
            self.station_VV_small = self.station_VV_small + 0
        elif value_VV >= 49 and value_VV < 200:
            self.station_VV_mid = self.station_VV_mid+0
        elif value_VV >= 199 and value_VV < 500:
            self.station_VV_big = self.station_VV_big + 0
        elif value_VV >= 499 and value_VV < 1000:
            self.station_VV_huge = self.station_VV_huge+0
        else:
            self.station_VV_more = self.station_VV_more+0
        # 绘图、排序数据
        self.plot_data['IIiii'].append(data['IIiii'].iloc[0])  
        self.plot_data['lat'].append(data['lat'].iloc[0]) 
        self.plot_data['lon'].append(data['lon'].iloc[0])
        self.plot_data['county'].append(data['county'].iloc[0])
        self.plot_data['town'].append(data['Town'].iloc[0])
        self.plot_data['name'].append(data['StationName'].iloc[0])
        self.plot_data['fFy'].append(fFy_data)
        self.plot_data['dFy'].append(dFy_data)
        self.plot_data['rsum'].append(value_rsum/10.0)
        self.plot_data['rmax'].append(data['RR'].max()/10.0)
        self.plot_data['tmax'].append(value_tmax)
        self.plot_data['tmin'].append(value_tmin)


       
        self.dic_station = {'IIiii': data['IIiii'].iloc[0],
                   'StationName': data['StationName'].iloc[0],
                   'County': data['county'].iloc[0],
                   'Town': data['Town'].iloc[0],
                   'lat': data['lat'].iloc[0],
                   'lon': data['lon'].iloc[0],
                   'rsum': value_rsum/10.0,
                   'rmax': data['RR'].max()/10.0,
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
                   'url_r': "station/"+str(data['IIiii'].iloc[0])+"/rain/",
                   'url_t': "station/"+str(data['IIiii'].iloc[0])+"/temp/",
                   'url_v': "station/"+str(data['IIiii'].iloc[0])+"/vv/",
                   'url_fy': "station/"+str(data['IIiii'].iloc[0])+"/fFy/",
                   'name': data['StationName'].iloc[0],

                   'value':[
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
            'name':data['StationName'].iloc[0],
            'rmax': data['RR'].max()/10.0,
            'rsum':value_rsum/10.0,
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
    def return_data_sort(self,sort_data,value_str):
        max_sort = max(sort_data[value_str])
        level_sort = np.linspace(start = 0.0, stop = max_sort, num = 9)
        sort_da = pd.DataFrame(sort_data)
        if value_str == "rsum" or 'rmax':
            sort_da['index'] = sort_da[value_str].rank(ascending=0,method='dense')
            sort_out = sort_da.sort_values(by =[value_str],ascending = [False])  
            list_data = []
            for row in sort_out.itertuples():
                dic_iter = {'index':int(getattr(row, 'index')),'IIiii':str(getattr(row, 'name')),
                    'county':getattr(row, 'county'),'town':getattr(row, 'town'),'data':getattr(row, value_str),
                    'value':[getattr(row, 'lon'),getattr(row, 'lat'),getattr(row,value_str)]}
                list_data.append(dic_iter)
        if value_str == "fFy":
            sort_da['index'] = sort_da[value_str].rank(ascending=0,method='dense')
            sort_out = sort_da.sort_values(by =[value_str],ascending = [False])  
            list_data = []
            for row in sort_out.itertuples():
                dic_iter = {'index':int(getattr(row, 'index')),'IIiii':str(getattr(row, 'name')),
                    'county':getattr(row, 'county'),'town':getattr(row, 'town'),'data':getattr(row, value_str),
                    'value':[getattr(row, 'lon'),getattr(row, 'lat'),getattr(row,value_str)],
                    'symbol':str(self.symbol_ffy[0]),
                    'symbolRotate':getattr(row, 'dFy')
                    }
                list_data.append(dic_iter)
        return list_data ,level_sort
    # 数据绘图

    def plot_imd(self, plot_data, value_str):
        # value_str = 'rsum'
        lat = plot_data['lat']
        lon = plot_data['lon']
        value = plot_data[value_str]
        func.plot_image(lat,lon,value)
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight')  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        return imd
    # 返回第一个页面的回调数据
    def data_output(self):
        # 排序
        RR_sum ,level_rain= self.return_data_sort(self.plot_data,'rsum')
        RR_rx,level_rmax = self.return_data_sort(self.plot_data,'rmax')
        data_fFy_list,level_fFy = self.return_data_sort(self.plot_data,'fFy')
        # 柱状图
        RR_station_rank = self.RR_rank
        nation_station = ['58660','58666','K8505','K8206','58665','58559','58655','K8271','58662','58653']
        tmp_station_bar = []
        tmp_station_bar.append(['product', '最高气温','最低气温'])
        RR_station_bar = []
        RR_station_bar.append(['product', '累计降水','最大雨强'])
        # 计算指标站nation_station的要素值 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for i in nation_station:
            # print(i)
            var_name = str(self.station_list[i]['name'])
            var_tmax = max(self.station_list[i]['T'])/10.0
            var_tmin = min(self.station_list[i]['T'])/10.0
            var_rsum = self.station_list[i]['rsum']
            var_rmax = self.station_list[i]['rmax']
            tmp_station_bar.append( [var_name,var_tmax,var_tmin])
            RR_station_bar.append( [var_name,var_rsum,var_rmax])
        # 散点图
        tmp_min_scatter = self.data_station
        tmp_max_scatter = self.data_station
        tmp_event_scatter = self.data_station
        VV_min_scatter = self.data_station
        fFy_wind7up_scatter = data_fFy_list
        # print(fFy_wind7up_scatter)
        # fFy_wind7up_scatter = self.data_station
        # 要素极值站点序列数据
        vv_time =  self.station_list[self.vv_min_name]['time']
        vv_value = self.station_list[self.vv_min_name]['V']
        data_vvmin =  pd.DataFrame()
        data_vvmin['tTime']= vv_time
        data_vvmin['VV']= vv_value
        data_vvmin.sort_values(by = 'tTime')
        tz_json =  self.read_shp_json()
        imd = self.plot_imd(self.plot_data,'rsum')
        imd_tmax = self.plot_imd(self.plot_data,'tmax')
        imd_tmin = self.plot_imd(self.plot_data,'tmin')
        



        return imd,imd_tmax,imd_tmin,tz_json,RR_sum ,RR_rx,level_rain,RR_station_rank,RR_station_bar,tmp_min_scatter,tmp_max_scatter,tmp_event_scatter,tmp_station_bar,VV_min_scatter,fFy_wind7up_scatter,vv_time,vv_value,data_fFy_list



# 画图的核心类
class plot_tz_product:
    def __init__(self, plot_type,plot_time):
        self.plot_time = plot_time
        self.plot_type = plot_type
        self.time_len = 0
        self.lat,self.lon,self.time,self.data_xr_nc  = self.read_data()
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
    def read_data(self):
        os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
        file_path = "static/data/TZ_self/"        
        # file_name = file_path +"20220401/"+'I20220401080000.'+self.plot_type+'.nc'
        file_name = file_path +"20220402/"+'I20220402080000.'+self.plot_type+'.nc'
        # f = xr.open_dataset(file_name)
        f = netCDF4.Dataset(file_name,"r",format="NETCDF4")
        data_xr_nc = f.variables[str(self.plot_type)]
        lat = f.variables['lat'][:]
        lon = f.variables['lon'][:]
        time = f.variables['time'][:]
        self.time_len = len(time)        
        return lat,lon,time,data_xr_nc
    def plot_img(self,item):
        lat = self.lat
        lon = self.lon
        time = self.time
        data_xr_nc = self.data_xr_nc
        data_xr = xr.DataArray(data_xr_nc[item,:,:],coords=[lat,lon], dims=["lat", "lon"])
        levels = np.linspace(start = 15, stop = 25, num = 5)#[10,20,30,40,50,60,70,80,90,100,110]
        self_define_list = [130,144,155,170,185,200,225,235,240,244]
        rgb_file = 'ncl_default'
        #以下是核心api,实质为调用Cmaps基类的listmap()方法
        cmaps = Cmaps('ncl_default',self_define_list).listmap()
        # plt.rcParams.update({'font.size': 20})
        fig = plt.figure(figsize=[10,10]) 
        ax = fig.add_subplot(111)
        shp_path = "static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(data_xr, shp_path+"taizhou.shp", "test")
        awash_da = shp_da.where(shp_da.test<7, other=np.nan)
        m = Basemap(llcrnrlon=120.0,
            llcrnrlat=27.8,
            urcrnrlon=122,
            urcrnrlat=29.5,
            resolution = None, 
            projection = 'cyl')
        # 设置colorbar
        cbar_kwargs = {
        #'orientation': 'horizontal',
        # 'label': 'Potential',
        'shrink': 0.5,
        }
        lons, lats = np.meshgrid(lon, lat)
        cs =m.contourf(lons,lats,data_xr,ax=ax, cmap='Spectral_r',levels =levels,cbar_kwargs=cbar_kwargs,add_labels=False)
        # m.colorbar(cs)
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
        rect1 = [0.75, 0.20, 0.03, 0.12]         
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
        return imd        
    def multy_plot(self):
        '''返回图片列表'''
        imd_list = []
        for i in range(self.time_len):
            imd = self.plot_img(i)
            imd_list.append(imd)
        return imd_list

# 自动站数据查询的class

class zdz_data:
    def __init__(self, start,end):
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
        #station_all = pd.read_csv("data_zdz_height.csv")
        station_all = pd.read_csv("static/data/data_zdz_height.csv")
        return station_all
    def event_data(self):
        '''
        1.数据库查询
        2.数据分段---每天
        3.每天统计
        ['2022-04-14', 200,'高温 浓雾', '降水 大风', 200],
        '''
        dateList = [ ] 
        # 多少天
        dates = self.return_daylist()
        for i in dates:
            #解析每天的数据
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
            temp_min = df_preday[(df_preday['Height']<6000)&(df_preday['T'] > -400)]['T'].min()
            temp_min = temp_min/10.0
            temp_max = df_preday[(df_preday['Height']<6000)&(df_preday['T'] > -400)]['T'].max()
            temp_max = temp_max/10.0
            temp_label = ''
            if temp_min < 3.0:
                temp_label = '低温'
            elif temp_max > 35.0:
                temp_label = '高温'
            else:
                temp_label = ' '
            # 能见度
            # df_VV= df_preday[(df_preday['VV']>0.1)&(df_preday['VV']<500)]['VV'].count()
            vv = df_preday[(df_preday['VV']>0.1)&(df_preday['VV']<500)]['VV'].count()
            if vv>0:
                view_label = '浓雾'
            else:
                view_label = ' '    
            # 降水
            df_RR= df_preday[(df_preday['RR']>0.3)&(df_preday['RR']<8888)]['RR'].count()
            
            if df_RR > 0:
                pre_label = '降水'
            else:
                pre_label = ' '
            # 大风
            wind_count = df_preday[df_preday['fFy']>187]['fFy'].count()
            if wind_count>0:
                wind_label = '大风'
            else:
                wind_label = ' '
            date_preday = [day_date,rainfall,temp_label + ' '+ view_label, pre_label +' '+ wind_label ,rainfall]
            dateList.append(date_preday)
        self.day_list = dateList
#         print('event_data:返回一个每天的数组和日历所需的数据','气温',dateList)
    def return_daylist(self):
        '''
        返回每天的起始列表
        '''
        dates = []
#         dt = datetime.datetime.strptime(self.start, "%Y-%m-%d")
        dt = datetime.datetime.strptime(self.start[0:10], "%Y-%m-%d")
        date = self.start[:10]
        while date <= self.end[:10]:
            dates.append(date)
            dt = dt + datetime.timedelta(1)
            date = dt.strftime("%Y-%m-%d")
        return dates
    def rain_data(self):
        '''
        1.根据起始时间计算面雨量的sql语句
        2.根据初始语句返回每站的总降水量
        '''
        station_all = self.station_all
        
        data_rain = {
            'rain_sum':{
                'time':[],
                'data':[]
            },
            'IIiii_data':{}
            
        }
        
        grouped_tTime = station_all.groupby('tTime')
        for i in grouped_tTime.size().index:  
            data= grouped_tTime.get_group(i)
            data['RR'].replace(-9999,np.nan,inplace=True)
            rain_mean = data['RR'].mean()/10.0
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
            data= grouped_IIiii.get_group(i)
            data['RR'].replace(-9999,np.nan,inplace=True)
            station_name = str(i) 
            data_rain['IIiii_data'][station_name] = data
            single_data = {}
            single_data['name'] = station_name
            single_data['value'] = [data['lon'].iloc[0],data['lat'].iloc[0],data['RR'].sum()/10.0]
            single_data['symble'] = 'circle'
            rain_scatter.append(single_data)
        self.station_data = data_rain['IIiii_data']
        self.rain_line =  [data_rain['rain_sum']['time'],data_rain['rain_sum']['data']]
        self.rain_scatter = rain_scatter  
    def wind_data(self):
        '''
        1.根据sql语句计算8及以上大风的分布和排序
        ''' 
        data_wind_list = []
        station_all = self.station_all
        sort_data = {
            'IIiii':[],
            'county':[],
            'town':[],
            'value':[]
        }
        #wind_max = station_all['fFy'].idxmax(axis=0)
        #station_max_name = station_all.iloc[wind_max,7]
        #station_max_data = self.station_data[station_max_name]
        grouped_IIiii = station_all.groupby('IIiii')
        for i in grouped_IIiii.size().index:  
            data= grouped_IIiii.get_group(i)
            if data['fFy'].max()>187:
                data_single = {}
                data_single['IIiii'] = data['IIiii'].iloc[0]
                data_single['county'] = data['county'].iloc[0]
                data_single['town'] = data['Town'].iloc[0]
                data_single['value'] = [data['lon'].iloc[0],data['lat'].iloc[0],data['fFy'].max()/10.0]
                data_single['symbol'] = 'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'
                data_single['symbolRotate'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
                sort_data['IIiii'].append(data['IIiii'].iloc[0])
                sort_data['county'].append(data['county'].iloc[0])
                sort_data['town'].append(data['Town'].iloc[0])
                sort_data['value'].append(data['fFy'].max()/10.0)
                data_wind_list.append(data_single)
        # 对数据进行排序
        max_sort = max(sort_data['value'])
        level_sort = np.linspace(start = 0.0, stop = max_sort, num = 9)
        sort_data = pd.DataFrame(sort_data)    
        sort_data['index'] = sort_data['value'].rank(ascending=0,method='dense')
        sort_out = sort_data.sort_values(by =['value'],ascending = [False]) 
        sort_html = ''
        # <tr><th>排序</th><th>乡镇</th><th>站点</th><th>能见度</th></tr>
        for i in sort_out['index']:
            table_html = '<tr><th>' + str(int(i)) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['town'].iloc[0]) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['IIiii'].iloc[0]) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['value'].iloc[0]) + '</th></tr>'
            sort_html = sort_html + table_html
        return data_wind_list , sort_html
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
            'IIiii':[],
            'county':[],
            'town':[],
            'value':[]
        }
        #wind_max = station_all['fFy'].idxmax(axis=0)
        #station_max_name = station_all.iloc[wind_max,7]
        #station_max_data = self.station_data[station_max_name]
        grouped_IIiii = station_all.groupby('IIiii')
        for i in grouped_IIiii.size().index:  
            data= grouped_IIiii.get_group(i)
            if data['fFy'].max()>187:
                data_single = {}
                data_single['IIiii'] = data['IIiii'].iloc[0]
                data_single['county'] = data['county'].iloc[0]
                data_single['town'] = data['Town'].iloc[0]
                data_single['value'] = [data['lon'].iloc[0],data['lat'].iloc[0],data['fFy'].max()/10.0]
                data_single['symbol'] = 'path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z'
                data_single['symbolRotate'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
                sort_data['IIiii'].append(data['IIiii'].iloc[0])
                sort_data['county'].append(data['county'].iloc[0])
                sort_data['town'].append(data['Town'].iloc[0])
                sort_data['value'].append(data['fFy'].max()/10.0)
                data_view_list.append(data_single)
        # 对数据进行排序
        max_sort = max(sort_data['value'])
        level_sort = np.linspace(start = 0.0, stop = max_sort, num = 9)
        sort_data = pd.DataFrame(sort_data)    
        sort_data['index'] = sort_data['value'].rank(ascending=0,method='dense')
        sort_out = sort_data.sort_values(by =['value'],ascending = [False]) 
        sort_html = ''
        # <tr><th>排序</th><th>乡镇</th><th>站点</th><th>能见度</th></tr>
        for i in sort_out['index']:
            table_html = '<tr><th>' + str(int(i)) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['town'].iloc[0]) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['IIiii'].iloc[0]) + '</th><th>' + \
                          str(sort_out[sort_out['index']==i]['value'].iloc[0]) + '</th></tr>'
            sort_html = sort_html + table_html
        return data_view_list , sort_html
    def temp_data(self):
        '''
        1.根据sql语句计算高低温的分布和排序
        '''
        temp_station_list = ['K8719','K8425','K8674']
        data_temp = []
        for i in temp_station_list:
            single_data = {}
            station_name = i
            single_data['name'] = i
            single_data['value'] = [self.station_data[i]['lon'].iloc[0],self.station_data[i]['lat'].iloc[0],self.station_data[i]['T'].max()/10.0,self.station_data[i]['T'].min()/10.0]
            data_temp.append(single_data)
        return data_temp   
    def pre_day(self,date):
        '''
        日报的响应时间、触发后统计对应天的灾情
        '2022-04-14', 200,'高温 浓雾', '降水 大风', 200
        '''
        print('计算指定日期的数据',date)
    def text_data(self):
        '''用来处理风雨情统计数据'''
        print('用来处理风雨情统计数据')
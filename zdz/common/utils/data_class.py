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
from datetime import timezone
from datetime import *
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

from pylab import *
from matplotlib.font_manager import FontProperties
import pymysql
from pymysql.converters import escape_string
import pickle
# 平滑
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling.models import Gaussian2D

# 雷达
import cinrad
# 自定义画图类
class nlcmap(LinearSegmentedColormap):
    """A nonlinear colormap"""

    name = 'nlcmap'

    def __init__(self, cmap, levels):
        self.cmap = cmap
        self.monochrome = self.cmap.monochrome
        self.levels = np.asarray(levels, dtype='float64')
        self._x = self.levels/ self.levels.max()
        self.levmax = self.levels.max()
        self.levmin = self.levels.min()
        self._y = np.linspace(self.levmin, self.levmax, len(self.levels))

    def __call__(self, xi, alpha=1.0, **kw):
        yi = np.interp(xi, self._x, self._y)
        return self.cmap(yi/self.levmax, alpha)
# 绘制数据的类
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
        data_min = data_xr_nc[:,:,:].min()
        data_max = data_xr_nc[:,:,:].max()
        data_xr = xr.DataArray(data_xr_nc[item,:,:],coords=[lat,lon], dims=["lat", "lon"])       
        #以下是核心api,实质为调用Cmaps基类的listmap()方法
        basicfile = '/home/workspace/Data/My_Git/web_met/'
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
        return data,data_min,data_max        
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
        else: 
            start = inittime + dtt.timedelta(hours = int(hours_list[item-1])) 
        label = str(start)[0:16] + " - " + self.plot_type
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

# 过程统计数据查询
class zdz_data:
    def __init__(self, start, end):
        self.start = start
        self.end = end  
    def sql_index(self):
        dailylist = self.day_button()
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        sql_location = f"""select lat,lon,sd.station_name, sd.station_no as station_no, sd.w_max as wind, max(w_dir) as w_dir,
        rr as rain,
        min(if(vis>0,vis,9999)) as view ,
        tn as t_min ,
        tx as t_max ,
        sd.station_city as city,
        sd.station_town as town,
        sd.station_county as county
        from (select station_no,max(if(w_max>0,w_max,null)) as wind,sum(p_total) as rr,min(t_min) as tn,max(t_max) as tx
        from station_data 
        where (datetime between '{self.start}' and '{self.end}') group by station_no) as wind
        inner join station_data as sd on sd.station_no = wind.station_no where( sd.w_max = wind.wind and datetime between '{self.start}' and '{self.end}')
        group by sd.datetime, sd.station_no, sd.w_max,lat,lon,station_name,station_city,station_town,station_county"""
        df_location = pd.read_sql(sql_location , con=conn)  
        point = []  
        table_data = []
        for i in range(len(df_location)):
            # 面雨量
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(df_location.iloc[i,6])
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [df_location.iloc[i,1], df_location.iloc[i,0]]
                }
            } 
            point.append(rain_data)
            # 数据表
            single = {
                "IIiii":str(df_location.iloc[i,3]),
                "city":str(df_location.iloc[i,10]),
                "town":str(df_location.iloc[i,11]),
                "county":str(df_location.iloc[i,12]),
                "StationName":str(df_location.iloc[i,2]),
                "fFy":str(df_location.iloc[i,4]/10.0),
                "dFy":str(df_location.iloc[i,5]),
                "RR":str(df_location.iloc[i,6]),
                "Tx":str(df_location.iloc[i,9]),
                "Tn":str(df_location.iloc[i,8]),
                "VV":str(df_location.iloc[i,7]),
                "lat":str(df_location.iloc[i,0]),
                "lon":str(df_location.iloc[i,1]),
                }
            table_data.append(single)
        return table_data,point,dailylist
    def day_button(self):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        start = self.start 
        end = self.end 
        select_rain = 'p_total' 
        select_tmax = 't_max'
        select_tmin = 't_min'
        select_wind = 'w_max'
        select_vis = 'vis'
        select_lat = 'lat'
        select_lon = 'lon'
        db_table = 'station_data'
        sql = f"""select datetime,station_no,
        {select_rain} ,{select_tmax},{select_tmin} ,{select_wind},{select_vis} 
        from {db_table}
        where (datetime between "{start}" and "{end}")"""
        df_location = pd.read_sql(sql , con=conn)
        df_location['day'] = df_location['datetime'].dt.day
        df_location['year'] = df_location['datetime'].dt.year
        df_location['month'] = df_location['datetime'].dt.month
        datagroup = df_location.groupby(['month','day','year'])
        dailylist = []
        for i in datagroup.size().index:
            df = datagroup.get_group(i)
            daily = {
                "time":str(df['datetime'].dt.year.iloc[0]) +"年"+ str(df['datetime'].dt.month.iloc[0]) + "月" + str(df['datetime'].dt.day.iloc[0]) + "日",
                "rain":False,
                "wind":False,
                "tmax":False,
                "tmin":False,
                "view":False,
            }
            if df[(df['vis']>0) & (df['vis']<750)]['vis'].count() >0:
                daily['view'] = True
            if df[df['t_max']>370]['t_max'].count()>0:
                daily['tmax'] = True
            if df[df['t_min']<-100]['t_min'].count() >0:
                daily['tmin'] = True
            if df[df['w_max']>108]['w_max'].count() >0:
                daily['wind'] = True
            if df[df['p_total']>0]['p_total'].count() >0:
                daily['rain'] = True
            dailylist.append(daily)
        return dailylist
    def sql_wind(self,start,end):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        sql_location = f"""select lat,lon,sd.station_name, sd.station_no as station_no, sd.w_max as wind, max(w_dir) as w_dir from
        (select station_no,max(if(w_max>0,w_max,null)) as wind from station_data where (datetime between '{start}' and '{end}') group by station_no) as wind
        inner join station_data as sd on sd.station_no = wind.station_no where( sd.w_max = wind.wind and datetime between '{start}' and '{end}')
        group by sd.datetime, sd.station_no, sd.w_max,lat,lon,station_name"""
        df_location = pd.read_sql(sql_location , con=conn)
        point = []  
        table_data = []
        for i in range(len(df_location)):
            # 面雨量
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(df_location.iloc[i,4]/10)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [df_location.iloc[i,1], df_location.iloc[i,0]]
                }
            } 
            point.append(rain_data)
            # 数据表
            single = {
                "IIiii":str(df_location.iloc[i,3]),
                "county":"台州市",
                "start":start,
                "end":end,
                "StationName":str(df_location.iloc[i,2]),
                "fFy":str(df_location.iloc[i,4]),
                "dFy":str(df_location.iloc[i,5]),
                "lat":str(df_location.iloc[i,0]),
                "lon":str(df_location.iloc[i,1]),
                }
            table_data.append(single)       
        return table_data , point
    def sql_rain(self,start,end):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        sql_location = f"""select lat,lon, station_name,station_no,
        sum(p_total>0) as rain
        from station_data where( datetime between '{start}' and '{end}' ) 
        group by lat,lon, station_name,station_no"""
        df_location = pd.read_sql(sql_location , con=conn)
        point = []  
        table_data = []
        for i in range(len(df_location)):
            # 面雨量
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(df_location.iloc[i,4])
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [df_location.iloc[i,1], df_location.iloc[i,0]]
                }
            } 
            point.append(rain_data)
            # 数据表
            single = {
                "IIiii":str(df_location.iloc[i,3]),
                "county":"台州市",
                "start":start,
                "end":end,
                "StationName":str(df_location.iloc[i,2]),
                "RR":str(df_location.iloc[i,4]),
                "lat":str(df_location.iloc[i,0]),
                "lon":str(df_location.iloc[i,1]),
                }
            table_data.append(single)       
        return table_data , point      
    def sql_temp(self,start,end,temp):
        if temp =="tmax":
            tempind = 5
        else:
            tempind = 4
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        sql_location = f"""select lat,lon, station_name,station_no,
        min(t_min) as t_min ,
        max(t_max) as t_max 
        from station_data where( datetime between '{start}' and '{end}' ) 
        group by lat,lon, station_name,station_no"""
        df_location = pd.read_sql(sql_location , con=conn)
        point = []  
        table_data = []
        for i in range(len(df_location)):
            # 面雨量
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(df_location.iloc[i,tempind])
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [df_location.iloc[i,1], df_location.iloc[i,0]]
                }
            } 
            point.append(rain_data)
            # 数据表
            single = {
                "IIiii":str(df_location.iloc[i,3]),
                "county":"台州市",
                "start":start,
                "end":end,
                "StationName":str(df_location.iloc[i,2]),
                "t_min":str(df_location.iloc[i,4]),
                "t_max":str(df_location.iloc[i,5]),
                "lat":str(df_location.iloc[i,0]),
                "lon":str(df_location.iloc[i,1]),
                }
            table_data.append(single)       
        return table_data , point 
    def sql_view(self,start,end):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="tzweb")
        sql_location = f"""select lat,lon, station_name,station_no,
        min(vis) as rain
        from station_data where( datetime between '{start}' and '{end}' ) 
        group by lat,lon, station_name,station_no"""
        df_location = pd.read_sql(sql_location , con=conn)
        point = []  
        table_data = []
        for i in range(len(df_location)):
            # 面雨量
            rain_data = {
                "type": "Feature",
                "properties": {
                    "value": str(df_location.iloc[i,4])
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [df_location.iloc[i,1], df_location.iloc[i,0]]
                }
            } 
            point.append(rain_data)
            # 数据表
            single = {
                "IIiii":str(df_location.iloc[i,3]),
                "county":"台州市",
                "start":start,
                "end":end,
                "StationName":str(df_location.iloc[i,2]),
                "VV":str(df_location.iloc[i,4]),
                "lat":str(df_location.iloc[i,0]),
                "lon":str(df_location.iloc[i,1]),
                }
            table_data.append(single)       
        return table_data,point
    def sql_click(self,start,end,station,click_type):
        conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        if click_type!='wind':
            if station[0]!='K':
                sql_location = f"""select tTime,{click_type} from TAB_Aws2019 where ( IIiii={station} and tTime between '{start}' and '{end}')"""
            else:
                sql_location = f"""select tTime,{click_type} from TAB_Mws2019 where ( IIiii='{station}' and tTime between '{start}' and '{end}')"""
        else:
            if station[0]!='K':
                sql_location = f"""select tTime,dFy,fFy from TAB_Aws2019 where ( IIiii={station} and tTime between '{start}' and '{end}')"""
            else:
                sql_location = f"""select tTime,dFy,fFy from TAB_Mws2019 where ( IIiii='{station}' and tTime between '{start}' and '{end}')"""
        
        df_location = pd.read_sql(sql_location , con=conn) 
        time = df_location.sort_values(by="tTime")["tTime"].to_list() 
        if click_type == "wind":
            values = [df_location.sort_values(by="tTime")['fFy'].to_list(),df_location.sort_values(by="tTime")['dFy'].to_list()]
        else:
            values = df_location.sort_values(by="tTime")[click_type].to_list()     
        return time,values,click_type,station

# ec数据的处理和对接
class ec_data_point:
    def __init__(self,start_time,end_time):
        self.timelist = [0,2,4,6,8,10,12,14,16,18,20,22,24,25,
                         26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                        41,42,43,44,45,46,47,48,49,50,51,52]
        self.test_time = '2022041700'
        self.file_path = "/home/workspace/Data/My_Git/" + self.test_time + "/" 
        self.lat_list = [27.6, 28.1, 28.4, 29.1, 29.1, 29.8, 28.7, 28.5, 28.5, 28.6]
        self.lon_list = [120.7, 121.1, 121.3, 121.2, 121.0, 120.7, 121.1, 121.4, 121.4, 121.2]
        self.exlat = [120.2,121.9]
        self.exlon = [27.3,30.6]
        self.name = ["台州", "玉环", "温岭", "三门", "天台", "仙居", "临海", "路桥", "椒江", "黄岩"]
        self.name_en = ['taizhou','yuhuan','wenling','sanmen','tiantai','xianju','linhai','luqiao','jiaojiang','huangyan']
        self.cp,self.t2,self.tp = self.read_data()
    # 外部函数
    def transform_from_latlon(self,lat, lon):
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        trans = Affine.translation(lon[0], lat[0])
        scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
        return trans * scale    
    def rasterize(self,shapes, coords, latitude='latS', longitude='lonS',fill=np.nan, **kwargs):
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
        xr_da[coord_name] = self.rasterize(shapes, xr_da.coords, longitude='lonS', latitude='latS')
        return xr_da
    # 读取数据
    def read_data(self):
        '''读取数据'''
        files = os.listdir(self.file_path)
        tp_list = [] 
        cp_list = [] 
        t2_list = [] 
        for fileitem in self.timelist:
            f=xr.open_dataset(self.file_path +files[fileitem],decode_times=False)
            tp = f.tp.sel(lonS=slice(118,123),latS=slice(32,26))
            cp = f.cp.sel(lonS=slice(118,123),latS=slice(32,26))
            t2 = f.t2.sel(lonS=slice(118,123),latS=slice(32,26)) 
            tp_list.append(tp)  
            cp_list.append(cp)  
            t2_list.append(t2)
            del f,tp,cp,t2  
        cp_all = xr.concat(cp_list,dim="time")
        t2_all = xr.concat(t2_list,dim="time")
        tp_all = xr.concat(tp_list,dim="time")
        return cp_all,t2_all,tp_all
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
        list_data= data.sel(lonS=lon, latS=lat,method='nearest').to_pandas().tolist()
        out_list = self.accum_data(list_data)
        return out_list
    def plot_line(self,lat,lon):
        '''返回单点的降水气温曲线图'''
        cp_line = self.rain_data(lat,lon,self.cp)
        totle_line = self.rain_data(lat,lon,self.tp)
        tp_line = []
        for i in range(len(totle_line)):
            tp_line.append(totle_line[i] - cp_line[i])
        t2_line = self.t2.sel(lonS=lon, latS=lat,method='nearest').to_pandas().tolist()
        return cp_line,tp_line,t2_line
    # 设置读取sql中的数据
    def read_sql(self):
        '''将数据传到sql中'''
        time = '2022041700'
        mydb =pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="051219",
            database="tzweb"
        )
        sql="""select data from  ec_data order by -create_time"""
        data=pd.read_sql(sql,mydb)
        # 数据转为python类型
        data_org = data.iloc[0].data
        data_python = json.loads(data_org)
        return data_python
    # 定时任务
    def conuty_data(self):
        '''将数据整理成json并存储到MySQL'''
        data_list = []
        mysql_setting = {
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'passwd': '051219',
            # 数据库名称
            'db': 'tzweb',
            'charset': 'utf8'
        }
        mydb =pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="051219",
            database="tzweb"
        )
        for i in range(len(self.lat_list)):
            lat = self.lat_list[i]
            lon = self.lon_list[i]
            cp_line,tp_line,t2_line = self.plot_line(lat,lon)
            cp = [0 if np.isnan(x) else x for x in cp_line]
            tp = [0 if np.isnan(x) else x for x in tp_line]
            t2 = [0 if np.isnan(x) else x for x in t2_line]
            data_single = {
                'name':self.name_en[i],
                'cp':cp,
                'tp':tp,
                't2':t2
            }
            data_list.append(data_single)
        # 数据库插入数据
        create_time = dtt.date.today()
        update_time = dtt.date.today()
        time = self.date_time
        model_type = "EC"
        model_city ="台州市" #self.name_en[i]
        data = json.dumps(data_list)
        create_user=0
        update_user=0
        cursor = mydb.cursor()
        sql = '''insert INTO ec_data(create_time,update_time,model_type,model_city,time,create_user,update_user,data) values ('{create_time}','{update_time}','{model_type}','{model_city}','{time}',0,0,'{data}') '''            
        #sql = '''replace into ec_data(model_type,model_city,time,create_user,update_user,data) select '{model_type}','{model_city}','{time}',0,0,'{data}'  '''  
        rsql = sql.format(create_time=create_time,update_time=update_time,model_type=model_type,model_city=model_city,time=time,create_user=create_user,update_user=update_user,data=escape_string(data)) 
        cursor.execute(rsql)               
        mydb.commit()
        cursor.close()
        mydb.close()
        return data_list
    # 截面数据的读取
    def rander_leaflet(self,start_time,end_time):
        '''返回数据'''
        if start_time==0:
            rain = end_rain 
        else:
            start_rain = self.tp[start_time,:,:]
            end_rain = self.tp[end_time,:,:]
            rain = end_rain - start_rain 
        filepath = "static/data/shpfile/"
        shp_da = self.add_shape_coord_from_data_array(rain, filepath+"taizhou.shp", "remain") 
        taizhou = shp_da.where(shp_da.remain<7, other=99999)
        len_lat = len(taizhou.latS.data)
        len_lon = len(taizhou.lonS.data)
        data = []
        exdata = []
        for j in range(len(self.exlat)):
            exlat = self.exlat[j]
            exlon = self.exlon[j]
            value = rain.sel(lonS=exlon, latS=exlat,method='nearest').to_pandas().tolist()
            # value = np.around(value,decimals=2)
            if not value:
                value = 0
            exsingle = {
                "type": "Feature",
                "properties": {
                    "value": str(value)
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [exlat,exlon ]
                }
            }
            exdata.append(exsingle)
        #print("数据测试",data)
        for i in range(len_lon-1):
            for j in range(len_lat-1):
                y0 = taizhou.latS.data[j]
                x0 = taizhou.lonS.data[i]
                if taizhou.data[j, i]!=99999:
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
        alldata = data + exdata
        return alldata




# 自订正绘图的类
class sql_plot:
    def __init__(self, start_time,end_time,select_type):
        self.start = start_time 
        self.end = end_time
        self.type = select_type
        self.station = ('58566','58567','58564','58569','58556','58555','58560','58654','58657','58658','58763','58656','58761','58559','58568','58660','58662','58663','58653','58652','58655','58665','58661','58666','58664','58669','58667','58668','K8705','K8706','K8818','K8609','K8821','K8611','K8903','K8910','K8282','K8217','K8201','K8301','K8413','K8505')
        self.Aws = ('58566','58567','58564','58569','58556','58555','58560','58654','58657','58658','58763','58656','58761','58559','58568','58660','58662','58663','58653','58652','58655','58665','58661','58666','58664','58669','58667','58668')
        self.Mws = ('K8705','K8706','K8818','K8609','K8821','K8611','K8903','K8910','K8282','K8217','K8201','K8301','K8413','K8505')  
        self.server = '127.0.0.1' 
        self.user = 'root'
        self.password = '051219'# 连接密码   
        self.port = 3306
        self.database = "ZJSZDZDB"
    def return_zdz_data(self):
        self.start ="'" +  self.start + "'" 
        self.end = "'" +  self.end + "'" 
        conn = pymysql.connect(
            host = self.server, 
            port = self.port,
            user = self.user, 
            password = self.password,
            database = self.database)
        sql_Aws = f'''select a.IIiii,b.lat,b.lon,SUM(a.RR)/10.0
            from TAB_Aws2019 AS a left join TAB_StationInfo AS b on a.IIiii=b.IIiii 
            WHERE (b.IIiii  IN  {self.Aws} and (tTime BETWEEN {self.start} and {self.end} )) 
            GROUP BY a.IIiii,lat,lon'''
        sql_Mws = f'''select a.IIiii,b.lat,b.lon,SUM(a.RR)/10.0
            from TAB_Mws2019 AS a left join TAB_StationInfo AS b on a.IIiii=b.IIiii 
            WHERE (b.IIiii  IN  {self.Mws} and (tTime BETWEEN {self.start} and {self.end} )) 
            GROUP BY a.IIiii,lat,lon '''
        dfa = pd.read_sql(sql_Aws, con=conn)
        dfm = pd.read_sql(sql_Mws, con=conn)
        df = pd.concat([dfa,dfm])
        # 输出数据
        data_canvas = {
            "station_list": [],
            "station": []
        }
        for i in range(df.shape[0]):
            station_data = []
            station_data.append(round(df.iloc[i, 2],2))#lon
            station_data.append(round(df.iloc[i, 1],2))#lat
            value = round(df.iloc[i, 3],2)
            if value!=-9999:
                station_data.append(value)
                data_canvas['station_list'].append(df.iloc[i,0])
                data_canvas['station'].append(station_data)
        canvas_data = data_canvas
        return canvas_data
    def return_ec_data(self):
        # 数据---文档
        test_time = '2022041700'
        file_path = "/home/workspace/Data/My_Git/" + test_time + "/"
        # 文件--数据
        conn = pymysql.connect(
            host = self.server, 
            port = self.port,
            user = self.user, 
            password = self.password,
            database = self.database)
        sql_location = f'''
            select IIiii,lat,lon from TAB_StationInfo
            where (IIiii in  {self.station} )     
        '''
        df = pd.read_sql(sql_location, con=conn)
        # 查询ec数据
        files = os.listdir(file_path)  
        fstart = xr.open_dataset(file_path + files[int(self.start)])
        tp_start = fstart.tp
        fend = xr.open_dataset(file_path + files[int(self.end)])
        tp_end = fend.tp 
        tp = tp_end.data - tp_start.data
        # 输出数据
        data_canvas = {
            "station_list": [],
            "station": []
        }
        for i in range(df.shape[0]):
            station_data = []
            
            station_data.append(round(df.iloc[i, 2],2))#lon
            station_data.append(round(df.iloc[i, 1],2))#lat
            # 求数据之差
            tp0 = tp_end.sel(lonS=round(df.iloc[i, 2],2), latS=round(df.iloc[i, 1],2),method='nearest').data.tolist()[0]
            tp1 = tp_start.sel(lonS=round(df.iloc[i, 2],2), latS=round(df.iloc[i, 1],2),method='nearest').data.tolist()[0]
            value = round((tp0-tp1),2)
            if not np.isnan(value):
                station_data.append(value)
                data_canvas['station_list'].append(df.iloc[i,0])
                data_canvas['station'].append(station_data)
        canvas_data = data_canvas
        return canvas_data
    def return_data(self):
        if self.type == "zdz":
            canvas_data = self.return_zdz_data()
        else:
            canvas_data = self.return_ec_data()
        return canvas_data

#上传EC的数据
class ec_data_upload:
    def __init__(self):
        self.timelist = [0,2,4,6,8,10,12,14,16,18,20,22,24,25,
                         26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
                        41,42,43,44,45,46,47,48,49,50,51,52]
        self.date_time = self.time_file()
        self.file_path = "/home/liyuan3970/Data/My_Git/" + self.date_time + "/" 
        self.lat_list = [27.6, 28.1, 28.4, 29.1, 29.1, 29.8, 28.7, 28.5, 28.5, 28.6]
        self.lon_list = [120.7, 121.1, 121.3, 121.2, 121.0, 120.7, 121.1, 121.4, 121.4, 121.2]
        self.name = ["台州", "玉环", "温岭", "三门", "天台", "仙居", "临海", "路桥", "椒江", "黄岩"]
        self.name_en = ['taizhou','yuhuan','wenling','sanmen','tiantai','xianju','linhai','luqiao','jiaojiang','huangyan']
        self.cp,self.t2,self.tp = self.read_data()
    def time_file(self):
        today = dtt.date.today()
        yesterday = today - dtt.timedelta(days = 1) 
        year = yesterday.strftime('%Y-%m-%d_%H%M%S')[0:4]
        month =  yesterday.strftime('%Y-%m-%d_%H%M%S')[5:7]
        day = yesterday.strftime('%Y-%m-%d_%H%M%S')[8:10]
        file_time = year + month + day + "00"
        file_time = '2022041700'
        return file_time      
    def read_data(self):
        '''读取数据'''
        files = os.listdir(self.file_path)
        tp_list = [] 
        cp_list = [] 
        t2_list = [] 
        for fileitem in self.timelist:
            f=xr.open_dataset(self.file_path +files[fileitem],decode_times=False)
            tp = f.tp.sel(lonS=slice(118,123),latS=slice(32,26))
            cp = f.cp.sel(lonS=slice(118,123),latS=slice(32,26))
            t2 = f.t2.sel(lonS=slice(118,123),latS=slice(32,26)) 
            tp_list.append(tp)  
            cp_list.append(cp)  
            t2_list.append(t2)
            del f,tp,cp,t2  
        cp_all = xr.concat(cp_list,dim="time")
        t2_all = xr.concat(t2_list,dim="time")
        tp_all = xr.concat(tp_list,dim="time")
        return cp_all,t2_all,tp_all
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
        list_data= data.sel(lonS=lon, latS=lat,method='nearest').to_pandas().tolist()
        out_list = self.accum_data(list_data)
        return out_list
    def plot_line(self,lat,lon):
        '''返回单点的降水气温曲线图'''
        cp_line = self.rain_data(lat,lon,self.cp)
        totle_line = self.rain_data(lat,lon,self.tp)
        tp_line = []
        for i in range(len(totle_line)):
            tp_line.append(totle_line[i] - cp_line[i])
        t2_line = self.t2.sel(lonS=lon, latS=lat,method='nearest').to_pandas().tolist()
        return cp_line,tp_line,t2_line
    def conuty_data(self):
        '''将数据整理成json并存储到MySQL'''
        data_list = []
        mysql_setting = {
            'host': '127.0.0.1',
            'port': 3306,
            'user': 'root',
            'passwd': '051219',
            # 数据库名称
            'db': 'tzweb',
            'charset': 'utf8'
        }
        mydb =pymysql.connect(
            host="127.0.0.1",
            user="root",
            password="051219",
            database="tzweb"
        )
        for i in range(len(self.lat_list)):
            lat = self.lat_list[i]
            lon = self.lon_list[i]
            cp_line,tp_line,t2_line = self.plot_line(lat,lon)
            cp = [0 if np.isnan(x) else x for x in cp_line]
            tp = [0 if np.isnan(x) else x for x in tp_line]
            t2 = [0 if np.isnan(x) else x for x in t2_line]
            data_single = {
                'name':self.name_en[i],
                'cp':cp,
                'tp':tp,
                't2':t2
            }
            data_list.append(data_single)
        # 数据库插入数据
        create_time = dtt.date.today()
        update_time = dtt.date.today()
        time = self.date_time
        model_type = "EC"
        model_city ="台州市" #self.name_en[i]
        data = json.dumps(data_list)
        create_user=0
        update_user=0
        cursor = mydb.cursor()
        sql = '''insert INTO ec_data(create_time,update_time,model_type,model_city,time,create_user,update_user,data) values ('{create_time}','{update_time}','{model_type}','{model_city}','{time}',0,0,'{data}') '''            
        #sql = '''replace into ec_data(model_type,model_city,time,create_user,update_user,data) select '{model_type}','{model_city}','{time}',0,0,'{data}'  '''  
        rsql = sql.format(create_time=create_time,update_time=update_time,model_type=model_type,model_city=model_city,time=time,create_user=create_user,update_user=update_user,data=escape_string(data)) 
        cursor.execute(rsql)               
        mydb.commit()
        cursor.close()
        mydb.close()

# 自动站分析系统的核心class

class station_zdz:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
    def time_today(self):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        return today
    def get_redis(self,date_type):
        '''根据date_type向redis中获取数据'''
        value = self.rs.get(date_type)
        if value:
            data = pickle.loads(value)
        else:
            data = None       
        #data = pickle.loads(self.rs.get(date_type))
        # 解析数据
        return data
    def hours_wind(self,data):
        data['TTime'] = pd.to_datetime(data['TTime'])
        data['Year'] = data['TTime'].dt.year
        data['Month'] = data['TTime'].dt.month
        data['Day'] = data['TTime'].dt.day
        data['Hour'] = data['TTime'].dt.hour
        grouped_IIIII = data.groupby(['Day','Hour'])
        time_list = []
        dir_list = []
        speed_list = []
        for i in grouped_IIIII.size().index:
            single = grouped_IIIII.get_group(i)
            value = single[ single['fFy']== single['fFy'].max()].head(1)
            time_list.append(value['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S').values[0])
            speed_list.append(value['fFy'].values[0])
            dir_list.append(value['dFy'].values[0])
        return time_list,speed_list,dir_list
    def single_station(self,station):
        '''获取单站数据并解析'''
        # 获取当前时间
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        end = today.strftime('%Y-%m-%d %H:%M:%S')
        offset = dtt.timedelta(days=-1)
        start = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        # 造假数据
        start = '2019-08-08 16:53:00'
        end = '2019-08-09 16:53:00' 
        yesday = start[0:10] + " 20:00:00"
        today = end[0:10] + " 20:00:00"
        hours = dtt.datetime.strptime(end,'%Y-%m-%d %H:%M:%S').hour
        # 数据库中读取单站数据并解析
        sql = """select TTime,IIIII,Ri,T,V,FF as fFy,DF as dFy 
        from Tab_AM_M where ( TTime between '{start_time}' and '{end_time}' and IIIII='{station}')  order by TTime  """
        rsql = sql.format(start_time=start,end_time=end,station=station)
        data = pd.read_sql(rsql , con=self.conn) 
        wind_time,wind_speed,wind_dir =  self.hours_wind(data)
        wind_data = pd.DataFrame(data={'TTime':wind_time,'dFy':wind_dir,'fFy':wind_speed})   
        data['TTime'] = data['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        wind_data['TTime'] = pd.to_datetime(wind_data['TTime'])
        if hours>=20:
            # 实时数据
            now_data = data[data['TTime']>=today]
            now_data = now_data[now_data['TTime']<=end]
            now_wind = wind_data[wind_data['TTime']>=today]
            now_wind = now_wind[now_wind['TTime']<=end]
            # 历史数据
            his_data = data[data['TTime']>start]
            his_data = his_data[his_data['TTime']<=today]
            his_wind = wind_data[wind_data['TTime']>start]
            his_wind = his_wind[his_wind['TTime']<=today]
        else:
            # 实时数据
            now_data = data[data['TTime']>=yesday]
            now_data = now_data[now_data['TTime']<=end]
            now_wind = wind_data[wind_data['TTime']>=yesday]
            now_wind = now_wind[now_wind['TTime']<=end]
            # 历史数据
            his_data = data[data['TTime']>start]
            his_data = his_data[his_data['TTime']<=yesday]
            his_wind = wind_data[wind_data['TTime']>start]
            his_wind = his_wind[his_wind['TTime']<=yesday]
        his_wind['TTime'] = his_wind['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        now_wind['TTime'] = now_wind['TTime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        output = pd.concat([now_data,his_data])  
        history = his_data.to_json(orient='values',force_ascii=False)
        nowdata = now_data.to_json(orient='values',force_ascii=False)
        windhis = his_wind.to_json(orient='values',force_ascii=False)
        windnow = now_wind.to_json(orient='values',force_ascii=False)
        output = {
            "now":nowdata,
            "his":history,
            "wind_time":wind_time,
            "wind_speed":wind_speed,
            "wind_dir":wind_dir
        }
        # 解析数据成两个序列
        return nowdata,history,windhis,windnow
    def sql_hours(self,boundary,value_index):
        """同步小时数据表"""
        start_times = "2023-07-18 01:15:00"
        end_times = "2023-07-18 13:27:00"
        if value_index==3:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(wind) as wind
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and wind>0 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['WIN_S_Gust_Max'] = data.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
            data['WIN_D_Gust_Max'] = data.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
            data['value'] = data['WIN_S_Gust_Max']
        elif value_index==0:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, sum(rain) as PRE_sum
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and rain<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['PRE_sum']
        elif value_index==1:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(tmax) as tmax
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and tmax<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmax']
        elif value_index==2:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(tmin) as tmin
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and tmin<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmin']
        elif value_index==4:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(view) as view
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}' and view<30000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['view']
        return data 
    def sql_range(self,boundary,value_index):
        # 尽量两小时为主---
        start_times = "2023-07-18 11:15:00"
        end_times = "2023-07-18 13:27:00"
        if value_index==3:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(WIN_S_Gust_Max*10000 + WIN_D_Gust_Max) as wind
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and WIN_S_Gust_Max<5000 and WIN_D_Gust_Max<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['WIN_S_Gust_Max'] = data.apply(lambda x: (x.wind - int(str(int(x.wind))[-3:]))/10000, axis = 1)
            data['WIN_D_Gust_Max'] = data.apply(lambda x: int(str(int(x.wind))[-3:]), axis = 1)
            data['value'] = data['WIN_S_Gust_Max']
        elif value_index==0:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, sum(PRE) as PRE_sum
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and PRE<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['PRE_sum']
        elif value_index==1:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, max(TEM) as tmax
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and TEM<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmax']
        elif value_index==2:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(TEM) as tmin
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and TEM<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['tmin']
        elif value_index==4:
            sql = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, min(VIS_HOR_1MI) as view
            from table_station_min 
            where Datetime between '{start_times}' and '{end_times}' and VIS_HOR_1MI<30000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}
            group by Station_Id_C""" 
            rsql = sql.format(start_times=start_times,end_times=end_times,lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
            data = pd.read_sql(rsql, con=self.conn)
            data['value'] = data['view']
        return data
    def sql_now(self,boundary,value_index):
        '''根据date_type向redis中获取数据'''
        now_times = "2023-07-18 13:27:00"
        value_list = ["PRE","TEM","TEM","WIN_S_Gust_Max","VIS_HOR_1MI"]
        if value_index!=3:
            sql = """select City,Cnty,Station_Id_C,Station_levl,Province,Station_Name,Town,Alti,Lat,Lon, {value} 
            from table_station_min 
            where Datetime= '{time}' and {value}<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}"""
        else:
            sql = """select City,Cnty,Station_Id_C,Station_levl,Province,Station_Name,Town,Alti,Lat,Lon, {value}, WIN_D_Gust_Max
            from table_station_min 
            where Datetime= '{time}' and {value}<5000 and Lat> {lat0} and Lat< {lat1} and Lon >{lon0} and Lon<{lon1}"""            
        rsql = sql.format(time=now_times,value=value_list[value_index],lat0=boundary[0],lat1 = boundary[1],lon0 = boundary[2],lon1 = boundary[3])
        data = pd.read_sql(rsql, con=self.conn)
        data['value'] = data[value_list[value_index]]
        return data
    def get_regin(self,boundary,table_type,tables_name,value_index,zoom):
        '''解码单站数据'''
        ascending_list = [False,False,True,False,True]
        if tables_name=="now":
            data = self.sql_now(boundary,value_index)
            boundary_data =  data
        else:
            if tables_name in ["3hours","2hours","1hours","45mins","30mins"]:
                data = self.sql_range(boundary,value_index)
                boundary_data =  data
            else:
                data = self.sql_hours(boundary,value_index)
                lat0 = boundary[0]
                lat1 = boundary[1]
                lon0 = boundary[2]
                lon1 = boundary[3]
                boundary_data =  data[(data['Lat']>lat0) & (data['Lat']<lat1)  &  (data['Lon']<lon1) & (data['Lon']>lon0)]  
        if table_type=="nation":
            remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
        if table_type=="regin":
            remain = boundary_data[boundary_data['Station_levl']==14]
        elif table_type=="all":
            remain = boundary_data
        elif table_type=="main":
            remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==15)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
        elif table_type=="auto":   
            if value_index!=0:        
                if (boundary[3]-boundary[2])<2.9:
                    remain = boundary_data
                elif (boundary[3]-boundary[2])<6:
                    remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==15)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
                else:
                    remain = boundary_data[(boundary_data['Station_levl']==11)| (boundary_data['Station_levl']==12)| (boundary_data['Station_levl']==13)| (boundary_data['Station_levl']==16)]
            else:
                remain = boundary_data


        remain.sort_values(by="value",axis=0,ascending=ascending_list[value_index],inplace=True) # 从大到小 
            
        output = remain.to_json(orient='records',force_ascii=False)
        return output


# 气象服务快报的相关
class station_plot:
    def __init__(self):
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        self.shp_path = "static/data/shpfile/country/"
    # 外部函数
    def colormap(self,plot_value,color_label):
        '''色标的自定义'''
        plt.rcParams['axes.facecolor']='snow'
        # 降水
        if plot_value=="rain":
            if color_label =="rain_24hours":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 24降水
                levels = [0,1,10,25,50,100,250,1000]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
            elif color_label =="rain_12hours":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 12降水
                levels = [0,1,5,15,30,70,140,250]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
            elif color_label =="rain_06hours":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 06降水
                levels = [0,1,4,13,25,60,120,250]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
            elif color_label =="rain_03hours":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 03降水
                levels = [0,1,3,10,20,50,70,150]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
            elif color_label =="rain_01hours":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 01降水
                levels = [0,1,2,7,15,40,50,100]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
        elif plot_value=="tmax" or plot_value=="tmin":
            if color_label =="temp_normal":
                #colorslist = ['#1F1FFF',"#3B3BFF","#5757FF","#7272FF","#8F8FFF","#ABABFF","#C7C7FF","#E3E3FF","#FDFC8B","#F8E08B","#F3C36F","#EFA76E","#EC8A51","#F31717"]# 气温
                #colornum = len(colorslist)
                level = list(np.linspace(self.min-1, self.max+1, num=14, endpoint=True, retstep=False, dtype=None))
                levels = [round(i,1) for i in level]
                cmap_nonlin = 'seismic'#'coolwarm'#
            elif color_label =="temp_high":
                level = list(np.linspace(self.min-1, self.max+1, num=14, endpoint=True, retstep=False, dtype=None))
                levels = [round(i,1) for i in level]
                cmap_nonlin = 'Reds'
            elif color_label =="temp_low":
                level = list(np.linspace(self.min-1, self.max+1, num=14, endpoint=True, retstep=False, dtype=None))
                levels = [round(i,1) for i in level]
                cmap_nonlin = 'Blues_r'
        elif plot_value=="wind":
            if color_label =="wind_normal":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 风力
                levels = [0,1,10,15,25,50,100,250]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels) 
            elif color_label =="wind_other":
                colorslist = ['#FFFFFF','#A6F28f','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 风力其他
                levels = [0,1,10,15,25,50,100,250]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)
        elif plot_value=="view":
            if color_label =="view_normal":
                colorslist = ['#A93434','#FF9600','#FFFD37',"#55FF37","#ABF3D3","#5EC6EB","#B1F1EF"]# 能见度
                levels = [0,50,200,500,1000,1500,3000,20000]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels) 
            elif color_label =="view_other":
                colorslist = ['#FFFFFF','#FF9600','#3DBA3D',"#61B8FF","#0000E1","#FA00FA","#800040"]# 能见度
                levels = [0,1,10,15,25,50,100,250]
                cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=7)
                cmap_nonlin = nlcmap(cmaps, levels)             
        return cmap_nonlin ,levels
    def city_shp(self,data_xr):
        shp_da = self.add_shape_coord_from_data_array(data_xr,self.shp_path+"taizhou.shp", "country")
        awash_da = shp_da.where(shp_da.country<7, other=np.nan)
        return awash_da
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
            if shape_rec.record[0]:  
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
    # 内部函数
    def get_sql_data(self,start,end):      
        '''sql 获取数据'''
        start = '2023-07-17 13:00:00'
        end = '2023-07-18 12:00:00'
        sqltall = """select max(City) as City,max(Cnty) as Cnty, Station_Id_C , max(Province) as Province,max(Station_levl) as Station_levl,
            max(Station_Name) as Station_Name, max(Town) as Town, max(Alti) as Alti, max(Lat) as Lat,max(Lon) as Lon, sum(rain) as rain,max(tmax) as tmax,min(tmin) as tmin ,max(wind)as wind 
            from table_station_hour 
            where Datetime between '{start_times}' and '{end_times}'
            group by Station_Id_C""" 
        rsql = sqltall.format(start_times=start,end_times=end)     
        data = pd.read_sql(rsql , con=self.conn)
        # print("测试数据",data)
        return data
    def extra_download(self,start,end,city,country):
        station_all = self.get_sql_data(start,end)
        # station_all['lock'] = "true"
        if country=="all":
            country = city
            data = station_all[station_all['City']==city]
        else:
            country = country
            data = station_all[station_all['Cnty']==country]
        output = data.to_json(orient='records',force_ascii=False)
        return output
    def get_plot_data(self,city,country,plot_data,plot_value):
        '''解析前段获取的数据'''
        x = []
        y = []
        z = []
        for i in range(len(plot_data)):
            if plot_data[i][plot_value]!=-9999.0:
                x.append(plot_data[i]['Lon'])
                y.append(plot_data[i]['Lat'])
                z.append(plot_data[i][plot_value]/10)
        lat = np.array(y)
        lon = np.array(x)
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
    def plot(self,start,end,city,country,plot_data,plot_value,color_label):
        data_xr = self.get_plot_data(city,country,plot_data,plot_value)
        # 平滑
        #data_xr = scipy.ndimage.zoom(data_xr, 3)
        # ##########色标和大小#############################
        cmaps ,levels = self.colormap(plot_value,color_label)
        fig = plt.figure(figsize=[10,10]) 
        ax = fig.add_subplot(111)
        awash_da = self.city_shp(data_xr)
        lat = data_xr.lat
        lon = data_xr.lon
        m = Basemap(llcrnrlon=120.2,
            llcrnrlat=27.8,
            urcrnrlon=122,
            urcrnrlat=29.5,
            resolution = None, 
            projection = 'cyl')
        lons, lats = np.meshgrid(lon, lat)
        cs =m.contourf(lons,lats,data_xr,ax=ax, cmap=cmaps,levels =levels,add_labels=True)
        ##########标题#############################
        font = FontProperties(fname="static/data/simkai.ttf", size=14)
        label  = start + " 至 " + end + "   "  + "累积雨量"
        plt.text(120.2,29.4, label,fontsize=15, fontproperties=font)
        ##########标题#############################
        m.readshapefile(self.shp_path + 'taizhou','taizhou',color='k',linewidth=1.2)
        plt.axis('off')
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
        self.basemask(cs, ax, m, self.shp_path+'taizhou') 
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight',transparent=True)  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        return imd   

class station_sql_data:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
        self.conn = pymysql.connect(host="127.0.0.1",port=3306,user="root",passwd="051219",db="ZJSZDZDB")
        self.redis_name = {
            "24hours":"table_24hours",
            "12hours":"table_12hours",
            "6hours":"table_6hours",
            "3hours":"table_3hours",
            "2hours":"table_2hours",
            "1hours":"table_1hours",
            "45mins":"table_45mins",
            "30mins":"table_30mins",
            "15mins":"table_15mins"         
        }
    def rain_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,sum(Ri) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.Ri!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_rain"
        self.rs.set(redis_name_str, pickle.dumps(data))
        
    def wind_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select wind1.IIiii as IIiii,wind1.StationName,wind1.Province,wind1.City,wind1.County,wind1.Town,wind1.ZoomLevel,wind1.Type, max(wind1.lon) as lon, max(wind1.lat) as lat, wind1.fFy as value, max(wind2.dFy) as dFy  from 
        (SELECT  st.IIiii AS  IIiii,StationName,Province,City,County,Town,ZoomLevel,Type, max(st.lon) AS  lon, max(st.lat) AS  lat, max(sd.fFy) AS  fFy  FROM  TAB_StationInfo  AS  st
        LEFT JOIN Tab_AM_M  AS  sd ON  sd.IIIII = st.IIiii WHERE  (st.Province='浙江' AND  (sd.TTime >'{time}') and sd.fFy!=-9999 and sd.dFy!=-9999 ) 
        GROUP BY  st.IIiii,st.StationName,st.Province,st.City,st.County,st.Town,st.ZoomLevel,st.Type) as wind1
        inner join Tab_AM_M as wind2 on wind2.IIIII = wind1.IIiii and wind2.fFy = wind1.fFy and wind2.TTime >'{time}'
        group by wind1.IIiii, wind1.fFy,StationName,Province,City,County,Town,ZoomLevel,Type"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_wind"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def tmax_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,max(T) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.T!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_tmax"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def tmin_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql ="""select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,min(T) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and  ta.T!=-9999) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_tmin"
        self.rs.set(redis_name_str, pickle.dumps(data))
    def view_sql(self,tables_name,timesdelay):
        SHA_TZ = timezone(
            dtt.timedelta(hours=8),
            name='Asia/Shanghai',
        )
        utc_now = dtt.datetime.utcnow().replace(tzinfo=dtt.timezone.utc)
        today = utc_now.astimezone(SHA_TZ)
        offset = dtt.timedelta(minutes=-timesdelay)
        start_time = (today + offset).strftime('%Y-%m-%d %H:%M:%S')
        sql = """select ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon,min(V) as value
        from Tab_AM_M as ta inner join TAB_StationInfo as station on ta.IIIII=station.IIiii and station.Province ='浙江' 
        where (TTime >'{time}' and ta.V!=-9999 ) 
        group by  ta.IIIII,station.StationName,station.Province,station.City,station.County,station.Town,station.ZoomLevel,station.Type,station.lat,station.lon"""
        rsql = sql.format(time=start_time)
        station_all = pd.read_sql(rsql, con=self.conn)
        # 设置redis的键值对儿
        data = {
            "time":start_time,
            "data":station_all
        }
        redis_name_str = self.redis_name[tables_name] + "_view"
        self.rs.set(redis_name_str, pickle.dumps(data))

# 雷达类
class radar_data:
    def __init__(self):
        self.rs = redis.Redis(host='127.0.0.1', port=6379)
    def to_redis(self,imd):
        if self.rs.get("radar"):
            data = pickle.loads(self.rs.get("radar"))
            data['imglist'].append(imd)
        else:
            imglist = [imd]
            data = {
                "imglist":imglist
            }
        self.rs.set("radar", pickle.dumps(data))       
    def plot_data(self):
        path ="static/data/downfile/" 
        f = cinrad.io.CinradReader(path+'Z_RADR_I_Z9576_20150809120400_O_DOR_SA_CAP.bin.bz2')
        tilt_number = 0
        data_radius = 230
        data_dtype = 'REF' # stands for reflectivity
        # 数据加载核心
        ra = f.get_data(tilt_number, data_radius, data_dtype)
        rl = list(f.iter_tilt(230, 'REF'))
        cr = cinrad.easycalc.quick_cr(rl)
        data = ra.data
        m = Basemap(llcrnrlon=119.16,llcrnrlat=26.55,urcrnrlon=123.88,urcrnrlat=30.69)
        ## 颜色 ------
        colorslist = ['#00C800','#019000','#FFFF00','#E7C000','#FF9000','#D60000','#C00000','#FF00F0','#780084','#AD90F0','#AE0AF5']# 组合反射率
        levels1 = [15,20,25,30,35,40,45,50,55,60,65,70]
        cmaps = LinearSegmentedColormap.from_list('mylist',colorslist,N=11)
        plt.contourf(cr.lon, cr.lat, cr.data,cmap=cmaps,levels = levels1)
        plt.axis('off') 
        buffer = BytesIO()
        plt.savefig(buffer,bbox_inches='tight',transparent=True)  
        plot_img = buffer.getvalue()
        imb = base64.b64encode(plot_img) 
        ims = imb.decode()
        imd = "data:image/png;base64,"+ims
        self.to_redis(imd)
    def get_radar(self):
        data = pickle.loads(self.rs.get("radar"))
        return data
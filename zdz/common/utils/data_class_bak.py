from math import isnan

import numpy as np
import pandas as pd


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
        # 测试数据
        print(self.timecounts)

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

    def comput_IIiii(self):
        self.station_dot_comput = {}
        lat = []
        lon = []
        tx = []
        tn = []
        rr = []
        rx = []
        county = []
        name = []
        vv = []
        town = []
        vv_min = " "  # {"name":[],"value":0.0}
        vv_value = 1000.0
        station_RR_small = 0.0
        station_RR_mid = 0.0
        station_RR_big = 0.0
        station_RR_huge = 0.0
        station_RR_bighuge = 0.0
        station_RR_more = 0.0
        station_VV_small = 0.0
        station_VV_mid = 0.0
        station_VV_big = 0.0
        station_VV_huge = 0.0
        station_VV_more = 0.0
        station_wind7 = 0.0
        station_wind8 = 0.0
        station_wind9 = 0.0
        station_wind10 = 0.0
        station_wind11 = 0.0
        station_wind12 = 0.0
        station_wind13 = 0.0
        station_wind14 = 0.0
        station_wind15 = 0.0
        station_wind16 = 0.0
        station_wind17 = 0.0

        station_vv = []
        VV_scatter_list = []
        # 指标站数据
        fFy_name_list = []
        dFy_scater_list = []
        fFy_scater_list = []
        # 温度站点
        temp_event_list = []
        nation_station = ['58660', '58666', 'K8505', 'K8206', '58665', '58559', '58655', 'K8271', '58662', '58653']
        temp_scatter_list = ['58559', 'K8705', 'K8706', '58652', 'K8903', '58568', 'K8818', '58662', 'K8821', '58660',
                             '58653', 'K8609', 'K8505', '58667', '58664', 'K8413', '58655', 'K8282', 'K8217', 'K8201',
                             'K8301', '58665']
        for i in self.grouped_IIiii.size().index:
            data = self.grouped_IIiii.get_group(i)
            # print(data)
            data['VV'].replace(-9999, np.nan, inplace=True)
            data['RR'].replace(-9999, np.nan, inplace=True)
            data['Tn'].replace(-9999, np.nan, inplace=True)
            data['Tx'].replace(-9999, np.nan, inplace=True)
            #             data['fFy'].replace(-9999,np.nan,inplace=True)
            dic = {}
            dic['IIiii'] = data['IIiii'].iloc[0]
            dic['tTime'] = data['tTime'].tolist()
            dic['StationName'] = data['StationName'].iloc[0]
            dic['county'] = data['county'].iloc[0]
            dic['lat'] = data['lat'].iloc[0]
            dic['lon'] = data['lon'].iloc[0]
            dic['Town'] = data['Town'].iloc[0]
            dic['fFy'] = data['fFy'].max()
            dic['dFy'] = data[data['fFy'] == data['fFy'].max()]['dFy'].iloc[0]
            dic['T'] = data['T'].tolist()
            dic['VList'] = data['VV'].tolist()
            dic['fFyList'] = data['fFy'].tolist()
            dic['dFyList'] = data['dFy'].tolist()
            dic['Tx'] = data['Tx'].max() / 10.0

            dic['Tn'] = data['Tn'].min() / 10.0
            if not isnan(data['VV'].min()):
                # 统计能见度自动站名称
                station_vv.append(data['IIiii'].iloc[0])
                if data['VV'].min() < vv_value:
                    #                     print(data['IIiii'].iloc[0])
                    vv_min = str(data['IIiii'].iloc[0])
                    vv_value = data['VV'].min()
            dic['VV'] = data['VV'].min()
            dic['RR'] = data['RR'].sum()
            dic['RMax'] = data['RR'].max()
            # 降水分级别
            if not isnan(data['RR'].sum()):
                value_rsum = data['RR'].sum()
                if value_rsum >= 0 and value_rsum < 10:
                    station_RR_small = station_RR_small + 1
                elif value_rsum >= 10 and value_rsum < 25:
                    station_RR_mid = station_RR_mid + 1
                elif value_rsum >= 25 and value_rsum < 50:
                    station_RR_big = station_RR_big + 1
                elif value_rsum >= 50 and value_rsum < 100:
                    station_RR_huge = station_RR_huge + 1
                elif value_rsum >= 100 and value_rsum < 250:
                    station_RR_RR_bighuge = station_RR_bighuge + 1
                else:
                    station_RR_more = station_RR_more + 1
            # 气温警报分级统计
            if dic['Tx'] > 35.0 or dic['Tn'] < 3.0:
                temp_event_list.append(str(data['IIiii'].iloc[0]))
            # 能见度分级别  
            if not isnan(data['VV'].min()):
                VV_scatter_list.append(data['IIiii'].iloc[0])
                value_VV = data['VV'].min()
                #                 print(value_VV)
                if value_VV >= 0 and value_VV < 50:
                    station_VV_small = station_VV_small + 1
                elif value_VV >= 50 and value_VV < 200:
                    station_VV_mid = station_VV_mid + 1
                elif value_VV >= 200 and value_VV < 500:
                    station_VV_big = station_VV_big + 1
                elif value_VV >= 500 and value_VV < 5000:
                    station_VV_huge = station_VV_huge + 1
                else:
                    station_VV_more = station_VV_more + 1
                    # 大风分级
            if not isnan(data['fFy'].max()):
                if data['fFy'].max() > 138:
                    fFy_scater_list.append(str(data['IIiii'].iloc[0]))
                fFy_data = data['fFy'].max()
                if fFy_data > 13.8 and fFy_data <= 17.1:
                    station_wind7 = station_wind7 + 1
                elif fFy_data > 17.1 and fFy_data <= 20.7:
                    station_wind8 = station_wind8 + 1
                elif fFy_data > 20.7 and fFy_data <= 24.4:
                    station_wind9 = station_wind9 + 1
                elif fFy_data > 24.4 and fFy_data <= 28.4:
                    station_wind10 = station_wind10 + 1
                elif fFy_data > 28.4 and fFy_data <= 32.6:
                    station_wind11 = station_wind11 + 1
                elif fFy_data > 32.6 and fFy_data <= 36.9:
                    station_wind12 = station_wind12 + 1
                elif fFy_data > 36.9 and fFy_data <= 41.4:
                    station_wind13 = station_wind13 + 1
                elif fFy_data > 41.4 and fFy_data <= 46.1:
                    station_wind14 = station_wind14 + 1
                elif fFy_data > 46.1 and fFy_data <= 51.0:
                    station_wind15 = station_wind15 + 1
                elif fFy_data > 51.0 and fFy_data <= 56.1:
                    station_wind16 = station_wind16 + 1
                else:
                    station_wind17 = station_wind17 + 1

            lat.append(data['lat'].iloc[0])
            town.append(data['Town'].iloc[0])
            lon.append(data['lon'].iloc[0])
            county.append(data['county'].iloc[0])
            name.append(data['IIiii'].iloc[0])
            tx.append(data['Tx'].max() / 10.0)
            tn.append(data['Tn'].max() / 10.0)
            rr.append(data['RR'].sum() / 10.0)
            rx.append(data['RR'].max() / 10.0)
            #             print(dic)
            self.station_dot_comput[str(i)] = dic
        #         print(lat)
        # 排序数据
        rain_max = max(rr)
        rain_min = min(rr)
        level_rain = np.linspace(start=rain_min, stop=rain_max, num=9)
        print(level_rain)

        data_rx = pd.DataFrame()
        data_rx['name'] = name
        data_rx['county'] = county
        data_rx['town'] = town
        data_rx['rx'] = rx
        data_rx['lat'] = lat
        data_rx['lon'] = lon
        data_rx['index'] = data_rx['rx'].rank(ascending=0, method='dense')
        data_rr_rx = data_rx.sort_values(by=['rx'], ascending=[False])
        RR_rx = []
        for row in data_rr_rx.itertuples():
            dic_rr = {'index': int(getattr(row, 'index')), 'IIiii': str(getattr(row, 'name')),
                      'county': getattr(row, 'county'), 'town': getattr(row, 'town'), 'data': getattr(row, 'rx'),
                      'value': [getattr(row, 'lon'), getattr(row, 'lat'), getattr(row, 'rx')],
                      'url': "station/" + str(getattr(row, 'name'))}
            RR_rx.append(dic_rr)

        # 按照累计降水进行排序
        data_rsum = pd.DataFrame()
        data_rsum['name'] = name
        data_rsum['county'] = county
        data_rsum['town'] = town
        data_rsum['rsum'] = rr
        data_rsum['lat'] = lat
        data_rsum['lon'] = lon
        data_rr_plot = [lat, lon, rr]
        data_rsum['index'] = data_rsum['rsum'].rank(ascending=0, method='dense')
        data_rr_sum = data_rsum.sort_values(by=['rsum'], ascending=[False])
        RR_sum = []
        for row in data_rr_sum.itertuples():
            dic_rr = {'index': int(getattr(row, 'index')), 'IIiii': str(getattr(row, 'name')),
                      'county': getattr(row, 'county'), 'town': getattr(row, 'town'), 'data': getattr(row, 'rsum'),
                      'value': [getattr(row, 'lon'), getattr(row, 'lat'), getattr(row, 'rsum')],
                      'url': "station/" + str(getattr(row, 'name'))}
            RR_sum.append(dic_rr)
        #         data_rsum['index'] = [a for i in ]
        #         print(data_rx.sort_values(by =['rx'],ascending = [False]))
        # 最大值对应的站点序列
        data_vv = vv_min
        # print("最低能见度",data_vv,vv_min)
        data_vvmin = pd.DataFrame()
        data_vvmin['tTime'] = self.station_dot_comput[data_vv]['tTime']
        data_vvmin['VV'] = self.station_dot_comput[data_vv]['VList']
        # print("data:",data_vvmin)
        # 降水分级
        # RR_station_rank = [
        #     { "value": station_RR_small, "name": '小雨' },
        #     { "value": station_RR_mid, "name": '中雨' },
        #     { "value": station_RR_big, "name": '大雨' },
        #     { "value": station_RR_huge, "name": '暴雨' },
        #     { "value": station_RR_bighuge, "name": '大暴雨' },
        #     { "value": station_RR_more, "name": '特大暴雨' }
        # ]
        RR_station_rank = [station_RR_small, station_RR_mid, station_RR_big, station_RR_huge, station_RR_bighuge,
                           station_RR_more]
        tmp_station_bar = []
        tmp_station_bar.append(['product', '最高气温', '最低气温'])
        RR_station_bar = []
        RR_station_bar.append(['product', '累计降水', '最大雨强'])
        # 计算指标站nation_station的要素值
        for i in nation_station:
            tmp_station_bar.append([self.station_dot_comput[i]['StationName'], self.station_dot_comput[i]['Tx'],
                                    self.station_dot_comput[i]['Tn']])
            RR_station_bar.append([self.station_dot_comput[i]['StationName'], self.station_dot_comput[i]['RR'],
                                   self.station_dot_comput[i]['RMax']])

        # 返回站点气温数据
        tmp_min_scatter = []
        tmp_max_scatter = []
        for i in temp_scatter_list:
            dic_temp_max = {"value": [], "url": "", }
            dic_temp_max['value'].append(self.station_dot_comput[i]['lon'])
            dic_temp_max['value'].append(self.station_dot_comput[i]['lat'])
            dic_temp_max['value'].append(self.station_dot_comput[i]['Tx'])
            dic_temp_max['url'] = "station/" + str(self.station_dot_comput[i]['IIiii'])
            dic_temp_max['label'] = str(self.station_dot_comput[i]['Tx'])
            dic_temp_max['name'] = str(self.station_dot_comput[i]['StationName'])
            dic_temp_min = {"value": [], "url": ""}
            dic_temp_min['value'].append(self.station_dot_comput[i]['lon'])
            dic_temp_min['value'].append(self.station_dot_comput[i]['lat'])
            dic_temp_min['value'].append(self.station_dot_comput[i]['Tn'])
            dic_temp_min['url'] = "station/" + str(self.station_dot_comput[i]['IIiii'])
            dic_temp_min['name'] = str(self.station_dot_comput[i]['StationName'])
            dic_temp_min['label'] = str(self.station_dot_comput[i]['Tn'])
            tmp_max_scatter.append(dic_temp_max)
            tmp_min_scatter.append(dic_temp_min)
        # 返回气温警报数据
        tmp_event_scatter = []
        for i in temp_event_list:
            tmp_event = {"value": [], "url": ""}
            tmp_event['value'].append(self.station_dot_comput[i]['lon'])
            tmp_event['value'].append(self.station_dot_comput[i]['lat'])
            if self.station_dot_comput[i]['Tx'] > 350:
                tmp_event['value'].append(self.station_dot_comput[i]['Tx'])
            elif self.station_dot_comput[i]['Tn'] < 40:
                tmp_event['value'].append(self.station_dot_comput[i]['Tn'])
            tmp_event['url'] = "station/tmp/" + str(self.station_dot_comput[i]['IIiii'])
            tmp_event['name'] = str(self.station_dot_comput[i]['StationName'])
            tmp_event_scatter.append(tmp_event)
            # 返回站点能见度数据
        VV_min_scatter = []
        for i in VV_scatter_list:
            dic_VV_min = {"value": [], "url": ""}
            dic_VV_min['value'].append(self.station_dot_comput[str(i)]['lon'])
            dic_VV_min['value'].append(self.station_dot_comput[str(i)]['lat'])
            dic_VV_min['value'].append(self.station_dot_comput[str(i)]['VV'])
            dic_VV_min['url'] = "station/" + str(self.station_dot_comput[str(i)]['IIiii'])
            dic_VV_min['name'] = str(self.station_dot_comput[str(i)]['StationName'])
            VV_min_scatter.append(dic_VV_min)
            # 能见度分级
        VV_station_rank = [
            {"value": station_VV_small, "name": '强浓雾'},
            {"value": station_VV_mid, "name": '浓雾'},
            {"value": station_VV_big, "name": '大雾'},
            {"value": station_VV_huge, "name": '雾'},
            {"value": station_VV_more, "name": '轻雾'}
        ]
        # 大风分级
        fFy_station_rank = [
            {"value": station_wind7, "name": '7级'},
            {"value": station_wind8, "name": '8级'},
            {"value": station_wind9, "name": '9级'},
            {"value": station_wind10, "name": '10级'},
            {"value": station_wind11, "name": '11级'},
            {"value": station_wind12, "name": '12级'},
            {"value": station_wind13, "name": '13级'},
            {"value": station_wind14, "name": '14级'},
            {"value": station_wind15, "name": '15级'},
            {"value": station_wind16, "name": '16级'},
            {"value": station_wind17, "name": '17级'},
        ]
        # 返回站点级大风数据
        fFy_wind7up_scatter = []
        fFy_name = []
        fFy_county = []
        fFy_town = []
        fFy_value = []
        symbol_ffy = ['path://M10 10L50 10 50 20 20 20 20 40 50 40 50 50 20 50 20 100 10 100 10 10z']
        for i in fFy_scater_list:
            dic_fFy = {"value": [], "url": ""}
            fFy_name.append(self.station_dot_comput[str(i)]['IIiii'])
            fFy_county.append(self.station_dot_comput[str(i)]['county'])
            fFy_town.append(self.station_dot_comput[str(i)]['Town'])
            fFy_value.append(self.station_dot_comput[str(i)]['fFy'])
            dic_fFy['value'].append(self.station_dot_comput[str(i)]['lon'])
            dic_fFy['value'].append(self.station_dot_comput[str(i)]['lat'])
            dic_fFy['value'].append(self.station_dot_comput[str(i)]['fFy'])
            dic_fFy['value'].append(self.station_dot_comput[str(i)]['StationName'])

            dic_fFy['symbol'] = str(symbol_ffy[0])
            dic_fFy['symbolRotate'] = self.station_dot_comput[str(i)]['dFy']
            dic_fFy['url'] = "station/" + str(self.station_dot_comput[str(i)]['IIiii'])
            dic_fFy['name'] = str(self.station_dot_comput[str(i)]['StationName'])
            fFy_wind7up_scatter.append(dic_fFy)
            # 按照级大风进行排序
        data_fFy = pd.DataFrame()
        data_fFy['name'] = fFy_name
        data_fFy['county'] = fFy_county
        data_fFy['town'] = fFy_town
        data_fFy['fFy'] = fFy_value
        data_fFy['index'] = data_fFy['fFy'].rank(ascending=0, method='dense')
        data_fFy_all = data_fFy.sort_values(by=['fFy'], ascending=[False])
        data_fFy_list = []
        for row in data_fFy_all.itertuples():
            dic_ffy = {'index': int(getattr(row, 'index')), 'IIiii': str(getattr(row, 'name')),
                       'county': getattr(row, 'county'), 'town': getattr(row, 'town'), 'data': getattr(row, 'fFy')}
            data_fFy_list.append(dic_ffy)

        max_fFy_station = data_fFy_all['name'].iloc[0]
        data_fFymax = pd.DataFrame()
        data_fFymax['tTime'] = self.station_dot_comput[max_fFy_station]['tTime']
        data_fFymax['fFy'] = self.station_dot_comput[max_fFy_station]['fFyList']
        data_fFymax['dFy'] = self.station_dot_comput[max_fFy_station]['dFyList']
        # print(data_fFy_all,max_fFy_station)
        return data_rr_plot, level_rain, RR_rx, RR_sum, RR_station_rank, RR_station_bar, tmp_station_bar, tmp_min_scatter, tmp_max_scatter, tmp_event_scatter, data_vvmin.sort_values(
            by='tTime'), VV_min_scatter, VV_station_rank, data_fFy_list, fFy_wind7up_scatter

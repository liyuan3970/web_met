"""URL Configuration
"""
from django.urls import path

from . import views

urlpatterns = [
    path('zdz_kuaibao', views.kuaibao),
    path('index_kb', views.index_kb),
    path('post_data', views.post_data),
    path('url_data', views.url_data),
    path('station/<str:station_name>/', views.station_view),
    path('index', views.index_main),
    path('login', views.login_main),
    path('home', views.home),
    path('quick_look', views.quick_look),
    # 自定义绘图
    path('upload_selfplot_data', views.plot_self_data),
    # 订正产品
    path('upload_select_taizhou_data', views.upload_select_taizhou_data),
    # 新建文件
    # 获取创建文件的基本信息
    path('create_new_doc', views.create_new_doc),
    # 像数据库添加文档基本数据
    path('create_doc_data', views.create_new_doc_data),
    # 呈送发的代码
    path('leader_Data_post', views.leader_Data_post),
    # EC站点数据的查询渲染
    path('ec_single_data', views.ec_single_data),

    # 测试url
    path('test', views.test),
    path('error', views.test_error),
]

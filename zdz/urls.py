"""URL Configuration
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter

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
    # 测试模块的页面
    path('test_demo', views.test_demo),
    # 测试页面的
    path('pdf_report', views.pdf_report),
    path('pdf_view', views.pdf_view),
    # 自定义绘图
    path('upload_selfplot_data', views.plot_self_data),
    # 自定义画图数据下载
    path('self_plot_download', views.self_plot_download),
    # 订正产品
    path('upload_select_taizhou_data', views.upload_select_taizhou_data),
    # 新建文件
    # 获取创建文件的基本信息
    path('create_new_doc', views.create_new_doc),
    # 像数据库添加文档基本数据
    path('create_doc_data', views.create_new_doc_data),
    # 呈送发的代码
    path('leader_Data_post', views.leader_Data_post),
    # 自动站历史数据的查询交互
    path('tool_zdz_date', views.tool_zdz_date),
    # 自动站历史数据大风的查询
    path('tool_zdz_wind', views.tool_zdz_wind),
    # 自动站历史数据能见度的查询
    path('tool_zdz_view', views.tool_zdz_view),
    # 自动站历史数据气温的查询
    path('tool_zdz_temp', views.tool_zdz_temp),
    # 自动站日报daily 的数据查询
    path('tool_zdz_daily', views.tool_zdz_daily),
    # EC站点数据的查询渲染
    path('ec_single_data', views.ec_single_data),
]

router = DefaultRouter()
router.register(r"user", views.UserViewSet)
urlpatterns += [
    path("", include(router.urls)),
    path(r"file/pdf", views.pdf)
]

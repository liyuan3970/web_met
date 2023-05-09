"""URL Configuration
"""
from django.urls import path, include, re_path
from django.views.static import serve

from . import views
from .middleware import StandardRouter

urlpatterns = [
    path('zdz_kuaibao', views.kuaibao),
    path('index_kb', views.index_kb),
    path('post_data', views.post_data),
    path('url_data', views.url_data),
    path('station/<str:station_name>/', views.station_view),
    path('index', views.index_main),
    path('login', views.login_main),
    path('home/', views.home), 
    path('quick_look', views.quick_look),
    # 可以直接预览媒体文件
    re_path('^media/(?P<path>.*?)$', serve, kwargs={'document_root': 'media/'}),
    # canvas画图的功能
    path('canvas_plot', views.canvas_plot),
    # 实况监测的页面
    path('station_zdz', views.station_zdz),
    # 网址功能
    path('website', views.website),
    # 历史文档查阅功能history_file
    path('history_file', views.history_file),
    # tinymce插件footer
    path('tinymce/footer', views.tinymce_footer),
    # tinymce插件advicemet
    path('tinymce/advicemet', views.tinymce_advicemet),
    # tinymce插件selfmode
    path('tinymce/selfmode', views.tinymce_selfmode),
    # 十天数据的获取
    path('tinymce/longmet', views.longmet),
    # 短期数据的获取
    path('tinymce/shortmet', views.shortmet),
    # 自定义绘图
    path('upload_selfplot_data', views.plot_self_data),
    # 自定义画图数据下载
    path('self_plot_download', views.self_plot_download),
    # 订正产品
    path('upload_select_taizhou_data', views.upload_select_taizhou_data),
    # 订正时次
    path('upload_select_taizhou_data/item', views.upload_select_taizhou_data),
    # 打开文件（查询select的选项）
    path('open_old_doc', views.open_old_doc),
    # 打开所选类型的文档并展开列表
    path('open_doc_data', views.open_doc_data),
    # 下载选取文件的数据
    path('open_load_object', views.open_load_object),
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
    # 自订正降水的查询
    path('select_self_plot', views.select_self_plot),
    # 以下为station的链接
    path('station_zdz_data', views.station_zdz_data),

]

router = StandardRouter()
router.register(r"user", views.UserViewSet)
router.register(r"file", views.FileViewSet, basename="file")
urlpatterns += [
    path("", include(router.urls))
]

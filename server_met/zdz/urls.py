"""URL Configuration
"""
from django.urls import path, include, re_path
from django.views.static import serve

from . import views
from .middleware import StandardRouter

urlpatterns = [ 
    path('login', views.login_main),
    # 可以直接预览媒体文件
    re_path('^media/(?P<path>.*?)$', serve, kwargs={'document_root': 'media/'}),
    # 首页
    path('index', views.station_zdz),
    # 数据交互的所有接口
    path('station_zdz_data', views.station_zdz_data),
    # 报警的所有接口
    path('station_zdz_warring', views.station_zdz_warring),

]

router = StandardRouter()
router.register(r"user", views.UserViewSet)
router.register(r"file", views.FileViewSet, basename="file")
urlpatterns += [
    path("", include(router.urls))
]

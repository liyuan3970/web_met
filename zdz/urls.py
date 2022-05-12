"""tzdemo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'zdz_kuaibao',views.kuaibao),

    url(r'index_kb',views.index_kb),
    url(r'post_data',views.post_data),
    url(r'url_data',views.url_data),
    url(r'^station/(.+)/$',views.station_view),
    url(r'^index',views.index_main),
    url(r'^login',views.login_main),
    url(r'^home',views.home),
    url(r'^quick_look',views.quick_look),
    # 自定义绘图
    url(r'^upload_selfplot_data',views.plot_self_data),
    # 订正产品
    url(r'^upload_select_taizhou_data',views.upload_select_taizhou_data),
    # 新建文件
    # 获取创建文件的基本信息
    url(r'^create_new_doc',views.create_new_doc),
    # 像数据库添加文档基本数据
    url(r'^create_doc_data',views.create_new_doc_data),

]

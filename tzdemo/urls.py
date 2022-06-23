"""URL Configuration
"""
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views import static

urlpatterns = [
    # debug=false路由静态文件
    re_path(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'),
    path('admin', admin.site.urls),
    path('', include('zdz.urls')),
]

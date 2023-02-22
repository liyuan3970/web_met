"""URL Configuration
"""
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views import static

from zdz.views.user import LoginView, LoginRefreshView

urlpatterns = [
    path('admin', admin.site.urls),
    # rest_framework_simplejwt自带的得到token
    path('user/login', LoginView.as_view()),
    # 刷新jwt
    path('user/login/refresh', LoginRefreshView.as_view()),
    path('', include('zdz.urls')),
]

# debug=false路由静态文件
if not settings.DEBUG:
    urlpatterns.append(
        re_path(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'))

"""URL Configuration
"""
from django.conf import settings
from django.contrib import admin
from django.urls import include, path, re_path
from django.views import static
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    # debug=false路由静态文件
    re_path(r'^static/(?P<path>.*)$', static.serve, {'document_root': settings.STATIC_ROOT}, name='static'),
    path('admin', admin.site.urls),
    # rest_framework_simplejwt自带的得到token
    path('user/token/', TokenObtainPairView.as_view()),
    # 刷新jwt
    path('user/token/refresh/', TokenRefreshView.as_view()),
    path('', include('zdz.urls')),
]

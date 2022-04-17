from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$',views.index_views),
    url(r'^login/$',views.login_views),
    url(r'^register/$',views.register_views),
]

urlpatterns += [
    url(r'^01-register/$',views.register01_views),
]
from django.contrib import admin

from .models import *

# Register your models here.
admin.site.register(Document)
admin.site.register(Unity)
admin.site.register(Publisher)
admin.site.register(DocumentType)
admin.site.register(Picture)
admin.site.register(LeaderData)
admin.site.register(Writer)
admin.site.register(SelfModule)
admin.site.register(SelfPlot)
# 网站相关
admin.site.register(WebPicture)
admin.site.register(WebClass)
admin.site.register(WebUnity)
admin.site.register(WebCache)

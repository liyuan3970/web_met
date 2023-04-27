from django.contrib import admin

from .models import *

# Register your models here.
admin.site.register(Document)
admin.site.register(Unity)
admin.site.register(Publisher)
admin.site.register(DocumentType)
admin.site.register(LeaderData)
admin.site.register(Writer)
admin.site.register(SelfModule)

# 网站相关
admin.site.register(WebPicture)
admin.site.register(WebClass)
admin.site.register(DocumentSelfDefine)
# 数据相关
admin.site.register(EcData)
admin.site.register(SelfPlot)
admin.site.register(DocumentAdvice)
# 用户
admin.site.register(User)
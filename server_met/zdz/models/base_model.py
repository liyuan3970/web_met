from django.db import models


class BaseModel(models.Model):
    """抽象基类"""
    id = models.BigAutoField(primary_key=True, verbose_name="主键")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    create_user = models.BigIntegerField(verbose_name="创建用户")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    update_user = models.BigIntegerField(verbose_name="更新用户")

    class Meta:
        app_label = "zdz"
        abstract = True

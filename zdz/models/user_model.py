from django.db import models

from .base_model import BaseModel


class UserModel(BaseModel):
    """用户表"""
    id = models.BigAutoField(primary_key=True, verbose_name="主键")
    username = models.CharField(max_length=50, verbose_name="用户名")
    password = models.CharField(max_length=50, verbose_name="密码")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")

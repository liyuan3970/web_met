from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """用户表"""

    # 删除字段
    # email = None
    # first_name = None
    # last_name = None
    # date_joined = None

    # 新建字段
    id = models.BigAutoField(primary_key=True, verbose_name="主键")
    username = models.CharField(unique=True, max_length=50, verbose_name="用户名")
    password = models.CharField(max_length=255, verbose_name="密码")
    name = models.CharField(max_length=50, verbose_name="姓名")
    role_type = models.IntegerField(default=0, verbose_name="1、领导 2、高工 3、预报员")
    company_name = models.CharField(max_length=50, verbose_name="单位名称")
    company_type = models.IntegerField(default=1, verbose_name="1、省 2、县 3、市")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    update_time = models.DateTimeField(auto_now_add=True, verbose_name="更新时间")

    class Meta:
        app_label = "zdz"
        verbose_name = verbose_name_plural = '用户表'
        db_table = 'user'  # 通过db_table自定义数据表名

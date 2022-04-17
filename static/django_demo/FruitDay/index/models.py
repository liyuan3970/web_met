from django.db import models

# Create your models here.
class Users(models.Model):
    uphone = models.CharField(max_length=11)
    upwd = models.CharField(max_length=30)
    uname = models.CharField(max_length=30)
    uemail = models.EmailField()

#创建商品类型类
class GoodsType(models.Model):
  title = models.CharField(
    max_length=40,
    verbose_name='标题')

  picture = models.ImageField(
    upload_to='static/upload/goodstype',
    verbose_name='类型图片')

  desc = models.TextField(verbose_name='类型描述')

  isActive = models.BooleanField(
    default=True,verbose_name='是否激活')

  class Meta:
    db_table = 'goodsType'
    verbose_name = '商品类型'
    verbose_name_plural = verbose_name

class Goods(models.Model):
  title = models.CharField(
    max_length=80,verbose_name='商品名称')

  price = models.DecimalField(max_digits=7,decimal_places=2,verbose_name='商品价格')

  spec = models.CharField(max_length=20,verbose_name='商品规格')

  picture = models.ImageField(upload_to='static/upload/goods',null=True,verbose_name='商品图片')

  isActive = models.BooleanField(default=True,verbose_name='是否上架')

  class Meta:
      db_table = 'goods'
      verbose_name = '商品'
      verbose_name_plural = verbose_name









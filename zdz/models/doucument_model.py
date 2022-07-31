from django.db import models

from .base_model import BaseModel


class Document(BaseModel):
    """文档字段"""
    types = models.TextField(max_length=200, verbose_name='类别')
    item = models.IntegerField(verbose_name='期数')
    year = models.IntegerField(verbose_name='年份')
    unity = models.CharField(max_length=200, verbose_name='单位')
    pub_date = models.CharField(max_length=200, verbose_name='发布日期')
    verson_content = models.TextField(
        max_length=21844, verbose_name='版本内容')  # 存放字典用来记录版本信息
    writer = models.CharField(max_length=50, verbose_name='撰稿人')
    publisher = models.CharField(
        max_length=200, verbose_name='签发人', default="翁之梅")

    class Meta:
        verbose_name = verbose_name_plural = '文档'
        db_table = 'document'  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.year) + "年" + self.types + ":第" + str(self.item) + "期"


class SelfDefine(BaseModel):
    types = models.CharField(max_length=200, verbose_name='类别')
    name = models.CharField(max_length=200, verbose_name='名称')
    data = models.JSONField(null=False)

    class Meta:
        verbose_name = verbose_name_plural = '预定义类型'
        db_table = 'selfmode'  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.type) + ":" + self.name


class Unity(BaseModel):
    """单位名称"""
    name = models.CharField(max_length=200, verbose_name='单位')

    class Meta:
        verbose_name = verbose_name_plural = '单位'
        db_table = 'unity'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Publisher(BaseModel):
    """签发人"""
    name = models.CharField(max_length=200, verbose_name='签发人')

    class Meta:
        verbose_name = verbose_name_plural = '签发人'
        db_table = 'publisher'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Writer(BaseModel):
    """撰稿人"""
    name = models.CharField(max_length=200, verbose_name='撰稿人')

    class Meta:
        verbose_name = verbose_name_plural = '撰稿人'
        db_table = 'writer'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class DocumentType(BaseModel):
    """文档类别"""
    name = models.CharField(max_length=200, verbose_name='材料类型')

    class Meta:
        verbose_name = verbose_name_plural = '材料类型'
        db_table = 'documenttype'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Picture(BaseModel):
    """图片二维码"""
    img = models.ImageField(upload_to='img/')
    name = models.CharField(max_length=200, verbose_name='宣传二维码')

    class Meta:
        verbose_name = verbose_name_plural = '宣传二维码'
        db_table = 'picture'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class LeaderData(BaseModel):
    """呈发送"""
    name = models.CharField(max_length=20, verbose_name='发布版本')
    picture_list = models.CharField(
        max_length=20, verbose_name='二维码种类')  # 存放二维码的字典
    service_name = models.TextField(max_length=21844, verbose_name='呈报对象')
    service_unity = models.TextField(max_length=21844, verbose_name='发布单位')
    recive_unity = models.TextField(max_length=21844, verbose_name='抄送单位')

    class Meta:
        verbose_name = verbose_name_plural = '呈送发'
        db_table = 'leader_data'  # 通过db_table自定义数据表名

    def __str__(self):
        return "呈送发:" + self.name

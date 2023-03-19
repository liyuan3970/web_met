from django.db import models

from .base_model import BaseModel


class Document(BaseModel):
    """文档字段"""
    document_type = models.TextField(max_length=200, verbose_name="材料类别")
    item = models.IntegerField(verbose_name="期数")
    year = models.IntegerField(verbose_name="年份")
    unity = models.CharField(max_length=200, verbose_name="单位")
    pub_date = models.CharField(max_length=200, verbose_name="发布日期")
    version_content = models.TextField(
        max_length=21844, verbose_name="模块列表")  # 存放字典用来记录版本信息
    writer = models.CharField(max_length=50, verbose_name="撰稿人")
    publisher = models.CharField(
        max_length=200, verbose_name="签发人", default="翁之梅")

    class Meta:
        verbose_name = verbose_name_plural = "文档"
        db_table = "document"  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.year) + "年" + self.document_type + ":第" + str(self.item) + "期"


class SelfModule(BaseModel):
    """存放模块内容和更新"""
    document_type = models.CharField(max_length=200, verbose_name="材料类别")
    name = models.CharField(max_length=200, verbose_name="模块名称")
    item = models.IntegerField(verbose_name="期数")
    year = models.IntegerField(verbose_name="年份")
    unity = models.CharField(max_length=200, verbose_name="单位")
    data = models.JSONField(null=False)

    class Meta:
        verbose_name = verbose_name_plural = "个人模块"
        db_table = "self_module"  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.document_type) + ":" + self.name


class SelfPlot(BaseModel):
    """存放自定义画图的数据"""
    document_type = models.CharField(max_length=200, verbose_name="材料类别")
    time = models.CharField(max_length=200, verbose_name="时间")
    data = models.JSONField(null=False)

    class Meta:
        verbose_name = verbose_name_plural = "自定义画图"
        db_table = "self_plot"  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.document_type)


class Unity(BaseModel):
    """单位名称"""
    name = models.CharField(max_length=200, verbose_name="单位")

    class Meta:
        verbose_name = verbose_name_plural = "单位"
        db_table = "unity"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Publisher(BaseModel):
    """签发人"""
    name = models.CharField(max_length=200, verbose_name="签发人")
    unity = models.CharField(max_length=200, verbose_name="单位")

    class Meta:
        verbose_name = verbose_name_plural = "签发人"
        db_table = "publisher"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Writer(BaseModel):
    """撰稿人"""
    name = models.CharField(max_length=200, verbose_name="撰稿人")
    unity = models.CharField(max_length=200, verbose_name="单位")

    class Meta:
        verbose_name = verbose_name_plural = "撰稿人"
        db_table = "writer"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class DocumentType(BaseModel):
    """文档类别"""
    name = models.CharField(max_length=200, verbose_name="材料类型")
    unity = models.CharField(max_length=200, verbose_name="单位")

    class Meta:
        verbose_name = verbose_name_plural = "材料类型"
        db_table = "document_type"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Picture(BaseModel):
    """图片二维码"""
    img = models.ImageField(upload_to="img/")
    name = models.CharField(max_length=200, verbose_name="宣传二维码")

    class Meta:
        verbose_name = verbose_name_plural = "宣传二维码"
        db_table = "picture"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


# 网站交互的图片资源
class WebPicture(BaseModel):
    """网站图片"""
    img = models.ImageField(upload_to="web/")
    name = models.CharField(max_length=200, verbose_name="网站名称")
    url = models.CharField(max_length=200, verbose_name="网站地址")
    unity = models.TextField(max_length=21844, verbose_name="单位")
    web_class = models.TextField(max_length=21844, verbose_name="类别")

    class Meta:
        verbose_name = verbose_name_plural = "网站图片"
        db_table = "web_picture"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class WebClass(BaseModel):
    """网站分类"""
    name = models.CharField(max_length=200, verbose_name="类别")
    unity = models.TextField(max_length=21844, verbose_name="单位")

    class Meta:
        verbose_name = verbose_name_plural = "网站分类"
        db_table = "web_class"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class WebCache(BaseModel):
    """交互图片"""
    name = models.CharField(max_length=200, verbose_name="图片名称")
    unity = models.CharField(max_length=200, verbose_name="单位")
    data = models.JSONField(null=False)

    class Meta:
        verbose_name = verbose_name_plural = "缓存图片"
        db_table = "web_cache"  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


# 网站交互的图片资源
class LeaderData(BaseModel):
    """呈发送"""
    name = models.CharField(max_length=20, verbose_name="呈送发对象")
    picture_list = models.CharField(
        max_length=20, verbose_name="二维码种类")  # 存放二维码的字典
    service_name = models.TextField(max_length=21844, verbose_name="呈报对象")
    service_unity = models.TextField(max_length=21844, verbose_name="发布单位")
    receive_unity = models.TextField(max_length=21844, verbose_name="抄送单位")

    class Meta:
        verbose_name = verbose_name_plural = "呈送发"
        db_table = "leader_data"  # 通过db_table自定义数据表名

    def __str__(self):
        return "呈送发:" + self.name

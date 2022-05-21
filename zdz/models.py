from django.db import models

# Create your models here.


class BaseModel(models.Model):
    """抽象基类

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    id = models.BigAutoField(primary_key=True, verbose_name="主键")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    create_user = models.BigIntegerField(verbose_name="创建用户")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")
    update_user = models.BigIntegerField(verbose_name="更新用户")

    class Meta:
        abstract = True


class Document(BaseModel):
    """文档字段

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    types = models.CharField(max_length=50, verbose_name='类别')
    item = models.IntegerField(verbose_name='期数')
    year = models.IntegerField(verbose_name='年份')
    unity = models.CharField(max_length=50, verbose_name='单位')
    pub_date = models.CharField(max_length=50, verbose_name='发布日期')
    verson_content = models.TextField(
        max_length=21844, verbose_name='版本内容')  # 存放字典用来记录版本信息
    writer = models.CharField(max_length=50, verbose_name='撰稿人')
    publisher = models.CharField(
        max_length=50, verbose_name='签发人', default="翁之梅")

    class Meta:
        verbose_name = verbose_name_plural = '文档'
        db_table = 'document'  # 通过db_table自定义数据表名

    def __str__(self):
        return str(self.year)+"年"+self.types + ":第"+str(self.item)+"期"


class Unity(BaseModel):
    """单位名称

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = models.CharField(max_length=50, verbose_name='单位')

    class Meta:
        verbose_name = verbose_name_plural = '单位'
        db_table = 'unity'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Publisher(BaseModel):
    """签发人

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = models.CharField(max_length=50, verbose_name='签发人')

    class Meta:
        verbose_name = verbose_name_plural = '签发人'
        db_table = 'publisher'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Writer(BaseModel):
    """撰稿人

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = models.CharField(max_length=50, verbose_name='撰稿人')

    class Meta:
        verbose_name = verbose_name_plural = '撰稿人'
        db_table = 'writer'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class DocumentType(BaseModel):
    """文档类别

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = models.CharField(max_length=50, verbose_name='材料类型')

    class Meta:
        verbose_name = verbose_name_plural = '材料类型'
        db_table = 'documenttype'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Picture(BaseModel):
    """图片二维码

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    img = models.ImageField(upload_to='img/')
    name = models.CharField(max_length=20, verbose_name='宣传二维码')

    class Meta:
        verbose_name = verbose_name_plural = '宣传二维码'
        db_table = 'picture'  # 通过db_table自定义数据表名

    def __str__(self):
        return self.name


class Leader_Data(BaseModel):
    """呈发送

    Args:
        models (_type_): _description_

    Returns:
        _type_: _description_
    """
    name = models.CharField(max_length=20, verbose_name='发布版本')
    picture_list = models.CharField(
        max_length=20, verbose_name='二维码种类')  # 存放二维码的字典
    service_name = models.CharField(max_length=20, verbose_name='呈报对象')
    service_unity = models.CharField(max_length=20, verbose_name='发布单位')
    recive_unity = models.CharField(max_length=20, verbose_name='抄送单位')

    class Meta:
        verbose_name = verbose_name_plural = '呈送发'
        db_table = 'leader_data'  # 通过db_table自定义数据表名

    def __str__(self):
        return "呈送发:"+self.name

# # 以下是文档模块的字段，与文档存在关系映射
# # 文字内容 呈送发
# class Text_Data(models.Model):
#     name = models.CharField(primary_key=True,max_length=20,verbose_name='发布版本')
#     picture_list = models.CharField(max_length=20,verbose_name='二维码种类')
#     service_name = models.CharField(max_length=20,verbose_name='呈报对象')
#     service_unity = models.CharField(max_length=20,verbose_name='发布单位')
#     recive_unity = models.CharField(max_length=20,verbose_name='抄送单位')
#     class Meta:
#         verbose_name = verbose_name_plural = '发布版本'
#         db_table = 'leader_data' # 通过db_table自定义数据表名
#     # 定义后台列表中的字段名称(self是对应字段的内同)
#     def __str__(self):
#         return self.name

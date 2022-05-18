from django.db import models

# Create your models here.
# 1 文档字段
class Document(models.Model):
    id = models.AutoField(primary_key=True)
    types = models.CharField(max_length=50,verbose_name='类别')
    item = models.IntegerField(verbose_name='期数')
    year = models.IntegerField(verbose_name='年份')
    unity = models.CharField(max_length=50,verbose_name='单位')
    pub_date = models.CharField(max_length=50,verbose_name='发布日期')
    verson_content = models.TextField(max_length=21844,verbose_name='版本内容')# 存放字典用来记录版本信息
    writer = models.CharField(max_length=50,verbose_name='撰稿人')
    publisher = models.CharField(max_length=50,verbose_name='签发人',default="翁之梅")
    class Meta:
        verbose_name = verbose_name_plural = '文档'
        db_table = 'document' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return str(self.year)+"年"+self.types + ":第"+str(self.item)+"期"
# 2 单位名称
class Unity(models.Model):
    name = models.CharField(primary_key=True,max_length=50,verbose_name='单位')
    class Meta:
        verbose_name = verbose_name_plural = '单位'
        db_table = 'unity' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return self.name
# 3 签发人
class Publisher(models.Model):
    name = models.CharField(primary_key=True,max_length=50,verbose_name='签发人')
    class Meta:
        verbose_name = verbose_name_plural = '签发人'
        db_table = 'publisher' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return self.name
# 4 撰稿人
class Writer(models.Model):
    name = models.CharField(primary_key=True,max_length=50,verbose_name='撰稿人')
    class Meta:
        verbose_name = verbose_name_plural = '撰稿人'
        db_table = 'writer' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return self.name

# 5 文档类别
class DocumentType(models.Model):
    name = models.CharField(primary_key=True,max_length=50,verbose_name='材料类型')
    class Meta:
        verbose_name = verbose_name_plural = '材料类型'
        db_table = 'documenttype' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return self.name

# 6 图片二维码
class Picture(models.Model):
    img = models.ImageField(upload_to='img/')
    name = models.CharField(primary_key=True,max_length=20,verbose_name='宣传二维码')
    class Meta:
        verbose_name = verbose_name_plural = '宣传二维码'
        db_table = 'picture' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
    def __str__(self):
        return self.name

# 7 呈送发
class Leader_Data(models.Model):
    name = models.CharField(primary_key=True,max_length=20,verbose_name='发布版本')
    picture_list = models.CharField(max_length=20,verbose_name='二维码种类') # 存放二维码的字典
    service_name = models.CharField(max_length=20,verbose_name='呈报对象') 
    service_unity = models.CharField(max_length=20,verbose_name='发布单位')
    recive_unity = models.CharField(max_length=20,verbose_name='抄送单位')    
    class Meta:
        verbose_name = verbose_name_plural = '呈送发'
        db_table = 'leader_data' # 通过db_table自定义数据表名
    # 定义后台列表中的字段名称(self是对应字段的内同)
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







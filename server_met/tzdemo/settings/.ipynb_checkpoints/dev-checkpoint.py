from .base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
# DEBUG = True
# Database
# https://docs.djangoproject.com/en/1.11/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        # 数据库名称
        'NAME': 'tzweb',
        'USER': 'root',
        'PASSWORD': '051219',
        # 'HOST': '192.168.192.5',
        'HOST': '127.0.0.1',
        'PORT': '3306',
        'OPTIONS': {
            # 外键约束
            "init_command": "SET foreign_key_checks = 0;",

        }
    }
}

CACHES = {
    "default": {
        "BACKEND": "django_redis.cache.RedisCache",
        "LOCATION": "redis://192.168.192.5:6379/0",    # /0：代表redis的0号数据库
        # "LOCATION": "redis://127.0.0.1:6379/0",    # /0：代表redis的0号数据库 
        "OPTIONS": {
            "CLIENT_CLASS": "django_redis.client.DefaultClient",
            "PASSWORD": "lq9394"
        }
    }
}

#  配置定时完成任务的程序
# 解决中文乱码问题
CRONTAB_COMMAND_PREFIX = 'LANG_ALL=zh_cn.UTF-8'
# 存放log的路径 绝对路径
CRONJOBS_DIR = "/workspace/liyuan3970/Data/My_Git/web_met/static/data/log/"
# Log文件名
CRONJOBS_FILE_NAME = "contrab_data.log"
# 添加定时任务(函数中的输出语句,是输出在.log文件中的)
CRONJOBS = (
	# 每分钟执行一次zdz App中func的contrab_data函数，执行后将打印结果存储在log文件中
	#  '2>&1'每项工作执行后要做的事
    ('*/1 * * * *', 'zdz.func.contrab_data',  '>>'+CRONJOBS_DIR+CRONJOBS_FILE_NAME + ' 2>&1'), # 每分钟执行一次
    # ('00 11 * * *', 'TestCrontab.crontabFun.timedExecution',  '>>'+CRONJOBS_DIR+CRONJOBS_FILE_NAME + ' 2>&1'), # 每天11点执行
    # ('0 */1 * * *', 'TestCrontab.crontabFun.timedExecution',  '>>'+CRONJOBS_DIR+CRONJOBS_FILE_NAME + ' 2>&1'), # 每小时执行一次
    # * * * * *
   	# 分钟(0-59) 小时(0-23) 每个月的哪一天(1-31) 月份(1-12) 周几(0-6)

)


# CACHES = {
#     "default": {
#         "BACKEND": "django_redis.cache.RedisCache",
#         "LOCATION": "redis://192.168.192.5:6379/0",
#         "OPTIONS": {
#             "CLIENT_CLASS": "django_redis.client.DefaultClient",
#             "PASSWORD": ""
#         }
#     }
# }

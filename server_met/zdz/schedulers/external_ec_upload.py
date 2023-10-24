import datetime
import os

from django.db import connection

from tzdemo.settings import base
from zdz.common.utils import db_utils

from zdz.common.utils import data_class

if not os.environ.get("DJANGO_SETTINGS_MODULE"):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE",
                          "tzdemo.settings.%s" % base.ENV)


def ec_upload():
    # 开始
    ec_worker = data_class.ec_data_upload()
    ec_worker.conuty_data()



if __name__ == '__main__':
    ec_upload()

from rest_framework import status
from rest_framework.response import Response


class APIResponse(Response):

    def __init__(self, data=None):
        ret = {
            "code": status.HTTP_200_OK,
            "msg": "success"
        }
        if data:
            ret["data"] = data
        super().__init__(data=ret)

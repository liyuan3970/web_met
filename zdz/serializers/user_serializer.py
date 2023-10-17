from django.contrib.auth.hashers import make_password
from django.utils.translation import gettext_lazy as _
from rest_framework import serializers, status
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer, TokenRefreshSerializer
from ..models import User
import json
class RegisterSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            "username",
            "password",
            "name"
        ]
    def validate(self, attrs):
        # todo 参数校验
        # 密码加密
        attrs["password"] = make_password(attrs["password"])
        return attrs

class LoginSerializer(TokenObtainPairSerializer):
    
    default_error_messages = {
        "no_active_account": _("No active account found with the given credentials")
    }
    def validate(self, attrs):
        data = super().validate(attrs)
        refresh = self.get_token(self.user)
        # print("---------------返回对应的地形数据:---------",self.user.name)
        geofile = self.user.name + ".json"
        filename = "static/shps/" + geofile
        with open(filename, "r") as file:
            shp = json.load(file)
        data["refresh"] = str(refresh)
        data["access"] = str(refresh.access_token)
        data["user_id"] = self.user.id
        data["name"] = self.user.name
        data["role_type"] = self.user.role_type
        data["company_type"] = self.user.company_type
        data["company_name"] = self.user.company_name 
        data["shp"] = shp       
        ret = {
            "code": status.HTTP_200_OK,
            "msg": "success",
            "data": data
        }
        return ret


class LoginRefreshSerializer(TokenRefreshSerializer):
    default_error_messages = {
        "no_active_account": _("No active account found with the given credentials")
    }

    def validate(self, attrs):
        data = super().validate(attrs)
        ret = {
            "code": status.HTTP_200_OK,
            "msg": "success",
            "data": data
        }
        return ret



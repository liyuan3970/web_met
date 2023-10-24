from django.db import transaction
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

from ..common.utils.http_utils import APIResponse
from ..models import User
from ..serializers import RegisterSerializer, LoginSerializer, LoginRefreshSerializer


# 登录view
class LoginView(TokenObtainPairView):
    # print("a")
    serializer_class = LoginSerializer


class LoginRefreshView(TokenRefreshView):
    
    serializer_class = LoginRefreshSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    
    @transaction.atomic()
    @action(methods=["post"], url_path="register", detail=False, permission_classes=[permissions.AllowAny])
    def register(self, request):
        reg_ser = RegisterSerializer(data=request.data)
        reg_ser.is_valid(raise_exception=True)
        reg_ser.save()

        refresh = RefreshToken.for_user(reg_ser.instance)

        data = dict()
        data["access"] = str(refresh.access_token)
        data["refresh"] = str(refresh)
        data["user_id"] = reg_ser.instance.id
        data["name"] = reg_ser.instance.name

        data["role_type"] = reg_ser.instance.role_type
        data["company_type"] = reg_ser.instance.company_type
        data["company_name"] = reg_ser.instance.company_name

        return APIResponse(data)

from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from ..models import UserModel
from ..serializers import RegisterSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()

    @action(methods=["post"], url_path="login", detail=False)
    def login(self, request):
        return "login ok"

    @action(methods=["post"], url_path="register", detail=False)
    def register(self, request):
        reg_ser = RegisterSerializer(data=request.data)
        reg_ser.is_valid(raise_exception=True)
        reg_ser.save()
        reg_ser.instance()
        return Response(reg_ser.data)

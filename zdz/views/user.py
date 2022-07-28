from django.db import transaction
from rest_framework import viewsets, permissions
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken

from ..models import UserModel
from ..serializers import RegisterSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()

    @transaction.atomic()
    @action(methods=["post"], url_path="register", detail=False, permission_classes=[permissions.AllowAny])
    def register(self, request):
        reg_ser = RegisterSerializer(data=request.data)
        reg_ser.is_valid(raise_exception=True)
        reg_ser.save()

        print(reg_ser.instance)

        refresh = RefreshToken.for_user(reg_ser.instance)

        print(refresh.access_token)
        return Response(str(refresh))

    @action(methods=["post"], url_path="test", detail=False)
    def test(self, request):
        return Response("ok")

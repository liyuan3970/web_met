from rest_framework import viewsets
from rest_framework.decorators import action

from ..models import UserModel
from ..serializers import UserSerializer


class UserViewSet(viewsets.ModelViewSet):
    queryset = UserModel.objects.all()
    serializer_class = UserSerializer

    @action(methods=["post"], url_path="login", detail=False)
    def login(self, request):
        return "login ok"

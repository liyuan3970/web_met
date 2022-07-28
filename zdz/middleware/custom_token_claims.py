from rest_framework_simplejwt.serializers import TokenObtainPairSerializer


class CustomTokenClaims(TokenObtainPairSerializer):

    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)

        # Add custom claims
        token['name'] = user.name
        # ...

        return token

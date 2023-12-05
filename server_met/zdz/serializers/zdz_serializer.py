from rest_framework import serializers

class ZdzSerializer(serializers.Serializer):
    """
    定义接收字段，括号内可写约束条件
    """
    model = serializers.CharField()
    name = serializers.CharField()
    def validate(self, attrs):
        print("正在检验")
        data = {
            "model":"zdz",
            "name":"liyuan"
        }
        return data


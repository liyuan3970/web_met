from rest_framework import serializers


class DocumentSerializer(serializers.Serializer):
    """
    定义接收字段，括号内可写约束条件
    """
    document = serializers.CharField()
    data = serializers.ListField()
    type = serializers.CharField()
    name = serializers.ListField(allow_empty=False)
    year = serializers.IntegerField(min_value=0)
    item = serializers.IntegerField(min_value=0)
    unity = serializers.CharField()

    def create(self, validated_data):
        """
        保存数据写在这里
        return **Model.objects.create(validated_data)
        """
        return True

    def update(self, instance, validated_data):
        """
        更新数据写在这里
        instance.**arg = validated_data.get('**arg', instance.**arg)
        ...
        instance.save()
        """
        return instance

from rest_framework import serializers
from ..models import SelfDefine

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
        data = validated_data["data"]
        name = validated_data["name"]
        item = validated_data["item"]
        year = validated_data["year"]
        unity = validated_data["unity"]
        type = validated_data["type"]
        for i in range(len(data)):
            context = {
                "id": i,
                "data": data[i]
            }
            obj = SelfDefine.objects.create(
                types=type,
                name=name[i],
                item=item,
                year=year,
                unity=unity,
                data=context,
                create_user=0,
                update_user=0
            )
        return True

    def update(self, instance, validated_data):
        """
        更新数据写在这里
        instance.**arg = validated_data.get('**arg', instance.**arg)
        ...
        instance.save()
        """
        return instance

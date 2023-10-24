from rest_framework import serializers

from ..models import SelfModule


class PreviewAllSerializer(serializers.Serializer):
    """
    定义接收字段，括号内可写约束条件
    """
    document = serializers.CharField()
    data = serializers.ListField()
    document_type = serializers.CharField()
    name = serializers.ListField(allow_empty=False)
    year = serializers.IntegerField(min_value=0)
    item = serializers.IntegerField(min_value=0)
    unity = serializers.CharField()

    def create(self, validated_data):
        """
        保存数据写在这里
        """
        self_define_list = []
        data = validated_data["data"]
        name = validated_data["name"]
        item = validated_data["item"]
        year = validated_data["year"]
        unity = validated_data["unity"]
        document_type = validated_data["document_type"]

        for i in range(len(data)):
            sorted_data = {
                "id": i,
                "data": data[i]
            }
            existed_instance = SelfModule.objects.filter(year=year, name=name[i], item=item, unity=unity,
                                                         document_type=document_type).last()

            if existed_instance:
                validated_data["sorted_data"] = sorted_data
                self.update(existed_instance, validated_data)
            else:
                self_define_obj = SelfModule(
                    document_type=document_type,
                    name=name[i],
                    item=item,
                    year=year,
                    unity=unity,
                    data=sorted_data,
                    create_user=self.context["user_id"],
                    update_user=self.context["user_id"]
                )
                self_define_list.append(self_define_obj)

        return SelfModule.objects.bulk_create(self_define_list)

    def update(self, instance, validated_data):
        instance.data = validated_data["sorted_data"]
        instance.update_user = self.context["user_id"]
        instance.save()

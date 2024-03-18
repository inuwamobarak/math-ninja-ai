from rest_framework import serializers

class MathNinjaSerializer(serializers.Serializer):
    response = serializers.CharField()
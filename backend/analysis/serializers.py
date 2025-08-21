from rest_framework import serializers
from .models import CropImage, AnalysisResult


class CropImageSerializer(serializers.ModelSerializer):
    image_url = serializers.SerializerMethodField()
    
    class Meta:
        model = CropImage
        fields = ['id', 'image_url', 'crop_type', 'original_filename', 'uploaded_at', 'file_size', 'image_type']
        read_only_fields = ['id', 'uploaded_at', 'file_size', 'image_type']
    
    def get_image_url(self, obj):
        return obj.get_image_url()


class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        fields = [
            'disease_detected', 'disease_name', 'confidence', 'severity',
            'bacterial_infection', 'recommendations', 'prevention_strategies',
            'danger_level', 'economic_impact', 'treatment_timeline',
            'monitoring_advice', 'analyzed_at'
        ]


class AnalysisRequestSerializer(serializers.Serializer):
    file_id = serializers.UUIDField()
    crop_type = serializers.ChoiceField(choices=['rice', 'jute'])
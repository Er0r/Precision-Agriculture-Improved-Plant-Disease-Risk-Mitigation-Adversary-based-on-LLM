from django.contrib import admin
from .models import CropImage, AnalysisResult


@admin.register(CropImage)
class CropImageAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'crop_type', 'uploaded_at', 'file_size']
    list_filter = ['crop_type', 'uploaded_at']
    search_fields = ['original_filename']
    readonly_fields = ['id', 'uploaded_at', 'file_size']


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ['image', 'disease_name', 'confidence', 'disease_detected', 'crop_type', 'analyzed_at']
    list_filter = ['disease_detected', 'analyzed_at']
    search_fields = ['disease_name', 'image__original_filename']
    readonly_fields = ['analyzed_at']
    
    def crop_type(self, obj):
        return obj.image.crop_type
    crop_type.short_description = 'Crop Type'
from django.db import models
import uuid


class CropImage(models.Model):
    """Model to store uploaded crop images"""
    
    CROP_CHOICES = [
        ('rice', 'Rice'),
        ('jute', 'Jute'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ImageField(upload_to='images/')
    crop_type = models.CharField(max_length=10, choices=CROP_CHOICES)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.PositiveIntegerField()
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.crop_type} - {self.original_filename}"


class AnalysisResult(models.Model):
    """Model to store analysis results"""
    
    image = models.OneToOneField(CropImage, on_delete=models.CASCADE, related_name='analysis')
    disease_detected = models.BooleanField()
    disease_name = models.CharField(max_length=100)
    confidence = models.FloatField()
    severity = models.CharField(max_length=50)
    bacterial_infection = models.BooleanField(default=False)
    recommendations = models.JSONField(default=list)
    prevention_strategies = models.JSONField(default=list)
    danger_level = models.TextField(blank=True)
    economic_impact = models.TextField(blank=True)
    treatment_timeline = models.TextField(blank=True)
    monitoring_advice = models.TextField(blank=True)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Analysis for {self.image.original_filename} - {self.disease_name}"
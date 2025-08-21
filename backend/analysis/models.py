from django.db import models
import uuid
from PIL import Image
import io
from django.utils import timezone


class CropImage(models.Model):
    """Model to store uploaded crop images as binary data"""
    
    CROP_CHOICES = [
        ('rice', 'Rice'),
        ('jute', 'Jute'),
    ]
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image_data = models.BinaryField()
    image_type = models.CharField(max_length=20, default='image/jpeg')
    crop_type = models.CharField(max_length=10, choices=CROP_CHOICES)
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_size = models.PositiveIntegerField()
    
    class Meta:
        ordering = ['-uploaded_at']
    
    def __str__(self):
        return f"{self.crop_type} - {self.original_filename}"
    
    def save_image(self, image_file):
        """Save image file as binary data"""
        image_content = image_file.read()
        self.image_data = image_content
        self.image_type = image_file.content_type or 'image/jpeg'
        self.file_size = len(image_content)
        self.save()
    
    def get_image_url(self):
        """Get image URL for display"""
        return f"/api/images/{self.id}/"
    
    def get_image_data(self):
        """Get image binary data"""
        return self.image_data
    
    def get_image_pil(self):
        """Get PIL Image object from binary data"""
        try:
            return Image.open(io.BytesIO(self.image_data))
        except Exception as e:
            print(f"Error opening image: {e}")
            return None


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
    
    class Meta:
        db_table = 'analysis_data'
    
    def __str__(self):
        return f"Analysis for {self.image.original_filename} - {self.disease_name}"


class EvaluationResult(models.Model):
    """Store MCP evaluation results"""
    
    mcp_name = models.CharField(max_length=200)
    model_name = models.CharField(max_length=200, blank=True)
    results = models.JSONField(default=dict)
    raw_output = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"EvaluationResult for {self.mcp_name} ({self.model_name}) @ {self.created_at.isoformat()}"
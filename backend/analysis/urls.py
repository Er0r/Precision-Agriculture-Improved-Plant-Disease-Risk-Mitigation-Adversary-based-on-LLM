from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.HealthCheckView.as_view(), name='health_check'),
    path('upload/', views.ImageUploadView.as_view(), name='image_upload'),
    path('analyze/', views.AnalyzeImageView.as_view(), name='analyze_image'),
    path('history/', views.AnalysisHistoryView.as_view(), name='analysis_history'),
    path('images/<uuid:image_id>/', views.ImageDetailView.as_view(), name='image_detail'),
]
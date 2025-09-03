from django.urls import path
from . import views

urlpatterns = [
    path('health/', views.HealthCheckView.as_view(), name='health_check'),
    path('upload/', views.ImageUploadView.as_view(), name='image_upload'),
    path('analyze/', views.AnalyzeImageView.as_view(), name='analyze_image'),
    path('history/', views.AnalysisHistoryView.as_view(), name='analysis_history'),
    path('images/<uuid:image_id>/', views.ImageDetailView.as_view(), name='image_detail'),
    path('serve/<uuid:image_id>/', views.ImageServeView.as_view(), name='serve_image'),
    path('readability-results/', views.ReadabilityResultsView.as_view(), name='readability_results'),
    path('postgresql-evaluation/', views.PostgreSQLEvaluationView.as_view(), name='postgresql_evaluation'),
    path('postgresql-evaluation/export/', views.PostgreSQLEvaluationExportView.as_view(), name='postgresql_evaluation_export'),
    path('clarity-analytics/', views.ClarityAnalyticsView.as_view(), name='clarity_analytics'),
    path('sentiment-analytics/', views.SentimentAnalyticsView.as_view(), name='sentiment_analytics'),
    path('sentiment-analyze/', views.SentimentAnalyzeView.as_view(), name='sentiment_analyze'),
    path('individual-disease-sentiment/', views.IndividualDiseaseSentimentView.as_view(), name='individual_disease_sentiment'),
]
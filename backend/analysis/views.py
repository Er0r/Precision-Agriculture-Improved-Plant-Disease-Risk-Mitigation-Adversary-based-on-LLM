from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from PIL import Image
import os
import uuid

from .models import CropImage, AnalysisResult
from .serializers import CropImageSerializer, AnalysisResultSerializer, AnalysisRequestSerializer
from .mcp_integration import analyze_crop_disease


class HealthCheckView(APIView):
    """Health check endpoint"""
    
    def get(self, request):
        return Response({
            'status': 'healthy',
            'message': 'Crop Disease Analysis API is running'
        })


class ImageUploadView(APIView):
    """Handle image upload"""
    
    parser_classes = (MultiPartParser, FormParser)
    
    def post(self, request):
        try:
            file = request.FILES.get('file')
            crop_type = request.data.get('crop_type', 'rice')
            
            if not file:
                return Response(
                    {'error': 'No file selected'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Validate file type
            allowed_extensions = settings.CROP_DISEASE_SETTINGS['ALLOWED_EXTENSIONS']
            file_extension = file.name.split('.')[-1].lower()
            
            if file_extension not in allowed_extensions:
                return Response(
                    {'error': 'Invalid file type'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create crop image instance
            crop_image = CropImage.objects.create(
                image=file,
                crop_type=crop_type,
                original_filename=file.name,
                file_size=file.size
            )
            
            # Optimize image
            self._optimize_image(crop_image.image.path)
            
            serializer = CropImageSerializer(crop_image)
            
            return Response({
                'success': True,
                'file_id': crop_image.id,
                'message': 'Image uploaded successfully',
                'metadata': serializer.data
            })
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def _optimize_image(self, image_path):
        """Optimize uploaded image"""
        try:
            max_size = settings.CROP_DISEASE_SETTINGS['MAX_IMAGE_SIZE']
            quality = settings.CROP_DISEASE_SETTINGS['IMAGE_QUALITY']
            
            with Image.open(image_path) as img:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.save(image_path, optimize=True, quality=quality)
        except Exception as e:
            print(f"Error optimizing image: {e}")


class AnalyzeImageView(APIView):
    """Handle image analysis"""
    
    def post(self, request):
        serializer = AnalysisRequestSerializer(data=request.data)
        
        if not serializer.is_valid():
            return Response(
                {'error': 'Invalid request data'}, 
                status=status.HTTP_400_BAD_REQUEST
            )
        
        file_id = serializer.validated_data['file_id']
        crop_type = serializer.validated_data['crop_type']
        
        try:
            # Get the uploaded image
            crop_image = get_object_or_404(CropImage, id=file_id)
            
            # Check if analysis already exists
            if hasattr(crop_image, 'analysis'):
                analysis_serializer = AnalysisResultSerializer(crop_image.analysis)
                return Response({
                    'success': True,
                    'analysis': analysis_serializer.data
                })
            
            # Perform analysis
            image_path = crop_image.image.path
            analysis_result = analyze_crop_disease(image_path, crop_type)
            
            # Save analysis result
            analysis = AnalysisResult.objects.create(
                image=crop_image,
                disease_detected=analysis_result.get('disease_detected', False),
                disease_name=analysis_result.get('disease_name', 'Unknown'),
                confidence=analysis_result.get('confidence', 0.0),
                severity=analysis_result.get('severity', 'Unknown'),
                bacterial_infection=analysis_result.get('bacterial_infection', False),
                recommendations=analysis_result.get('recommendations', []),
                prevention_strategies=analysis_result.get('prevention_strategies', []),
                danger_level=analysis_result.get('danger_level', ''),
                economic_impact=analysis_result.get('economic_impact', ''),
                treatment_timeline=analysis_result.get('treatment_timeline', ''),
                monitoring_advice=analysis_result.get('monitoring_advice', '')
            )
            
            analysis_serializer = AnalysisResultSerializer(analysis)
            
            return Response({
                'success': True,
                'analysis': analysis_serializer.data
            })
            
        except CropImage.DoesNotExist:
            return Response(
                {'error': 'File not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'Analysis failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class AnalysisHistoryView(APIView):
    """Get analysis history"""
    
    def get(self, request):
        try:
            # Get recent analyses
            analyses = AnalysisResult.objects.select_related('image').order_by('-analyzed_at')[:10]
            
            history_data = []
            for analysis in analyses:
                history_data.append({
                    'filename': analysis.image.original_filename,
                    'crop_type': analysis.image.crop_type,
                    'disease_name': analysis.disease_name,
                    'confidence': analysis.confidence,
                    'analyzed_at': analysis.analyzed_at,
                    'file_size': analysis.image.file_size
                })
            
            return Response({'images': history_data})
            
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ImageDetailView(APIView):
    """Get specific image details"""
    
    def get(self, request, image_id):
        try:
            crop_image = get_object_or_404(CropImage, id=image_id)
            serializer = CropImageSerializer(crop_image)
            
            response_data = serializer.data
            
            # Add analysis data if available
            if hasattr(crop_image, 'analysis'):
                analysis_serializer = AnalysisResultSerializer(crop_image.analysis)
                response_data['analysis'] = analysis_serializer.data
            
            return Response(response_data)
            
        except CropImage.DoesNotExist:
            return Response(
                {'error': 'Image not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )
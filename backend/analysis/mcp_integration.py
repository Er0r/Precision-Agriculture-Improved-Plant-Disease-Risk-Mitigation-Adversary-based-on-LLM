"""
MCP Integration Module for Crop Disease Analysis
"""

import os
import sys
import json
import importlib
import tempfile
import mimetypes
import django
from pathlib import Path
from threading import Lock

try:
    import tensorflow as tf
except ImportError:
    tf = None


class MCPIntegration:
    """Handles integration with MCP services"""
    
    def __init__(self):
        self.mcp_base_path = Path(__file__).parent.parent.parent / "mcp"
        self.rice_mcp_path = self.mcp_base_path / "rice-disease-analysis"
        self.jute_mcp_path = self.mcp_base_path / "jute-disease-analysis"
    
    def _get_failure_result(self, error_message):
        """Return standardized failure result"""
        return {
            'disease_detected': False,
            'disease_name': 'Unknown',
            'confidence': 0.0,
            'severity': 'Unknown',
            'analysis_failure': True,
            'error': error_message
        }
    
    def _process_mcp_result(self, result):
        """Process and standardize MCP result"""
        predicted_class = result.get('predicted_class', 'Unknown')
        confidence = result.get('confidence', 0.0)
        all_probabilities = result.get('all_probabilities', {})
        
        predicted_normalized = predicted_class.strip().lower()
        is_healthy = (
            'healthy' in predicted_normalized or
            'fresh' in predicted_normalized or
            predicted_normalized in ('healthy', 'healthy_leaf', 'healthy leaf', 'healthy-leaf')
        )
        disease_detected = not is_healthy
        
        if confidence > 0.8:
            severity = 'High' if disease_detected else 'Healthy'
        elif confidence > 0.6:
            severity = 'Moderate'
        else:
            severity = 'Low'
        
        bacterial_infection = 'bacterial' in predicted_class.lower()
        
        return {
            'disease_detected': disease_detected,
            'disease_name': predicted_class,
            'confidence': confidence,
            'severity': severity,
            'bacterial_infection': bacterial_infection,
            'all_probabilities': all_probabilities,
            'recommendations': self._get_basic_recommendations(predicted_class, disease_detected),
            'prevention_strategies': self._get_basic_prevention(),
            'danger_level': severity,
            'economic_impact': self._get_economic_impact(disease_detected, severity),
            'treatment_timeline': self._get_treatment_timeline(disease_detected),
            'monitoring_advice': self._get_monitoring_advice()
        }
    
    def _get_basic_recommendations(self, disease_name, disease_detected):
        """Get basic treatment recommendations"""
        if not disease_detected:
            return ["Continue regular crop monitoring", "Maintain good agricultural practices"]
        
        basic_recs = [
            f"Apply appropriate fungicide for {disease_name}",
            "Remove affected plant parts",
            "Improve field drainage",
            "Increase plant spacing for better air circulation"
        ]
        return basic_recs
    
    def _get_basic_prevention(self):
        """Get basic prevention strategies"""
        return [
            "Use certified disease-free seeds",
            "Practice crop rotation",
            "Maintain proper field sanitation",
            "Apply preventive fungicide sprays",
            "Monitor weather conditions"
        ]
    
    def _get_economic_impact(self, disease_detected, severity):
        """Get economic impact assessment"""
        if not disease_detected:
            return "No economic impact expected"
        
        if severity == 'High':
            return "High economic impact - immediate treatment required"
        elif severity == 'Moderate':
            return "Moderate economic impact - treatment recommended"
        else:
            return "Low economic impact - monitoring advised"
    
    def _get_treatment_timeline(self, disease_detected):
        """Get treatment timeline"""
        if not disease_detected:
            return "No treatment needed"
        return "Begin treatment immediately, repeat after 7-10 days if symptoms persist"
    
    def _get_monitoring_advice(self):
        """Get monitoring advice"""
        return "Monitor crops weekly for disease symptoms and weather conditions"

    def analyze_rice_disease(self, image_path):
        """Analyze rice disease using the MCP service"""
        try:
            sys.path.insert(0, str(self.rice_mcp_path))
            try:
                rice_disease_analysis = importlib.import_module('rice_disease_analysis')
            except Exception:
                return self._get_failure_result('Failed to import rice disease module')

            if tf is None:
                return self._get_failure_result('TensorFlow not available')
            
            model = getattr(rice_disease_analysis, 'model', None)
            if model is None:
                model_path = self.rice_mcp_path / "models" / "rice_disease_cnn_model.h5"
                if model_path.exists():
                    model = tf.keras.models.load_model(str(model_path))
                else:
                    return self._get_failure_result('Model file not found')
            
            CLASS_NAMES = getattr(rice_disease_analysis, 'CLASS_NAMES', 
                                ['Healthy', 'Bacterial Leaf Blight', 'Rice Blast', 'Tungro'])
            IMG_SIZE = getattr(rice_disease_analysis, 'IMG_SIZE', (224, 224))

            if not os.path.exists(image_path):
                return self._get_failure_result(f'File not found: {image_path}')
                        
            if hasattr(rice_disease_analysis, 'predict_disease'):
                result = rice_disease_analysis.predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
            else:
                return self._get_failure_result('predict_disease function not found')

            return self._process_mcp_result(result) if result else self._get_failure_result('Prediction failed')
            
        except Exception as e:
            return self._get_failure_result(f'Rice disease analysis error: {e}')

    def analyze_jute_disease(self, image_path):
        """Analyze jute disease using the MCP service"""
        original_cwd = os.getcwd()
        try:
            # Change to jute directory for proper imports
            os.chdir(str(self.jute_mcp_path))
            sys.path.insert(0, str(self.jute_mcp_path))
            
            try:
                jute_disease_analysis = importlib.import_module('jute_disease_analysis')
            except Exception as e:
                return self._get_failure_result(f'Failed to import jute disease module: {e}')

            try:
                if hasattr(jute_disease_analysis, 'analyze_jute_disease'):
                    result = jute_disease_analysis.analyze_jute_disease(image_path)
                elif hasattr(jute_disease_analysis, 'predict_disease_from_image'):
                    result = jute_disease_analysis.predict_disease_from_image(image_path)
                elif hasattr(jute_disease_analysis, 'predict_disease'):
                    model = getattr(jute_disease_analysis, 'model', None)
                    class_names = getattr(jute_disease_analysis, 'CLASS_NAMES', [])
                    img_size = getattr(jute_disease_analysis, 'IMG_SIZE', (224, 224))
                    result = jute_disease_analysis.predict_disease(image_path, model, class_names, img_size)
                else:
                    return self._get_failure_result('No prediction function found')
            except Exception as e:
                return self._get_failure_result(f'Model execution failed: {e}')
            
            return self._process_mcp_result(result) if result else self._get_failure_result('Prediction failed')
            
        except Exception as e:
            return self._get_failure_result(f'Jute disease analysis error: {e}')
        finally:
            # Always change back to original directory
            os.chdir(original_cwd)
    
    def get_mcp_status(self):
        """Check the status of MCP services"""
        return {
            'rice_mcp_available': self.rice_mcp_path.exists(),
            'jute_mcp_available': self.jute_mcp_path.exists(),
            'rice_model_path': str(self.rice_mcp_path / "models"),
            'jute_model_path': str(self.jute_mcp_path / "models")
        }


def analyze_crop_disease(image_input, crop_type):
    """
    Main function to analyze crop disease and provide treatment recommendations
    """
    import io
    from PIL import Image
    
    using_temp_file = False
    temp_file_path = None
    
    if hasattr(image_input, 'read') and callable(image_input.read):
        try:
            image_input.seek(0)
            pil_image = Image.open(image_input)
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_file.close()
            
            pil_image.save(temp_file.name, format='JPEG')
            
            image_path = temp_file.name
            temp_file_path = temp_file.name
            using_temp_file = True
        except Exception as e:
            raise Exception(f"Error converting file-like object to file: {e}")
    else:
        image_path = image_input
    
    media_paths = [
        Path('f:/Personal/PrecisionAgriculture/backend/media'),
        Path('f:/Personal/PrecisionAgriculture/media'),
    ]
    
    if 'sample_images' in image_path:
        mcp_base_path = Path(__file__).parent.parent.parent / "mcp"
        rice_samples = mcp_base_path / "rice-disease-analysis" / "sample_images"
        jute_samples = mcp_base_path / "jute-disease-analysis" / "sample_images"
        media_paths.extend([rice_samples, jute_samples])
    
    if not Path(image_path).is_absolute():
        image_path = str(Path(image_path).resolve())
    
    filename = Path(image_path).name
    
    if not Path(image_path).exists():
        found = False
        for media_path in media_paths:
            alternate_path = media_path / filename
            if alternate_path.exists():
                image_path = str(alternate_path)
                found = True
                break
                
        if not found:
            filename_parts = filename.split('_')
            if len(filename_parts) > 0:
                file_prefix = filename_parts[0]
                
                for media_path in media_paths:
                    if media_path.exists():
                        for file_path in media_path.glob('*'):
                            if file_path.is_file() and file_prefix in file_path.name:
                                image_path = str(file_path)
                                found = True
                                break
                    if found:
                        break
    
    mcp = MCPIntegration()
    
    if crop_type.lower() == 'rice':
        mcp_result = mcp.analyze_rice_disease(image_path)
    elif crop_type.lower() == 'jute':
        mcp_result = mcp.analyze_jute_disease(image_path)
    else:
        raise ValueError(f"Unsupported crop type: {crop_type}")
    
    mcp_result['crop_type'] = crop_type
    mcp_result['llm_enhanced'] = False
    
    if using_temp_file and temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass
            
    try:
        _save_analysis_to_db(
            result=mcp_result,
            image_path=image_path,
            crop_type=crop_type
        )
    except Exception:
        pass
        
    return mcp_result


_CSV_LOCK = Lock()

def _ensure_django_setup():
    """Ensure Django is configured so we can use the ORM from this module."""
    try:
        from django.conf import settings
        if not settings.configured:
            os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'crop_disease_api.settings')
            django.setup()
    except Exception:
        pass


def _save_analysis_to_db(result: dict, image_path: str, crop_type: str):
    """Save analysis result into PostgreSQL using Django ORM."""
    with _CSV_LOCK:
        try:
            _ensure_django_setup()
            from .models import CropImage, AnalysisResult

            original_filename = os.path.basename(image_path) if image_path else 'unknown.jpg'
            mime_type, _ = mimetypes.guess_type(image_path)
            mime_type = mime_type or 'image/jpeg'

            image_bytes = b''
            file_size = 0
            try:
                if image_path and os.path.exists(image_path):
                    with open(image_path, 'rb') as f:
                        image_bytes = f.read()
                        file_size = len(image_bytes)
            except Exception:
                image_bytes = b''

            crop_image = CropImage(
                image_data=image_bytes,
                image_type=mime_type,
                crop_type=crop_type,
                original_filename=original_filename,
                file_size=file_size
            )
            crop_image.save()

            analysis = AnalysisResult(
                image=crop_image,
                disease_detected=bool(result.get('disease_detected', False)),
                disease_name=result.get('disease_name') or ("Healthy" if not result.get('disease_detected') else 'Unknown'),
                confidence=float(result.get('confidence', 0.0) or 0.0),
                severity=result.get('severity', 'Unknown'),
                bacterial_infection=bool(result.get('bacterial_infection', False)),
                recommendations=result.get('recommendations') or [],
                prevention_strategies=result.get('prevention_strategies') or [],
                danger_level=result.get('danger_level') or '',
                economic_impact=result.get('economic_impact') or '',
                treatment_timeline=result.get('treatment_timeline') or '',
                monitoring_advice=result.get('monitoring_advice') or ''
            )
            analysis.save()

        except Exception as e:
            print(f"Failed to save analysis to DB: {e}")


def save_evaluation_result_to_db(mcp_name: str, model_name: str = '', results=None, raw_output: str = ''):
    """Save evaluation results into DB"""
    try:
        _ensure_django_setup()
        from .models import EvaluationResult

        ev = EvaluationResult(
            mcp_name=mcp_name,
            model_name=model_name or '',
            results=results or {},
            raw_output=raw_output or ''
        )
        ev.save()
        return ev
    except Exception as e:
        print(f"Failed to save evaluation result to DB: {e}")
        return None

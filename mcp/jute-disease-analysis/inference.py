
import sys
import os
import argparse
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

try:
    from jute_disease_analysis import predict_disease, model, CLASS_NAMES, IMG_SIZE
except ImportError as e:
    print(f"Error importing jute disease analysis module: {e}")
    predict_disease = None
    model = None
    CLASS_NAMES = None
    IMG_SIZE = None

def quick_prediction(image_path, model_path=None):
    if predict_disease is None or model is None:
        return None
    
    try:
        import datetime
        result = predict_disease(image_path, model, CLASS_NAMES, IMG_SIZE)
        if result:
            result['timestamp'] = datetime.datetime.now().isoformat()
        return result
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Jute Disease Detection Inference')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', help='Path to model file (optional)')
    parser.add_argument('--show', '-s', action='store_true', help='Show the input image')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)')
    parser.add_argument('--save-to-db', action='store_true', help='Save results to central DB instead of file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found")
        return
    
    print(f"Analyzing image: {args.image}")
    
    result = quick_prediction(args.image, args.model)
    
    if result:
        print("\n=== Prediction Results ===")
        print(f"Predicted Disease: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Timestamp: {result['timestamp']}")
        
        if 'all_probabilities' in result:
            print("\nAll Class Probabilities:")
            for class_name, prob in result['all_probabilities'].items():
                print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        
        saved_to_db = False
        if args.save_to_db or (args.output and args.output.endswith('evaluation_results.json')):
            try:
                repo_root = Path(__file__).resolve().parent.parent.parent
                backend_path = repo_root / 'backend'
                sys.path.insert(0, str(backend_path))
                
                from analysis.mcp_integration import save_evaluation_result_to_db
                
                mcp_name = 'jute-disease-analysis'
                model_name = args.model or ''
                save_res = save_evaluation_result_to_db(
                    mcp_name=mcp_name, 
                    model_name=model_name, 
                    results=result, 
                    raw_output=json.dumps(result)
                )
                if save_res:
                    print(f"\nResults saved to DB: id={getattr(save_res,'id','unknown')}")
                    saved_to_db = True
            except Exception as e:
                print(f"Warning: Failed to save to DB, falling back to file: {e}")

        if not saved_to_db and args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        if args.show:
            try:
                img = Image.open(args.image)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.title(f"Predicted: {result['predicted_class']} (Confidence: {result['confidence']:.3f})")
                plt.axis('off')
                plt.show()
            except Exception as e:
                print(f"Could not display image: {e}")
    
    else:
        print("Prediction failed. Make sure the model is trained and saved.")

if __name__ == "__main__":
    main()
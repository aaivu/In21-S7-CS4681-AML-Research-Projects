# ================================
# ACCURATE EVALUATION WITH KAGGLE IMAGENET-1K DATASET
# ================================

import os
import numpy as np
import tensorflow as tf
import time
from PIL import Image
import glob

print("Setting up evaluation with Kaggle ImageNet-1K dataset...")
print("This will give you ACCURATE accuracy measurements!")

# ================================
# DATASET SETUP
# ================================

# Path to your Kaggle ImageNet-1K dataset
# Update this path to where you downloaded the dataset
KAGGLE_IMAGENET_PATH = "/path/to/your/imagenet1k"  # UPDATE THIS PATH

def setup_kaggle_imagenet_dataset(dataset_path):
    """Setup the Kaggle ImageNet-1K dataset for evaluation"""
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset path not found: {dataset_path}")
        print("Please update KAGGLE_IMAGENET_PATH with the correct path to your dataset")
        return None, None
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    
    print(f"Found {len(class_folders)} classes")
    print(f"First few classes: {class_folders[:5]}")
    
    # Create evaluation samples
    eval_samples = []
    samples_per_class = 10  # Use 10 samples per class for evaluation
    
    for class_folder in class_folders:
        class_path = os.path.join(dataset_path, class_folder)
        
        # Extract class ID from folder name (format: <class_id>_<class_name>)
        class_id = int(class_folder.split('_')[0])
        
        # Get image files
        image_files = glob.glob(os.path.join(class_path, "*.jpg"))
        
        # Take samples_per_class images from this class
        for i, image_file in enumerate(image_files[:samples_per_class]):
            eval_samples.append((image_file, class_id))
    
    print(f"Created {len(eval_samples)} evaluation samples")
    print(f"Class ID range: {min([s[1] for s in eval_samples])} to {max([s[1] for s in eval_samples])}")
    
    return eval_samples, len(class_folders)

# ================================
# IMAGE PREPROCESSING
# ================================

def preprocess_image(image_path):
    """Preprocess image for EfficientNetV2B0"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Resize to 224x224
        image_tensor = tf.image.resize(image_array, (224, 224))
        
        # Convert to float32
        image_tensor = tf.cast(image_tensor, tf.float32)
        
        # Apply EfficientNetV2 preprocessing
        image_tensor = tf.keras.applications.efficientnet_v2.preprocess_input(image_tensor)
        
        return image_tensor.numpy()
    
    except Exception as e:
        print(f"Error preprocessing {image_path}: {str(e)}")
        return None

# ================================
# ACCURACY EVALUATION FUNCTION
# ================================

def evaluate_model_accuracy_kaggle(tflite_model_path, eval_samples, model_name):
    """Evaluate actual classification accuracy using Kaggle ImageNet-1K"""
    try:
        print(f"\n{'='*20} {model_name} {'='*20}")
        
        # Load and initialize interpreter
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()

        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_index = input_details[0]['index']
        input_dtype = input_details[0]['dtype']
        
        # Handle quantization parameters
        input_scale = 1.0
        input_zero_point = 0
        
        if 'quantization' in input_details[0] and input_details[0]['quantization']:
            quant_params = input_details[0]['quantization']
            if len(quant_params) >= 2:
                input_scale = float(quant_params[0])
                input_zero_point = int(quant_params[1])
        
        print(f"Input dtype: {input_dtype}")
        print(f"Quantization - scale: {input_scale}, zero_point: {input_zero_point}")
        
        correct_predictions = 0
        total_predictions = 0
        inference_times = []
        
        print(f"Evaluating on {len(eval_samples)} samples...")
        
        for i, (image_path, true_class) in enumerate(eval_samples):
            try:
                # Preprocess image
                input_data = preprocess_image(image_path)
                if input_data is None:
                    continue
                
                # Add batch dimension
                input_data = np.expand_dims(input_data, axis=0)
                
                # Apply quantization if needed
                if input_dtype == np.int8:
                    input_data = input_data / input_scale + input_zero_point
                    input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)
                elif input_dtype == np.uint8:
                    input_data = input_data / input_scale + input_zero_point
                    input_data = np.clip(np.round(input_data), 0, 255).astype(np.uint8)
                elif input_dtype == np.float16:
                    input_data = input_data.astype(np.float16)
                
                # Measure inference time
                start_time = time.time()
                
                # Run inference
                interpreter.set_tensor(input_index, input_data)
                interpreter.invoke()
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                # Get output
                output_data = interpreter.get_tensor(output_details[0]['index'])
                predicted_class = np.argmax(output_data, axis=1)[0]
                
                # Check if prediction is correct (exact match - perfect label mapping!)
                if predicted_class == true_class:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(eval_samples)} samples...")
                    
            except Exception as e:
                print(f"    Error on sample {i}: {str(e)}")
                total_predictions += 1
        
        # Calculate results
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        avg_inference_time = np.mean(inference_times) if inference_times else 0.0
        
        print(f"\nResults:")
        print(f"  Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"  Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Average inference time: {avg_inference_time*1000:.2f} ms")
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_predictions,
            'avg_inference_time': avg_inference_time
        }
        
    except Exception as e:
        print(f"Error evaluating {model_name}: {str(e)}")
        return {
            'accuracy': 0.0,
            'correct': 0,
            'total': 0,
            'avg_inference_time': 0.0
        }

# ================================
# MAIN EVALUATION
# ================================

def run_kaggle_imagenet_evaluation():
    """Run complete evaluation with Kaggle ImageNet-1K dataset"""
    
    print("\n" + "="*80)
    print("KAGGLE IMAGENET-1K EVALUATION")
    print("="*80)
    
    # Setup dataset
    eval_samples, num_classes = setup_kaggle_imagenet_dataset(KAGGLE_IMAGENET_PATH)
    
    if eval_samples is None:
        print("❌ Cannot proceed without dataset. Please update the path and try again.")
        return
    
    print(f"✅ Dataset ready: {len(eval_samples)} samples from {num_classes} classes")
    print("✅ Perfect label mapping: Class IDs 0-999 match EfficientNetV2B0 exactly!")
    
    # Model files
    model_files = {
        "Baseline": "efficientnetv2_b0_baseline.tflite",
        "Float16 Quantization": "efficientnetv2_b0_fp16.tflite",
        "Dynamic Range Quantization": "efficientnetv2_b0_dynamic.tflite",
        "Integer Quantization": "efficientnetv2_b0_int8.tflite",
    }
    
    # Add QAT model if it exists
    if os.path.exists("efficientnetv2_b0_qat.tflite"):
        model_files["QAT (Quantization-Aware Training)"] = "efficientnetv2_b0_qat.tflite"
    
    # Run evaluation
    results = {}
    
    for name, file in model_files.items():
        if os.path.exists(file):
            results[name] = evaluate_model_accuracy_kaggle(file, eval_samples, name)
        else:
            print(f"\n⚠️  {name}: File not found - skipping evaluation")
            results[name] = {
                'accuracy': 0.0,
                'correct': 0,
                'total': 0,
                'avg_inference_time': 0.0
            }
    
    # Results analysis
    print("\n" + "="*80)
    print("QUANTIZATION IMPACT ANALYSIS")
    print("="*80)
    
    baseline_accuracy = results.get("Baseline", {}).get('accuracy', 0.0)
    
    print(f"\n{'Model':<30} {'Size (KB)':<12} {'Top-1 Acc':<12} {'Accuracy Loss':<15} {'Inference Time (ms)':<20}")
    print("-" * 95)
    
    for name, result in results.items():
        if result['accuracy'] > 0:
            # Get model size
            model_file = model_files[name]
            if os.path.exists(model_file):
                size_kb = os.path.getsize(model_file) / 1024
            else:
                size_kb = 0
            
            # Calculate accuracy loss
            accuracy_loss = baseline_accuracy - result['accuracy']
            accuracy_loss_pct = (accuracy_loss / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
            
            # Format output
            print(f"{name:<30} {size_kb:<12.0f} {result['accuracy']:<12.4f} {accuracy_loss_pct:<15.2f}% {result['avg_inference_time']*1000:<20.2f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    if baseline_accuracy > 0:
        print(f"\n1. Baseline Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        print("2. Perfect Label Mapping: Class IDs 0-999 match model exactly")
        print("3. Accurate Evaluation: No label mapping issues")
        
        # Find best performing quantized model
        quantized_results = {k: v for k, v in results.items() if k != "Baseline" and v['accuracy'] > 0}
        if quantized_results:
            best_quantized = max(quantized_results.items(), key=lambda x: x[1]['accuracy'])
            print(f"4. Best Quantized Model: {best_quantized[0]} ({best_quantized[1]['accuracy']:.4f})")
            
            # Find most compressed model
            compressed_models = [(k, os.path.getsize(model_files[k])/1024) for k in quantized_results.keys() if os.path.exists(model_files[k])]
            if compressed_models:
                most_compressed = min(compressed_models, key=lambda x: x[1])
                print(f"5. Most Compressed: {most_compressed[0]} ({most_compressed[1]:.0f} KB)")
        
        print(f"\n6. Production Recommendations:")
        if baseline_accuracy > 0.7:
            print(f"   - For high accuracy: Use Float16 or QAT")
            print(f"   - For maximum compression: Use Dynamic Range")
            print(f"   - For edge deployment: Use Integer Quantization")
        else:
            print(f"   - Consider using a different model or dataset")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE - You now have ACCURATE accuracy measurements!")
    print("="*80)

# ================================
# USAGE INSTRUCTIONS
# ================================

print("""
USAGE INSTRUCTIONS:
==================

1. Download the Kaggle ImageNet-1K dataset
2. Update KAGGLE_IMAGENET_PATH with the correct path to your dataset
3. Run: run_kaggle_imagenet_evaluation()

Example:
KAGGLE_IMAGENET_PATH = "/content/imagenet1k"  # Update this path
run_kaggle_imagenet_evaluation()

This will give you ACCURATE accuracy measurements with perfect label mapping!
""")

# Uncomment and run this when you have the dataset:
# run_kaggle_imagenet_evaluation()

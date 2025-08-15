"""
Evaluation and error analysis utilities
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_ground_truth(gt_path):
    """Load and preprocess ground truth data"""
    df = pd.read_csv(gt_path, sep=';', engine='python')
    df.columns = [col.strip() for col in df.columns]
    return df


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy between true and predicted labels"""
    return accuracy_score(y_true, y_pred)


def generate_confusion_matrix(y_true, y_pred):
    """Generate confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def generate_classification_report(y_true, y_pred):
    """Generate detailed classification report"""
    return classification_report(y_true, y_pred)


def interpret_prediction(prediction, thresholds):
    """Interpret prediction probability into risk levels"""
    if prediction > thresholds['high']:
        return "HIGH", "🔴"
    elif prediction > thresholds['moderate']:
        return "MODERATE", "🟡"
    else:
        return "LOW", "🟢"


def perform_error_analysis(results_df, model_name, label_col='Label', pred_col='predicted_class'):
    """
    Perform comprehensive error analysis on model results
    
    Args:
        results_df: DataFrame with ground truth and predictions
        model_name: Name of the model for reporting
        label_col: Column name for ground truth labels
        pred_col: Column name for predicted labels
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Model Error Analysis")
    print(f"{'='*60}")
    
    # Calculate accuracy
    if label_col in results_df.columns and pred_col in results_df.columns:
        y_true = results_df[label_col]
        y_pred = results_df[pred_col]
        
        accuracy = calculate_accuracy(y_true, y_pred) * 100
        print(f"Overall Accuracy: {accuracy:.2f}%")
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = generate_confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Classification Report
        print("\nClassification Report:")
        report = generate_classification_report(y_true, y_pred)
        print(report)
        
    #     # Error Analysis
    #     if isinstance(y_true.iloc[0], str):  # String labels (control/dementia)
    #         false_positives = results_df[
    #             (results_df[pred_col] == 'dementia') & (results_df[label_col] == 'control')
    #         ]
    #         false_negatives = results_df[
    #             (results_df[pred_col] == 'control') & (results_df[label_col] == 'dementia')
    #         ]
    #     else:  # Numeric labels (0/1)
    #         false_positives = results_df[
    #             (results_df[pred_col] == 1) & (results_df[label_col] == 0)
    #         ]
    #         false_negatives = results_df[
    #             (results_df[pred_col] == 0) & (results_df[label_col] == 1)
    #         ]
        
    #     print(f"\nFalse Positives: {len(false_positives)}")
    #     if len(false_positives) > 0:
    #         print("Files wrongly classified as positive:")
    #         print(false_positives[['ID']].values.flatten())
            
    #     print(f"\nFalse Negatives: {len(false_negatives)}")
    #     if len(false_negatives) > 0:
    #         print("Files wrongly classified as negative:")
    #         print(false_negatives[['ID']].values.flatten())
    
    # else:
    #     print(f"Error: Required columns not found in results DataFrame")
    #     print(f"Available columns: {list(results_df.columns)}")
    
    return results_df


def save_results(results_df, filename, model_name):
    """Save results to CSV file"""
    results_df.to_csv(filename, index=False)
    print(f"✓ {model_name} predictions saved to {filename}")


def compare_models(bilstm_results, andy_results, liam_results):
    """Compare performance across all three models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    models_data = [
        ("BiLSTM", bilstm_results, 'GroundTruth', 'PredictedLabel'),
        ("Andy", andy_results, 'Label', 'predicted_class'),
        ("Liam", liam_results, 'Label', 'predicted_class')
    ]
    
    comparison_results = []
    
    for model_name, df, label_col, pred_col in models_data:
        if label_col in df.columns and pred_col in df.columns:
            y_true = df[label_col]
            y_pred = df[pred_col]
            accuracy = calculate_accuracy(y_true, y_pred) * 100
            comparison_results.append({
                'Model': model_name,
                'Accuracy': f"{accuracy:.2f}%",
                'Test_Files': len(df)
            })
        else:
            comparison_results.append({
                'Model': model_name,
                'Accuracy': 'N/A',
                'Test_Files': len(df) if df is not None else 0
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    print(comparison_df.to_string(index=False))
    print(f"{'='*60}")
    
    return comparison_df
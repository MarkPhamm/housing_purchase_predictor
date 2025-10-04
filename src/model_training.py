"""
Model training and evaluation module for house purchase prediction.

This module implements multiple ML models and provides comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import logging
from typing import Dict, Tuple, Any
import joblib
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import XGBoost, but make it optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available. Skipping XGBoost model.")


def get_model_configs() -> Dict[str, Dict]:
    """
    Get configurations for different models.
    
    Returns:
        Dict[str, Dict]: Dictionary of model names and their configurations
    """
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            'description': 'Linear model with L2 regularization'
        },
        'Random Forest': {
            'model': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            ),
            'description': 'Ensemble of decision trees'
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'description': 'Sequential ensemble boosting method'
        },
        'Support Vector Machine': {
            'model': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            ),
            'description': 'Support vector classifier with RBF kernel'
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(
                n_neighbors=15,
                weights='distance',
                n_jobs=-1
            ),
            'description': 'Distance-based classifier'
        },
        'Naive Bayes': {
            'model': GaussianNB(),
            'description': 'Probabilistic classifier based on Bayes theorem'
        }
    }
    
    # Add XGBoost only if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = {
            'model': XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'description': 'Optimized gradient boosting framework'
        }
    
    return models


def train_model(model: Any,
                X_train: pd.DataFrame,
                y_train: pd.Series,
                model_name: str) -> Any:
    """
    Train a single model.
    
    Args:
        model: Scikit-learn compatible model
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        model_name (str): Name of the model
        
    Returns:
        Trained model
    """
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    logger.info(f"{model_name} training completed")
    return model


def evaluate_model(model: Any,
                   X: pd.DataFrame,
                   y: pd.Series,
                   model_name: str,
                   dataset_name: str = 'Validation') -> Dict:
    """
    Evaluate a trained model on a given dataset.
    
    Args:
        model: Trained model
        X (pd.DataFrame): Features
        y (pd.Series): True labels
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset (e.g., 'Validation', 'Test')
        
    Returns:
        Dict: Dictionary containing evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'dataset': dataset_name,
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1_score': f1_score(y, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }
    
    # ROC AUC if probabilities available
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
    
    logger.info(f"\n{model_name} - {dataset_name} Set Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
    if y_pred_proba is not None:
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def train_and_evaluate_all_models(X_train: pd.DataFrame,
                                   y_train: pd.Series,
                                   X_val: pd.DataFrame,
                                   y_val: pd.Series) -> Tuple[Dict, pd.DataFrame]:
    """
    Train and evaluate all models on the validation set.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation labels
        
    Returns:
        Tuple[Dict, pd.DataFrame]: Dictionary of trained models and DataFrame of results
    """
    models_config = get_model_configs()
    trained_models = {}
    results = []
    
    logger.info("="*70)
    logger.info("TRAINING AND EVALUATING ALL MODELS")
    logger.info("="*70)
    
    for model_name, config in models_config.items():
        try:
            # Train model
            trained_model = train_model(
                config['model'],
                X_train,
                y_train,
                model_name
            )
            
            # Evaluate on validation set
            val_metrics = evaluate_model(
                trained_model,
                X_val,
                y_val,
                model_name,
                'Validation'
            )
            
            # Store results
            trained_models[model_name] = trained_model
            results.append(val_metrics)
            
            logger.info("-" * 70)
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('f1_score', ascending=False)
    
    return trained_models, results_df


def select_best_model(results_df: pd.DataFrame,
                      trained_models: Dict,
                      metric: str = 'f1_score') -> Tuple[str, Any]:
    """
    Select the best model based on a given metric.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing model evaluation results
        trained_models (Dict): Dictionary of trained models
        metric (str): Metric to use for selection
        
    Returns:
        Tuple[str, Any]: Best model name and model object
    """
    best_model_name = results_df.iloc[0]['model_name']
    best_model = trained_models[best_model_name]
    best_score = results_df.iloc[0][metric]
    
    logger.info("\n" + "="*70)
    logger.info("BEST MODEL SELECTION")
    logger.info("="*70)
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"Best {metric}: {best_score:.4f}")
    logger.info("="*70)
    
    return best_model_name, best_model


def get_feature_importance(model: Any,
                           feature_names: list,
                           model_name: str,
                           top_n: int = 20) -> pd.DataFrame:
    """
    Get feature importance from tree-based models.
    
    Args:
        model: Trained model
        feature_names (list): List of feature names
        model_name (str): Name of the model
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: DataFrame of feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Features for {model_name}:")
        logger.info(feature_importance_df.head(top_n).to_string(index=False))
        
        return feature_importance_df
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\nTop {top_n} Features for {model_name}:")
        logger.info(feature_importance_df.head(top_n).to_string(index=False))
        
        return feature_importance_df
    else:
        logger.info(f"{model_name} does not support feature importance")
        return None


def save_model_and_artifacts(model: Any,
                              model_name: str,
                              scaler: Any,
                              encoders: Dict,
                              feature_names: list,
                              results_df: pd.DataFrame,
                              save_dir: str = 'models') -> None:
    """
    Save trained model and preprocessing artifacts.
    
    Args:
        model: Trained model
        model_name (str): Name of the model
        scaler: Fitted scaler
        encoders (Dict): Dictionary of encoders
        feature_names (list): List of feature names
        results_df (pd.DataFrame): DataFrame of model results
        save_dir (str): Directory to save artifacts
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"{save_dir}/best_model_{model_name.replace(' ', '_')}_{timestamp}.pkl"
    joblib.dump(model, model_filename)
    logger.info(f"Model saved to {model_filename}")
    
    # Save scaler
    scaler_filename = f"{save_dir}/scaler_{timestamp}.pkl"
    joblib.dump(scaler, scaler_filename)
    
    # Save encoders
    encoders_filename = f"{save_dir}/encoders_{timestamp}.pkl"
    joblib.dump(encoders, encoders_filename)
    
    # Save feature names
    feature_names_filename = f"{save_dir}/feature_names_{timestamp}.json"
    with open(feature_names_filename, 'w') as f:
        json.dump(feature_names, f)
    
    # Save results
    results_filename = f"{save_dir}/model_comparison_{timestamp}.csv"
    results_df.to_csv(results_filename, index=False)
    logger.info(f"Model comparison saved to {results_filename}")
    
    logger.info(f"All artifacts saved to {save_dir}/ directory")


def generate_detailed_report(results_df: pd.DataFrame,
                              best_model_name: str,
                              test_metrics: Dict) -> str:
    """
    Generate a detailed text report of model performance.
    
    Args:
        results_df (pd.DataFrame): DataFrame of validation results
        best_model_name (str): Name of the best model
        test_metrics (Dict): Test set metrics
        
    Returns:
        str: Formatted report
    """
    report = "\n" + "="*70 + "\n"
    report += "HOUSE PURCHASE PREDICTION - MODEL PERFORMANCE REPORT\n"
    report += "="*70 + "\n\n"
    
    report += "VALIDATION SET RESULTS (All Models):\n"
    report += "-"*70 + "\n"
    report += results_df[['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']].to_string(index=False)
    report += "\n\n"
    
    report += f"BEST MODEL: {best_model_name}\n"
    report += "-"*70 + "\n"
    report += f"Selected based on F1 Score\n\n"
    
    report += "TEST SET PERFORMANCE (Best Model):\n"
    report += "-"*70 + "\n"
    report += f"Accuracy:  {test_metrics['accuracy']:.4f}\n"
    report += f"Precision: {test_metrics['precision']:.4f}\n"
    report += f"Recall:    {test_metrics['recall']:.4f}\n"
    report += f"F1 Score:  {test_metrics['f1_score']:.4f}\n"
    if 'roc_auc' in test_metrics:
        report += f"ROC AUC:   {test_metrics['roc_auc']:.4f}\n"
    
    report += "\nConfusion Matrix:\n"
    cm = np.array(test_metrics['confusion_matrix'])
    report += f"              Predicted No    Predicted Yes\n"
    report += f"Actual No     {cm[0][0]:12d}    {cm[0][1]:13d}\n"
    report += f"Actual Yes    {cm[1][0]:12d}    {cm[1][1]:13d}\n"
    
    report += "\n" + "="*70 + "\n"
    
    return report


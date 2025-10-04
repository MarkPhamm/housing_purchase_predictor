"""
Main pipeline for house purchase prediction model.

This script orchestrates the entire ML pipeline from data loading to model evaluation.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_preprocessing import preprocess_pipeline
from model_training import (
    train_and_evaluate_all_models,
    select_best_model,
    evaluate_model,
    get_feature_importance,
    save_model_and_artifacts,
    generate_detailed_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main function to run the complete ML pipeline.
    """
    logger.info("Starting House Purchase Prediction Pipeline")
    logger.info("="*70)
    
    # Configuration - use absolute paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / 'data' / 'global_house_purchase_dataset.csv'
    MODELS_DIR = PROJECT_ROOT / 'models'
    RANDOM_STATE = 42
    
    # Convert to strings
    DATA_PATH = str(DATA_PATH)
    MODELS_DIR = str(MODELS_DIR)
    
    try:
        # Step 1: Data Preprocessing
        logger.info("\n" + "="*70)
        logger.info("STEP 1: DATA PREPROCESSING")
        logger.info("="*70)
        
        X_train, X_val, X_test, y_train, y_val, y_test, encoders, scaler, feature_names = preprocess_pipeline(
            filepath=DATA_PATH,
            random_state=RANDOM_STATE
        )
        
        # Step 2: Train and Evaluate Multiple Models
        logger.info("\n" + "="*70)
        logger.info("STEP 2: MODEL TRAINING AND VALIDATION")
        logger.info("="*70)
        
        trained_models, results_df = train_and_evaluate_all_models(
            X_train, y_train, X_val, y_val
        )
        
        # Step 3: Select Best Model
        logger.info("\n" + "="*70)
        logger.info("STEP 3: MODEL SELECTION")
        logger.info("="*70)
        
        best_model_name, best_model = select_best_model(results_df, trained_models)
        
        # Step 4: Evaluate Best Model on Test Set
        logger.info("\n" + "="*70)
        logger.info("STEP 4: FINAL EVALUATION ON TEST SET")
        logger.info("="*70)
        
        test_metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
            best_model_name,
            'Test'
        )
        
        # Step 5: Feature Importance Analysis
        logger.info("\n" + "="*70)
        logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*70)
        
        feature_importance_df = get_feature_importance(
            best_model,
            feature_names,
            best_model_name,
            top_n=20
        )
        
        # Step 6: Save Model and Artifacts
        logger.info("\n" + "="*70)
        logger.info("STEP 6: SAVING MODEL AND ARTIFACTS")
        logger.info("="*70)
        
        save_model_and_artifacts(
            model=best_model,
            model_name=best_model_name,
            scaler=scaler,
            encoders=encoders,
            feature_names=feature_names,
            results_df=results_df,
            save_dir=MODELS_DIR
        )
        
        # Step 7: Generate Final Report
        logger.info("\n" + "="*70)
        logger.info("STEP 7: GENERATING FINAL REPORT")
        logger.info("="*70)
        
        final_report = generate_detailed_report(
            results_df=results_df,
            best_model_name=best_model_name,
            test_metrics=test_metrics
        )
        
        print(final_report)
        
        # Save report to file
        report_filename = f"{MODELS_DIR}/model_report.txt"
        with open(report_filename, 'w') as f:
            f.write(final_report)
        logger.info(f"Final report saved to {report_filename}")
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*70)
        
        return best_model, results_df, test_metrics
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()


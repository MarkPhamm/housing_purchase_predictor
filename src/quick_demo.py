"""
Quick demo script to test the pipeline on a small subset of data.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent))

import logging

from data_preprocessing import preprocess_pipeline
from model_training import (
    evaluate_model,
    generate_detailed_report,
    get_feature_importance,
    select_best_model,
    train_and_evaluate_all_models,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    """Quick demo on a subset of data."""

    print("=" * 70)
    print("HOUSE PURCHASE PREDICTION - QUICK DEMO")
    print("=" * 70)
    print("\nThis demo uses a subset of the data for faster execution.")
    print("For full pipeline, run: python main_pipeline.py\n")

    # Load only first 10000 rows for demo - use absolute path
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "global_house_purchase_dataset.csv"
    logger.info("Loading subset of data (10,000 rows)...")
    df = pd.read_csv(str(DATA_PATH), nrows=10000)

    # Save temporarily
    temp_file = "/tmp/demo_data.csv"
    df.to_csv(temp_file, index=False)

    # Run preprocessing
    logger.info("\n" + "=" * 70)
    logger.info("DATA PREPROCESSING")
    logger.info("=" * 70)
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, scaler, feature_names = (
        preprocess_pipeline(filepath=temp_file, random_state=42)
    )

    # Train only a subset of fast models
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODELS (Fast models only for demo)")
    logger.info("=" * 70)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
    }

    results = []
    for name, model in models.items():
        logger.info(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        y_pred = model.predict(X_val)
        y_proba = (
            model.predict_proba(X_val)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred),
        }
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_val, y_proba)

        results.append(metrics)

        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1_score']:.4f}")
        if y_proba is not None:
            print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

    # Results summary
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("f1_score", ascending=False)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]["model_name"]
    best_f1 = results_df.iloc[0]["f1_score"]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print(f"F1 Score: {best_f1:.4f}")
    print("=" * 70)

    print("\n✅ Demo completed successfully!")
    print("📝 For full pipeline with all models, run: python main_pipeline.py")
    print("📊 This will train 7 models on the complete dataset and save the best one.")


if __name__ == "__main__":
    main()

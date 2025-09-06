import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from src import config
import os
import matplotlib.pyplot as plt
import seaborn as sns

def save_performance_plots(y_val, predictions, model, feature_names):
    """Saves plots visualizing model performance."""
    print("--- Saving performance plots ---")
    
    # Ensure the figures directory exists
    os.makedirs(config.OUTPUTS_DIR / "figures", exist_ok=True)
    
    # 1. Actual vs. Predicted Plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_val, y=predictions, alpha=0.5)
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], '--r', linewidth=2)
    plt.xlabel("Actual Preference (p)")
    plt.ylabel("Predicted Preference (p)")
    plt.title("Actual vs. Predicted Values")
    plt.savefig(config.OUTPUTS_DIR / "figures" / "actual_vs_predicted.png")
    plt.close()
    print("✅ Actual vs. Predicted plot saved.")

    # 2. Residuals Plot
    residuals = y_val - predictions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=predictions, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title("Residuals Plot")
    plt.savefig(config.OUTPUTS_DIR / "figures" / "residuals_plot.png")
    plt.close()
    print("✅ Residuals plot saved.")

    # 3. Feature Importance Plot
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': model.feature_importances_})
    top_20_features = feature_importances.sort_values(by='importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=top_20_features)
    plt.title("Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(config.OUTPUTS_DIR / "figures" / "feature_importance.png")
    plt.close()
    print("✅ Feature Importance plot saved.")


def train_and_evaluate(final_train_df):
    """
    Trains a LightGBM regression model, evaluates its performance, and saves it.
    """
    print("--- Starting Model Training and Evaluation ---")
    
    features_df = final_train_df.drop(columns=['segment', 'banner_id', config.TARGET_COLUMN])
    target = final_train_df[config.TARGET_COLUMN]

    X_train, X_val, y_train, y_val = train_test_split(
        features_df, target, test_size=0.2, random_state=config.RANDOM_STATE
    )
    
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)
    
    print("\nTraining LightGBM model...")
    model.fit(X_train, y_train)
    print("✅ Model training complete.")
    
    predictions = model.predict(X_val)
    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)
    competition_score = mae * 100
    
    print("\n--- Model Performance on Validation Set ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"Competition Score (MAE * 100): {competition_score:.4f}")
    
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = config.MODELS_DIR / "lightgbm_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Call the new function to save plots
    save_performance_plots(y_val, predictions, model, features_df.columns)
    
    return model
import numpy as np
import pandas as pd


def calculate_model_agreement(models, X_test):
    
    # Store each model's prediction
    individual_predictions = {}
    all_predictions = []
    
    # Skip these - they're not individual models (COMPLETE LIST)
    skip_keys = [
        # Metadata
        'feature_cols', 'training_date', 'version', 'scaler', 'learned_weights',
        # Complex ensembles
        'moe', 'tabnet_meta', 'bma_weights', 'attention_ensemble', 'attention_model_names',
        # Validation outputs
        'conformal_threshold', 'adversarial_auc', 'nas_config', 'diverse_selection', 
        'posterior_samples', 'flaml',
        # External data metadata
        'external_features', 'external_quality', 'external_importance',
        # Data quality
        'data_quality_score', 'data_completeness'
    ]
    
    # Ask each model for prediction
    for name, model in models.items():
        # Skip non-model entries
        if name in skip_keys:
            continue
        
        try:
            # Get prediction (0=SELL, 1=HOLD, 2=BUY)
            pred = model.predict(X_test)
            individual_predictions[name] = pred
            all_predictions.append(pred)
        except Exception as e:
            # If model fails, skip it
            print(f"   ⚠️ Model {name} failed: {e}")
            continue
    
    # If no models worked, return empty
    if len(all_predictions) == 0:
        return {
            'individual_predictions': {},
            'agreement_percentage': 0,
            'vote_counts': {},
            'majority_class': None,
            'total_models': 0
        }
    
    # Convert to numpy array for easy counting
    all_predictions = np.array(all_predictions)
    
    # Count votes for each class
    # For each data point, count how many models voted for each class
    vote_counts = {}
    for i in range(len(X_test)):
        votes = all_predictions[:, i]
        
        # Count: How many said 0 (SELL), 1 (HOLD), 2 (BUY)?
        unique, counts = np.unique(votes, return_counts=True)
        vote_counts[i] = dict(zip(unique, counts))
    
    # Calculate agreement for each prediction
    agreement_percentages = []
    majority_classes = []
    
    for i in range(len(X_test)):
        votes = all_predictions[:, i]
        
        # What did most models say?
        unique, counts = np.unique(votes, return_counts=True)
        majority_idx = np.argmax(counts)
        majority_class = unique[majority_idx]
        majority_count = counts[majority_idx]
        
        # Agreement = (votes for majority) / (total votes)
        agreement = (majority_count / len(votes)) * 100
        
        agreement_percentages.append(agreement)
        majority_classes.append(majority_class)
    
    return {
        'individual_predictions': individual_predictions,
        'agreement_percentage': np.array(agreement_percentages),
        'vote_counts': vote_counts,
        'majority_class': np.array(majority_classes),
        'total_models': len(all_predictions)
    }


def interpret_agreement(agreement_pct):

    if agreement_pct >= 80:
        return "STRONG"
    elif agreement_pct >= 65:
        return "MODERATE"
    elif agreement_pct >= 50:
        return "WEAK"
    else:
        return "CONFLICTING"


def should_skip_trade(agreement_pct, min_agreement=75.0):

    return agreement_pct < min_agreement


# Test function
if __name__ == "__main__":
    print("="*70)
    print("MODEL AGREEMENT CALCULATOR TEST")
    print("="*70)
    
    # Simulate 5 models making predictions on 3 stocks
    # 0=SELL, 1=HOLD, 2=BUY
    
    print("\nTest Case 1: Strong Agreement (4/5 models say BUY)")
    print("-" * 70)
    
    # Create fake models
    class FakeModel:
        def __init__(self, predictions):
            self.predictions = predictions
        def predict(self, X):
            return self.predictions
    
    models = {
        'model1': FakeModel(np.array([2, 0, 2])),  # BUY, SELL, BUY
        'model2': FakeModel(np.array([2, 0, 2])),  # BUY, SELL, BUY
        'model3': FakeModel(np.array([2, 1, 2])),  # BUY, HOLD, BUY
        'model4': FakeModel(np.array([2, 0, 2])),  # BUY, SELL, BUY
        'model5': FakeModel(np.array([1, 0, 2])),  # HOLD, SELL, BUY
    }
    
    # Fake data (3 stocks)
    X_test = pd.DataFrame({'feature1': [1, 2, 3]})
    
    result = calculate_model_agreement(models, X_test)
    
    print(f"Total models: {result['total_models']}")
    print(f"Agreement for each stock:")
    for i in range(3):
        agreement = result['agreement_percentage'][i]
        majority = result['majority_class'][i]
        rating = interpret_agreement(agreement)
        skip = "SKIP" if should_skip_trade(agreement) else "TRADE"
        
        signal = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}[majority]
        
        print(f"  Stock {i+1}: {signal} | Agreement: {agreement:.0f}% | Rating: {rating} | Decision: {skip}")
    
    print("\n" + "="*70)
    print("✅ If you see results above, the calculator works!")
    print("="*70)
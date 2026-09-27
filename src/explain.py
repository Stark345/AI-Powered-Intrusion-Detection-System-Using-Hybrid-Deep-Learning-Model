"""
Explainable AI (XAI) Module: SHAP & LIME Interpretability
Translates black-box neural network decisions into transparent cybersecurity intelligence.
"""
import numpy as np


def explain_sample_lime(model, sample_vector, feature_names, class_names=['Normal', 'Attack'], num_features=6):
    """
    Generates a localized surrogate explanation using LIME.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer

        def predict_proba(x):
            preds = model.predict(x.reshape(x.shape[0], x.shape[1], 1), verbose=0)
            return np.hstack([1.0 - preds, preds])

        # Dummy background distribution for single sample explanation
        bg = np.tile(sample_vector, (100, 1)) + np.random.normal(0, 0.05, (100, len(sample_vector)))

        explainer = LimeTabularExplainer(
            training_data=bg,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification'
        )

        exp = explainer.explain_instance(sample_vector, predict_proba, num_features=num_features)
        return exp.as_list()
    except Exception as e:
        return [("Feature Contribution", f"Explainability unavailable: {e}")]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve,auc, confusion_matrix

class BusinessModelEvaluator:
    """
    A class to perform a comprehensive business-focused evaluation of a model,
    including PR AUC, cost-sensitive analysis, and threshold optimization.
    """
    
    def __init__(self, y_true, y_pred_proba, model_name):
        self.y_true = y_true
        self.y_pred_proba = y_pred_proba
        self.model_name = model_name
        self.precision, self.recall, self.thresholds = precision_recall_curve(y_true, y_pred_proba)
        self.pr_auc = auc(self.recall, self.precision)
        

    def plot_precision_recall_curve(self):
        """Calculates and plots the Precision-Recall curve and PR AUC."""
        
        plt.figure(figsize=(10, 7))
        plt.plot(self.recall, self.precision, label=f'{self.model_name} (PR AUC = {self.pr_auc:.2f})', color='b')
        plt.xlabel('Recall (Coverage of Actual Churners)')
        plt.ylabel('Precision (Campaign Efficiency)')
        plt.title(f'Precision-Recall Curve for {self.model_name}')
        plt.legend()
        plt.grid(True)
        plt.show()
        print(f"Precision-Recall AUC for {self.model_name}: {self.pr_auc:.4f}")
        
    def cost_sensitive_analysis(self, cost_fp, cost_fn, threshold=0.5):
        """Performs a cost-sensitive analysis at a given threshold."""
        
        y_pred = (self.y_pred_proba >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        
        print(f"\n--- Cost-Sensitive Analysis at Threshold = {threshold} ---")
        print(f"False Positives (FP): {fp} -> Cost: ${fp * cost_fp:,.2f}")
        print(f"False Negatives (FN): {fn} -> Cost: ${fn * cost_fn:,.2f}")
        print(f"Total Business Cost: ${total_cost:,.2f}")
        return total_cost
    
    
    def find_optimal_threshold(self, cost_fp, cost_fn):
        """Finds the optimal prediction threshold to minimize business cost."""
        
        
        thresholds_with_end = np.append(self.thresholds, 1.0)
        costs = []
        for thresh in thresholds_with_end:
            y_pred_thresh = (self.y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred_thresh).ravel()
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            costs.append(total_cost)

        optimal_idx = np.argmin(costs)
        optimal_threshold = thresholds_with_end[optimal_idx]
        min_cost = costs[optimal_idx]

        print("\n--- Threshold Optimization ---")
        print(f"Optimal Threshold for Minimum Business Cost: {optimal_threshold:.4f}")
        print(f"Minimum Achievable Business Cost: ${min_cost:,.2f}")
        return optimal_threshold, min_cost
    
    
    
    
# --- Example Usage (you would run this in your notebook) ---
#
# # 1. Define your business costs
# COST_FP = 50
# COST_FN = 1000
#
# # 2. Get the prediction probabilities for your model
# y_probs_lr = model_lr.predict_proba(X_test_processed)[:, 1]
#
# # 3. Create an instance of the evaluator
# lr_evaluator = ModelEvaluator(
#     y_true=y_test,
#     y_pred_proba=y_probs_lr,
#     model_name="Logistic Regression"
# )
#
# # 4. Call each evaluation method separately
# lr_evaluator.plot_precision_recall_curve()
# lr_evaluator.cost_sensitive_analysis(cost_fp=COST_FP, cost_fn=COST_FN) # Cost at default 0.5 threshold
# optimal_thresh, min_cost = lr_evaluator.find_optimal_threshold(cost_fp=COST_FP, cost_fn=COST_FN)
# lr_evaluator.cost_sensitive_analysis(cost_fp=COST_FP, cost_fn=COST_FN, threshold=optimal_thresh) # Cost at optimal threshold

        
        
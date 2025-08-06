import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Custom transformer to clean the TotalCharges column
class TotalChargesCleaner(BaseEstimator, TransformerMixin):
    """
    This transformer converts the 'TotalCharges' column to a numeric type
    and fills any resulting NaN values with 0.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):

        X_copy = X.copy()
        X_copy['TotalCharges'] = pd.to_numeric(X_copy['TotalCharges'], errors='coerce')
        X_copy['TotalCharges'].fillna(0, inplace=True)
        
        return X_copy
    
    
# Custom transformer to flag new customers based on tenure
class NewCustomerFlagger(BaseEstimator, TransformerMixin):
    """
    This transformer creates a binary feature 'is_new_customer', 
    which is 1 if tenure is 0, and 0 otherwise.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['is_new_customer'] = (X_copy['tenure'] == 0).astype(int)
        
        return X_copy


# Custom transformer to create tenure segments
class TenureSegmenter(BaseEstimator, TransformerMixin):
    """
    This transformer creates a new categorical feature 'TenureSegment'
    by binning the 'tenure' column into 'New', 'Mid', and 'Loyal' categories.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        

        def tenure_segment(months):
            if months < 12:
                return 'New'
            elif 12 <= months <= 24:
                return 'Mid'
            else:
                return 'Loyal'

        X_copy['TenureSegment'] = X_copy['tenure'].apply(tenure_segment)
        
        return X_copy


# Custom transformer to classify service usage
class ServiceUsageClassifier(BaseEstimator, TransformerMixin):
    """
    This transformer creates a new categorical feature 'ServiceUsage'
    classifying customers as 'Heavy' or 'Light' users based on their
    adoption of add-on services.
    """
    def __init__(self):
        self.addon_services = [
            'StreamingTV', 'StreamingMovies', 'OnlineSecurity', 
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'MultipleLines'
        ]

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()

        def classify_service_usage(row):
            has_internet = row.get('InternetService', 'No') != 'No'
            uses_addon = any(row.get(service) == 'Yes' for service in self.addon_services)
            
            if has_internet and uses_addon:
                return 'Heavy'
            else:
                return 'Light'

        X_copy['ServiceUsage'] = X_copy.apply(classify_service_usage, axis=1)
        
        return X_copy


# Custom transformer to create the BillingLevel feature
class BillingLevelCreator(BaseEstimator, TransformerMixin):
    """
    This transformer creates a new categorical feature 'BillingLevel'
    by comparing the 'MonthlyCharges' to a median value.

    Crucially, the median is calculated only from the training data
    during the .fit() step to prevent data leakage.
    """
    def __init__(self):
        self.median = 0

    def fit(self, X, y=None):
        self.median = X['MonthlyCharges'].median()
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['BillingLevel'] = X_copy['MonthlyCharges'].apply(
            lambda x: 'High' if x > self.median else 'Low'
        )
        
        return X_copy


# Custom transformer to create the BillingTenureSegment feature
class BillingTenureSegmenter(BaseEstimator, TransformerMixin):
    """
    This transformer creates a new interaction feature 'BillingTenureSegment'
    by combining the 'BillingLevel' and 'TenureSegment' features.
    
    IMPORTANT: This transformer must be placed in the pipeline *after*
    the BillingLevelCreator and TenureSegmenter have run.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        X_copy['BillingLevel'] = X_copy['BillingLevel'].astype(str)
        X_copy['TenureSegment'] = X_copy['TenureSegment'].astype(str)
        X_copy['BillingTenureSegment'] = X_copy['BillingLevel'] + '-' + X_copy['TenureSegment']
        
        return X_copy
    
    
# Custom transformer to calculate the EngagementScore
class EngagementScorer(BaseEstimator, TransformerMixin):
    """
    Calculates a numerical EngagementScore based on a customer's
    service adoption.
    """
    def __init__(self):
        self.engagement_features = [
            'PhoneService', 'MultipleLines', 'InternetService', 
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        self.engagement_map = {
            'Yes': 1, 'No': 0, 'No phone service': 0,
            'No internet service': 0, 'DSL': 1, 'Fiber optic': 1
        }

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['EngagementScore'] = X_copy[self.engagement_features].replace(self.engagement_map).sum(axis=1)
        return X_copy


# Custom transformer to create EngagementSegment from EngagementScore
class EngagementSegmenter(BaseEstimator, TransformerMixin):
    """
    Creates a categorical 'EngagementSegment' by binning the
    'EngagementScore' into 'LowEngage', 'MidEngage', and 'HighEngage'.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
  
        def bin_engagement(score):
            if score <= 3:
                return 'LowEngage'
            elif score <= 6:
                return 'MidEngage'
            else:
                return 'HighEngage'
        
        X_copy['EngagementSegment'] = X_copy['EngagementScore'].apply(bin_engagement)
        return X_copy


# Custom transformer to create the BillingEngageSegment feature
class BillingEngageSegmenter(BaseEstimator, TransformerMixin):
    """
    Creates a final interaction feature 'BillingEngageSegment' by
    combining the 'BillingLevel' and 'EngagementSegment' features.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        X_copy['BillingLevel'] = X_copy['BillingLevel'].astype(str)
        X_copy['EngagementSegment'] = X_copy['EngagementSegment'].astype(str)
        X_copy['BillingEngageSegment'] = X_copy['BillingLevel'].str.replace(' ', '') + '-' + X_copy['EngagementSegment']
        return X_copy


# Custom transformer to create the TenureEngageSegment feature
class TenureEngageSegmenter(BaseEstimator, TransformerMixin):
    """
    Creates a final interaction feature 'TenureEngageSegment' by
    combining the 'TenureSegment' and 'EngagementSegment' features.
    
    IMPORTANT: This transformer must be placed in the pipeline *after*
    the TenureSegmenter and EngagementSegmenter have run.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Ensure the required columns are strings before concatenating
        X_copy['TenureSegment'] = X_copy['TenureSegment'].astype(str)
        X_copy['EngagementSegment'] = X_copy['EngagementSegment'].astype(str)
        
        # Create the new interaction feature
        X_copy['TenureEngageSegment'] = X_copy['TenureSegment'] + '-' + X_copy['EngagementSegment']
        
        return X_copy

# Custom transformer to create the IsMonthToMonth binary flag
class MonthToMonthFlagger(BaseEstimator, TransformerMixin):
    """
    This transformer creates a binary feature 'IsMonthToMonth', which is 1
    if the Contract is 'Month-to-month', and 0 otherwise.
    """
    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything from the data.
        return self
    
    def transform(self, X):
        # Make a copy to avoid changing the original DataFrame
        X_copy = X.copy()
        
        # Create the new binary feature
        X_copy['IsMonthToMonth'] = (X_copy['Contract'] == 'Month-to-month').astype(int)
        
        return X_copy
    
    
# Custom transformer to create the HasCoreProtection binary flag
class CoreProtectionFlagger(BaseEstimator, TransformerMixin):
    """
    This transformer creates a binary feature 'HasCoreProtection', which is 1
    if the customer has either OnlineSecurity or TechSupport, and 0 otherwise.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['HasCoreProtection'] = ((X_copy['OnlineSecurity'] == 'Yes') | 
                                     (X_copy['TechSupport'] == 'Yes')).astype(int)
        return X_copy
    
    
# Custom transformer to create the HighRiskFinancialProfile binary flag
class HighRiskFinancialProfileFlagger(BaseEstimator, TransformerMixin):
    """
    This transformer creates a binary feature 'HighRiskFinancialProfile', 
    which is 1 if the customer has both a 'Month-to-month' contract
    and uses 'Electronic check' as their payment method, and 0 otherwise.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['HighRiskFinancialProfile'] = ((X_copy['Contract'] == 'Month-to-month') & 
                                              (X_copy['PaymentMethod'] == 'Electronic check')).astype(int)
        
        return X_copy
    
    
# Custom transformer to calculate the OverallRiskScore
class OverallRiskScorer(BaseEstimator, TransformerMixin):
    """
    Calculates a numerical 'OverallRiskScore' by summing up several
    binary high-risk indicators.
    
    IMPORTANT: This transformer must be placed in the pipeline *after*
    the CoreProtectionFlagger (which creates 'HasCoreProtection') has run.
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        risk_score = (
            (X_copy['SeniorCitizen'] == 1).astype(int) +
            (X_copy['Dependents'] == 'No').astype(int) +
            (X_copy['Contract'] == 'Month-to-month').astype(int) +
            (X_copy['InternetService'] == 'Fiber optic').astype(int) +
            (X_copy['HasCoreProtection'] == 0).astype(int)
        )
        
        X_copy['OverallRiskScore'] = risk_score
        
        return X_copy
    

class KMeansClusterer(BaseEstimator, TransformerMixin):
    """
    A custom transformer that creates a 'Cluster' feature by running
    K-Means on a set of behavioral features.
    
    Crucially, it learns the cluster centers only from the training data
    to prevent data leakage.
    """
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.preprocessor = None
        self.behavioral_features = [
            'tenure', 'MonthlyCharges', 'Contract', 'PaymentMethod', 
            'PaperlessBilling', 'InternetService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

    def fit(self, X, y=None):
        # Select only the behavioral features for clustering
        X_behavioral = X[self.behavioral_features]
        
        # Define preprocessing steps specifically for these features
        categorical_cols = X_behavioral.select_dtypes(include=['object', 'category']).columns
        numerical_cols = X_behavioral.select_dtypes(include=np.number).columns
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Preprocess the training data
        X_processed = self.preprocessor.fit_transform(X_behavioral)
        
        # Fit the K-Means algorithm on the processed training data
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        self.kmeans.fit(X_processed)
        
        return self
    
    def transform(self, X):
        # Make a copy to avoid changing the original DataFrame
        X_copy = X.copy()
        
        # Select the behavioral features from the incoming data
        X_behavioral = X_copy[self.behavioral_features]
        
        # Preprocess the data using the *already fitted* preprocessor
        X_processed = self.preprocessor.transform(X_behavioral)
        
        # Predict the cluster labels using the *already fitted* KMeans model
        cluster_labels = self.kmeans.predict(X_processed)
        
        # Add the new 'Cluster' feature to the DataFrame
        X_copy['Cluster'] = cluster_labels
        
        return X_copy

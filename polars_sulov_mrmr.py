import numpy as np
np.random.seed(42)
import polars as pl
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from xgboost import XGBClassifier, XGBRegressor
from typing import List, Dict
from itertools import combinations
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import copy
from collections import Counter
import pdb

class Sulov_MRMR(BaseEstimator, TransformerMixin): # Class name 
    def __init__(self, corr_threshold: float = 0.7, ## minimum 0.7 is recommended
                 model_type = 'classification', # optional 'regression'
                 verbose: int = 0):
        self.corr_threshold = corr_threshold
        self.model_type = model_type
        self.verbose = verbose
        self.selected_features = []
        self.target = None
        self.min_features = 2
        self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.fitted_ = False
        self.random_state = 12

    def sulov_mrmr(self, X: pl.DataFrame, y: pl.Series) -> List[str]:
        """
        Complete SULOV-MRMR implementation with:
        - Mixed data type handling
        - Adaptive correlation thresholding
        - Diagnostic logging
        - Minimum feature count enforcement
        """
        # Initialize diagnostics
        if self.verbose > 0:
            print("\n" + "="*40)
            print("Polars Featurewiz SULOV-MRMR Feature Selection Algorithm")
            print(f"Initial Features: {len(X.columns)}")
            print("="*40)

        # Separate numeric and categorical features
        numeric_cols = X.select(pl.col(pl.NUMERIC_DTYPES)).columns
        cat_cols = X.select(pl.col(pl.String, pl.Categorical)).columns
        features = numeric_cols + cat_cols
        
        # Handle empty dataset edge case
        if not features:
            raise ValueError("Input 'X' must be a Polars DataFrame and have features.")

        # 1. Calculate Mutual Information Scores
        if cat_cols and not self.fitted_:
            self.encoder.fit(X[features])
        mis_scores = self._calculate_mis(X, y, features, cat_cols, self.encoder)
        if self.verbose > 1:
            self._log_mis_scores(mis_scores)

        # 2. Calculate Feature Correlations
        corr_pairs = self._calculate_correlations(X, features, numeric_cols, cat_cols)
        if self.verbose > 1:
            self._log_correlations(corr_pairs)

        # 3. Adaptive Feature Removal (revised)
        features_to_remove = self._adaptive_removal(corr_pairs, mis_scores)
        if self.verbose > 1:
            self._log_removals(features_to_remove)

        # 4. Final Selection with enhanced thresholds
        remaining = [f for f in features if f not in features_to_remove]
        selected = self._enforce_min_features(remaining, mis_scores)
        
        if self.verbose > 0:
            print(f"SULOV removed Features ({len(features_to_remove)}): {', '.join(features_to_remove)}")
        
        return selected

    def _calculate_mis(self, X: pl.DataFrame, y: pl.Series, features: List[str], 
                      cat_cols: List[str], encoder) -> Dict[str, float]:
        """Calculate Mutual Information Scores with proper encoding"""
        if cat_cols:
            X_encoded = encoder.transform(X[features].to_pandas())
            discrete_mask = [f in cat_cols for f in features]
        else:
            X_encoded = X[features].to_pandas()
            discrete_mask = [False]*len(features)

        y = y.to_numpy()

        if self.model_type == 'classification':
            mis_values = mutual_info_classif(
                X_encoded, y, 
                discrete_features=discrete_mask,
                random_state=self.random_state
            )
        else:
            mis_values = mutual_info_regression(
                X_encoded, y, 
                discrete_features=discrete_mask,
                random_state=self.random_state
            )
                        
        return dict(zip(features, mis_values))

    def _calculate_correlations(self, X: pl.DataFrame, features: List[str],
                               numeric_cols: List[str], cat_cols: List[str]) -> List[tuple]:
        """Calculate correlations with appropriate metrics"""
        corr_pairs = []
        for f1, f2 in combinations(features, 2):
            if f1 in numeric_cols and f2 in numeric_cols:
                corr = X[[f1, f2]].corr().row(0)[1]
            else:
                # Cramer's V for categorical pairs
                confusion = X[[f1, f2]].pivot(
                    values=f2, 
                    index=f1, 
                    aggregate_fn=pl.len()
                ).fill_null(0)
                chi2, _, _, _ = chi2_contingency(confusion.to_numpy())
                n = X.height
                phi2 = chi2 / n
                r, c = confusion.shape
                corr = np.sqrt(phi2 / min((r-1), (c-1)))
                
            corr_pairs.append((f1, f2, abs(corr)))
                                
        return corr_pairs

    def _adaptive_removal(self, corr_pairs: List[tuple], mis_scores: Dict[str, float]) -> set:
        """Remove only features that are both:
        1. Correlated with better alternatives
        2. Have MIS < 50% of max feature's score
        """
        removal_candidates = defaultdict(int)
        max_mis = max(mis_scores.values())
        
        for f1, f2, corr in corr_pairs:
            if corr > self.corr_threshold:
                # Only consider removal if one feature is significantly better
                ratio = mis_scores[f1] / (mis_scores[f2] + 1e-9)
                if ratio < 0.7:  # f2 is at least 30% better
                    removal_candidates[f1] += 1
                elif ratio > 1.4:  # f1 is at least 40% better
                    removal_candidates[f2] += 1

        return {f for f, count in removal_candidates.items() 
            if mis_scores[f] < 0.5 * max_mis}

    def _enforce_min_features(self, remaining: List[str], mis_scores: Dict[str, float]) -> List[str]:
        """Adaptive thresholding based on MIS distribution"""
        scores = np.array(list(mis_scores.values()))
        q75 = np.percentile(scores[scores > 0], 75)
        min_score = max(0.1 * q75, 0.01)  # At least 10% of 75th percentile
        
        filtered = [f for f in remaining if mis_scores[f] >= min_score]
        
        # Ensure minimum features with fallback
        if len(filtered) < self.min_features:
            filtered = sorted(remaining, key=lambda x: -mis_scores[x])
        
        return filtered[:max(self.min_features, len(filtered))]

    def recursive_xgboost(self, X: pl.DataFrame, y: pl.Series) -> List[str]:
        """Stabilized recursive feature selection with consistent performance"""
        from kneed import KneeLocator
        # Initialize with sorted features for consistency
        sorted_features = sorted(X.columns)
        total_features = len(sorted_features)
        feature_votes = defaultdict(int)
        
            # Create importance tiers based on initial full model
        full_model = self._get_xgboost_model(y)
        full_model.fit(X, y)
        base_importances = full_model.feature_importances_
        tier_thresholds = np.percentile(base_importances, [10, 90])

        if self.verbose:
            print('base importances: ', base_importances, 'tier_thresholds[1]', tier_thresholds[1])
            
        # Stratify features into importance tiers
        tiers = {
            'high': [f for f, imp in zip(sorted_features, base_importances) 
                    if imp >= tier_thresholds[1]],
            'medium': [f for f, imp in zip(sorted_features, base_importances) 
                    if tier_thresholds[0] <= imp < tier_thresholds[1]],
            'low': [f for f, imp in zip(sorted_features, base_importances) 
                    if imp < tier_thresholds[0]]
        }
        initial_features = tiers['high']

        # Dynamic configuration based on feature count
        iter_limit = max(3, int(total_features/5))  # set this number high to avoid selecting unimportant features
        top_ratio = 0.05 if total_features > 50 else 0.1
        top_num = max(2, int(total_features * top_ratio)) ## set this number low to avoid selecting too many
        
        # Fixed model parameters for consistency
        base_params = {
            'n_estimators': 100,
            'max_depth': 4,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        if self.verbose > 0:
            print('iter limit', iter_limit)
        # Chunked feature processing
        for i in range(0, total_features, iter_limit):

            if self.verbose:
                print('iteration #', i)

            chunk_features = sorted_features[i:i+iter_limit]
            #print('chunk features = ', chunk_features)
            if len(chunk_features) < 2: continue
            
            # Train on feature chunk
            model = self._get_xgboost_model(y)
            model.set_params(**base_params)
            model.fit(X[chunk_features], y)
            
            # Get normalized importances
            importances = pd.Series(model.feature_importances_, index=chunk_features)
            max_imp = importances.max()
            if max_imp == 0.05: continue  # Skip chunks with no importance
            if self.verbose > 0:
                print('feature importances: \n', importances)

            # Select features with importance >50% of max
            threshold = max_imp * 0.80 ### keep this as high as possible to avoid unnecessary features
            selected = importances[importances >= threshold].index.tolist()
            
            # Fallback to top N features if threshold too strict
            if self.verbose > 0:
                print('selected after importance: ', selected)
            
            # Update votes with exponential weighting (later chunks matter more)
            weight = 1 + (i/(total_features*2))  # 1x to 1.5x weight
            for feat in selected:
                feature_votes[feat] += weight
        
        # Dynamic cutoff using knee detection
        votes = pd.Series(feature_votes).sort_values(ascending=False)
        
        if self.verbose > 0:
            print('final votes per feature:\n', votes)
        if len(votes) > 10:
            # Find natural cutoff point using knee detection
            kneedle = KneeLocator(range(len(votes)), votes.values, curve='convex')
            cutoff = kneedle.knee or len(votes)//2
        else:
            cutoff = len(votes)
        
        # Ensure minimum feature retention
        min_features = max(2, int(total_features * 0.1))
        if len(votes) <= 1:
            final_features =  initial_features
        else:
            final_features = votes.index[:max(cutoff, min_features)].tolist()
            final_features.reverse()
            if self.verbose:
                print(f'final features:\n{final_features}')
            ### This final next step is going to boost your selectiion further with high tier features
            if self.verbose:
                print(f'initial features combined with final features: \n{initial_features} to boost performance')
            final_features = list(np.unique(initial_features + final_features))
        
        return final_features


    def _get_upper_triangle(self, corr_matrix: pl.DataFrame) -> pl.DataFrame:
        """Extract upper triangle pairs from Polars correlation matrix"""
        # Get feature names and their indices
        features_list = corr_matrix.columns
        
        # Add feature names as a column to correlation matrix
        corr_with_names = corr_matrix.with_columns(
            feature_a=pl.Series(features_list)
        )
        
        # Melt to long format (feature_a, feature_b, correlation)
        long_format = corr_with_names.melt(
            id_vars="feature_a",
            variable_name="feature_b",
            value_name="correlation"
        )
        
        # Create index-based comparison for upper triangle
        return long_format.with_columns(
            pl.col("feature_a").map_elements(
                lambda x: features_list.index(x)
            ).alias("idx_a"),
            pl.col("feature_b").map_elements(
                lambda x: features_list.index(x)
            ).alias("idx_b"),
        ).filter(
            pl.col("idx_a") < pl.col("idx_b")
        ).drop(["idx_a", "idx_b"])

    def _coerce_datasets(self, X, y):
        """Coerce datasets X and y into Polars dataframes and series."""
        if not isinstance(X, pl.DataFrame):
            if type(X) == tuple:
                ### In some cases, X comes in as a Tuple and you need to split it
                ### You can ignore the y in this case
                X_pl, y = X
            elif isinstance(y, np.ndarray):
                print("Input 'X' is a numpy array. It must be a Polars DataFrame. Returning as-is...")
                return (X, y)
            else:
                X_pl = pl.from_pandas(X)
        else:
            X_pl = X

        if not isinstance(y, pl.DataFrame):
            if isinstance(y, pl.Series):
                y_pl = y
            elif isinstance(y, np.ndarray):
                y_pl = pl.DataFrame(y, name=self.target)
            else:
                y_pl = pl.Series(y)
                
        return X_pl, y_pl

    def _check_pandas(self, X):
        if isinstance(X, pd.DataFrame):
            return pl.from_pandas(X)
        else:
            return X

    def _get_xgboost_model(self, y: np.ndarray):
        """Get appropriate XGBoost model based on target type"""
        if self.model_type == 'classification':
            return XGBClassifier(n_estimators=100, random_state=self.random_state)
        return XGBRegressor(n_estimators=100, random_state=self.random_state)

    def fit(self, X: pl.DataFrame, y: pl.Series) -> pl.DataFrame:
        """
        Fits the PolarsFeaturewiz to the data.
        Optimized feature selection pipeline.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)
            y (Polars Series or array-like): Target vector of shape (n_samples,)

        Returns:
            self
        """
        ## check data
        if self.model_type == 'regression':
           print('Model type: Regression')
        else:
            print('Model type: Classification')
        X, y = self._coerce_datasets(X, y)
        self.target = y.name
        self.min_features = max(2, int(0.8*len(X.columns)))

        # Step 1: SULOV-MRMR
        sulov_features = self.sulov_mrmr(X, y)
        print(f'SULOV selected Features ({len(sulov_features)}): {sulov_features}')
        
        # Step 2: Recursive XGBoost with expanded features
        if len(sulov_features) > self.min_features:
            self.selected_features = self.recursive_xgboost(
                X.select(sulov_features), y)
        else:
            self.selected_features = sulov_features
        
        print(f'\nRecursive XGBoost selected Features ({len(self.selected_features)}): {self.selected_features}')
        self.fitted_ = True

        return self

    def transform(self, X, y=None):
        """
        Transforms the data by selecting the top MRMR features in Polars DataFrame.

        Args:
            X (Polars DataFrame): Feature DataFrame of shape (n_samples, n_features)

        Returns:
            Polars DataFrame: Transformed DataFrame with selected features, shape (n_samples, n_selected_features)
        """
        check_is_fitted(self, 'fitted_')
        X, y = self._coerce_datasets(X, y)
        if len(y) == 0:
            ## in some cases, y comes back as a Null series in Polars. So you have to ignore it!
            return X[self.selected_features]
        if y is None:
            return X[self.selected_features] # Select columns by names in Polars
        else:
            return X[self.selected_features], y # Select columns by names in Polars

    def _log_mis_scores(self, mis_scores: dict):
        """Log Mutual Information Scores"""
        print("\nMutual Information Scores:")
        for feat, score in sorted(mis_scores.items(), key=lambda x: -x[1]):
            print(f"    {feat}: {score:.4f}")

    def _log_correlations(self, corr_pairs: list):
        """Log correlation pairs above threshold"""
        print("\nHigh Correlation Pairs (correlation > threshold):")
        for f1, f2, corr in corr_pairs:
            if corr >= self.corr_threshold:
                print(f"    {f1} vs {f2}: {corr:.4f}")

    def _log_removals(self, features_to_remove: set):
        """Log features being removed"""
        if features_to_remove:
            print("\nFeatures removed due to correlation:")
            print(", ".join(features_to_remove))
        else:
            print("\nNo features removed for correlation")

    def get_feature_names_out(self):
        return self.selected_features

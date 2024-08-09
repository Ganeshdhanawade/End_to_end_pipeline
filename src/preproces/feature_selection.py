# feature_selection.py
import pandas as pd
from scipy.stats import pearsonr, kendalltau, spearmanr
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

class FeatureSelector:
    def __init__(self, corr_threshold=0.5, multicoll_threshold=0.9, top_features=20):
        self.corr_threshold = corr_threshold
        self.multicoll_threshold = multicoll_threshold
        self.top_features = top_features

    def select_corr_features(self, X, y):
        selected_features = []
        corr_matrix = X.corr()

        for col in X.columns:
            x = X[col]
            # Pearson correlation for linear relationship
            pearson_corr, _ = pearsonr(x, y)
            if abs(pearson_corr) >= self.corr_threshold:
                selected_features.append(col)
            else:
                # Kendall's Tau for non-linear relationship
                _, kendall_p_value = kendalltau(x, y)
                if kendall_p_value < 0.05:
                    # Spearman's Rank Correlation
                    spearman_corr, _ = spearmanr(x, y)
                    if abs(spearman_corr) >= self.corr_threshold:
                        selected_features.append(col)

        # Handle multicollinearity
        corr_matrix = corr_matrix.abs()
        to_remove = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.multicoll_threshold:
                    colname = corr_matrix.columns[i]
                    to_remove.add(colname)

        final_features = [col for col in selected_features if col not in to_remove]
        
        return final_features

    def select_top_features(self, X, y):
        X_encoded = pd.get_dummies(X, drop_first=True)
        
        selected_features_dict = {
            'anova': [],
            'mutual_info': [],
            'extra_trees': [],
            'random_forest': []
        }
        
        # ANOVA F-test
        f_scores, p_values = f_regression(X_encoded, y)
        anova_scores_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'F-Score': f_scores
        }).sort_values(by='F-Score', ascending=False)
        selected_features_dict['anova'] = anova_scores_df.head(self.top_features)['Feature'].tolist()
        
        # Mutual Information
        mi_scores = mutual_info_regression(X_encoded, y)
        mi_scores_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'MI Score': mi_scores
        }).sort_values(by='MI Score', ascending=False)
        selected_features_dict['mutual_info'] = mi_scores_df.head(self.top_features)['Feature'].tolist()
        
        # ExtraTreesRegressor Feature Importance
        model = ExtraTreesRegressor()
        model.fit(X_encoded, y)
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        selected_features_dict['extra_trees'] = feature_importance_df.head(self.top_features)['Feature'].tolist()
        
        # RandomForestRegressor Feature Importance
        model_rf = RandomForestRegressor()
        model_rf.fit(X_encoded, y)
        importances_rf = model_rf.feature_importances_
        feature_importance_rf_df = pd.DataFrame({
            'Feature': X_encoded.columns,
            'Importance': importances_rf
        }).sort_values(by='Importance', ascending=False)
        selected_features_dict['random_forest'] = feature_importance_rf_df.head(self.top_features)['Feature'].tolist()
        
        # Combine features from all methods (union of sets)
        combined_features = set(selected_features_dict['anova']) | set(selected_features_dict['mutual_info']) | \
                            set(selected_features_dict['extra_trees']) | set(selected_features_dict['random_forest'])
        
        return list(combined_features)

    def save_selected_features(self, X, y):
        corr_feature = self.select_corr_features(X,y)
        selected_features = self.select_top_features(X, y)
        combined_features = list(set(corr_feature) | set(selected_features))
        X_encoded = pd.get_dummies(X, drop_first=True)
        selected_features_df = X_encoded[combined_features]
        selected_features_df['res']=y
        return selected_features_df

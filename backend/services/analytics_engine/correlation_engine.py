"""
Cross-Domain Correlation Engine
Identifies relationships and dependencies between different metrics and domains
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import networkx as nx
import structlog

logger = structlog.get_logger(__name__)

class CorrelationEngine:
    """
    Analyzes correlations between different domains and metrics
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.graph = nx.Graph()
        self.scaler = StandardScaler()
        
    async def calculate_correlation_matrix(self,
                                          data: pd.DataFrame,
                                          method: str = 'pearson') -> np.ndarray:
        """
        Calculate correlation matrix for multiple metrics
        
        Args:
            data: DataFrame with metrics as columns
            method: Correlation method (pearson, spearman, kendall)
            
        Returns:
            Correlation matrix
        """
        
        if method == 'pearson':
            correlation_matrix = data.corr(method='pearson')
        elif method == 'spearman':
            correlation_matrix = data.corr(method='spearman')
        elif method == 'kendall':
            correlation_matrix = data.corr(method='kendall')
        else:
            raise ValueError(f"Unsupported correlation method: {method}")
            
        return correlation_matrix.values
        
    async def find_strong_correlations(self,
                                     data: pd.DataFrame,
                                     threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find strongly correlated metric pairs
        
        Args:
            data: DataFrame with metrics
            threshold: Correlation threshold
            
        Returns:
            List of strong correlations
        """
        
        correlation_matrix = await self.calculate_correlation_matrix(data)
        strong_correlations = []
        
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                correlation = correlation_matrix[i, j]
                
                if abs(correlation) >= threshold:
                    strong_correlations.append({
                        'metric1': data.columns[i],
                        'metric2': data.columns[j],
                        'correlation': float(correlation),
                        'type': 'positive' if correlation > 0 else 'negative',
                        'strength': self._classify_correlation_strength(correlation),
                        'p_value': self._calculate_p_value(
                            data.iloc[:, i],
                            data.iloc[:, j]
                        )
                    })
                    
        return sorted(strong_correlations, key=lambda x: abs(x['correlation']), reverse=True)
        
    async def analyze_causal_relationships(self,
                                         data: pd.DataFrame,
                                         target_metric: str) -> List[Dict[str, Any]]:
        """
        Analyze potential causal relationships using Granger causality
        
        Args:
            data: Time series data
            target_metric: Target metric to analyze
            
        Returns:
            Potential causal relationships
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        causal_relationships = []
        
        for column in data.columns:
            if column != target_metric:
                try:
                    # Prepare data for Granger test
                    test_data = pd.DataFrame({
                        'target': data[target_metric],
                        'predictor': data[column]
                    }).dropna()
                    
                    # Run Granger causality test
                    max_lag = min(10, len(test_data) // 5)
                    results = grangercausalitytests(
                        test_data[['target', 'predictor']],
                        maxlag=max_lag,
                        verbose=False
                    )
                    
                    # Find best lag with significant p-value
                    best_lag = None
                    best_p_value = 1.0
                    
                    for lag in range(1, max_lag + 1):
                        p_value = results[lag][0]['ssr_ftest'][1]
                        if p_value < best_p_value:
                            best_p_value = p_value
                            best_lag = lag
                            
                    if best_p_value < 0.05:  # Significant at 5% level
                        causal_relationships.append({
                            'predictor': column,
                            'target': target_metric,
                            'lag': best_lag,
                            'p_value': float(best_p_value),
                            'significance': 'significant' if best_p_value < 0.01 else 'moderate'
                        })
                        
                except Exception as e:
                    logger.warning(f"Granger test failed for {column}: {e}")
                    
        return causal_relationships
        
    async def detect_correlation_clusters(self,
                                        data: pd.DataFrame,
                                        eps: float = 0.3) -> List[List[str]]:
        """
        Detect clusters of correlated metrics using DBSCAN
        
        Args:
            data: DataFrame with metrics
            eps: Maximum distance for clustering
            
        Returns:
            Clusters of correlated metrics
        """
        
        # Calculate correlation distance matrix
        correlation_matrix = await self.calculate_correlation_matrix(data)
        distance_matrix = 1 - np.abs(correlation_matrix)
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=eps, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        # Extract clusters
        clusters = []
        for label in set(labels):
            if label != -1:  # Skip noise points
                cluster_indices = np.where(labels == label)[0]
                cluster_metrics = [data.columns[i] for i in cluster_indices]
                clusters.append(cluster_metrics)
                
        return clusters
        
    async def build_correlation_network(self,
                                       data: pd.DataFrame,
                                       threshold: float = 0.5) -> nx.Graph:
        """
        Build a network graph of metric correlations
        
        Args:
            data: DataFrame with metrics
            threshold: Minimum correlation for edge creation
            
        Returns:
            NetworkX graph
        """
        
        self.graph.clear()
        correlation_matrix = await self.calculate_correlation_matrix(data)
        
        # Add nodes
        for column in data.columns:
            self.graph.add_node(column)
            
        # Add edges for significant correlations
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                correlation = correlation_matrix[i, j]
                
                if abs(correlation) >= threshold:
                    self.graph.add_edge(
                        data.columns[i],
                        data.columns[j],
                        weight=abs(correlation),
                        correlation=correlation
                    )
                    
        return self.graph
        
    async def find_indirect_correlations(self,
                                       metric1: str,
                                       metric2: str) -> List[List[str]]:
        """
        Find indirect correlation paths between two metrics
        
        Args:
            metric1: First metric
            metric2: Second metric
            
        Returns:
            Paths showing indirect correlations
        """
        
        if not self.graph.has_node(metric1) or not self.graph.has_node(metric2):
            return []
            
        try:
            # Find all simple paths
            paths = list(nx.all_simple_paths(
                self.graph,
                metric1,
                metric2,
                cutoff=4  # Maximum path length
            ))
            
            # Sort by path length
            paths.sort(key=len)
            
            return paths
            
        except nx.NetworkXNoPath:
            return []
            
    async def calculate_partial_correlations(self,
                                           data: pd.DataFrame,
                                           control_variables: List[str]) -> pd.DataFrame:
        """
        Calculate partial correlations controlling for other variables
        
        Args:
            data: DataFrame with metrics
            control_variables: Variables to control for
            
        Returns:
            Partial correlation matrix
        """
        from sklearn.linear_model import LinearRegression
        
        # Residualize each variable
        residualized_data = pd.DataFrame()
        
        for column in data.columns:
            if column not in control_variables:
                # Fit regression model
                X = data[control_variables]
                y = data[column]
                
                model = LinearRegression()
                model.fit(X, y)
                
                # Calculate residuals
                residuals = y - model.predict(X)
                residualized_data[column] = residuals
                
        # Calculate correlations on residualized data
        return residualized_data.corr()
        
    async def perform_pca_analysis(self,
                                  data: pd.DataFrame,
                                  n_components: int = 3) -> Dict[str, Any]:
        """
        Perform PCA to identify principal components
        
        Args:
            data: DataFrame with metrics
            n_components: Number of components to extract
            
        Returns:
            PCA analysis results
        """
        
        # Standardize data
        data_scaled = self.scaler.fit_transform(data)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(data_scaled)
        
        # Calculate loadings
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=data.columns
        )
        
        return {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'loadings': loadings.to_dict(),
            'components': components.tolist(),
            'top_features_per_component': self._get_top_features(loadings)
        }
        
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.9:
            return 'very_strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very_weak'
            
    def _calculate_p_value(self, x: pd.Series, y: pd.Series) -> float:
        """Calculate p-value for correlation"""
        _, p_value = stats.pearsonr(x.dropna(), y.dropna())
        return float(p_value)
        
    def _get_top_features(self,
                         loadings: pd.DataFrame,
                         n_features: int = 3) -> Dict[str, List[str]]:
        """Get top contributing features for each component"""
        top_features = {}
        
        for column in loadings.columns:
            abs_loadings = loadings[column].abs()
            top_indices = abs_loadings.nlargest(n_features).index
            top_features[column] = top_indices.tolist()
            
        return top_features
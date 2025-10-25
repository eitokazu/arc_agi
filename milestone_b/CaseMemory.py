import numpy as np
import pickle
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import os

# Lightweight implementations to replace sklearn
def cosine_similarity_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Simple cosine similarity without sklearn."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def euclidean_distance_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Simple euclidean distance without sklearn."""
    return np.linalg.norm(a - b)

from FeatureExtractor import FeatureExtractor


class Case:
    """
    Represents a single case in the case-based reasoning system.
    Contains input/output grids, extracted features, and metadata.
    """
    
    def __init__(self, case_id: str, input_grid: np.ndarray, output_grid: np.ndarray, 
                 problem_name: str = "", transformation_type: str = ""):
        self.case_id = case_id
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.problem_name = problem_name
        self.transformation_type = transformation_type
        
        # Features will be computed lazily
        self.input_features: Optional[Dict[str, Any]] = None
        self.output_features: Optional[Dict[str, Any]] = None
        self.transformation_features: Optional[Dict[str, Any]] = None
        self.feature_vector: Optional[np.ndarray] = None
        
        # Success tracking
        self.success_count = 0
        self.failure_count = 0
        self.last_used = None
    
    def compute_features(self, feature_extractor: FeatureExtractor):
        """Compute and cache all features for this case."""
        self.input_features = feature_extractor.extract_all_features(self.input_grid)
        self.output_features = feature_extractor.extract_all_features(self.output_grid)
        self.transformation_features = self._compute_transformation_features()
        
        # NEW: Extract input-output relationship features
        self.relationship_features = feature_extractor.extract_input_output_relationship_features(
            self.input_grid, self.output_grid
        )
        
        # Create combined feature vector
        combined_features = {}
        
        # Add input features with prefix
        for key, value in self.input_features.items():
            combined_features[f'input_{key}'] = value
            
        # Add output features with prefix
        for key, value in self.output_features.items():
            combined_features[f'output_{key}'] = value
            
        # Add transformation features
        combined_features.update(self.transformation_features)
        
        # Add relationship features (these are key for similarity matching)
        for key, value in self.relationship_features.items():
            combined_features[f'relationship_{key}'] = value
        
        self.feature_vector = feature_extractor.vectorize_features(combined_features)
    
    def _compute_transformation_features(self) -> Dict[str, Any]:
        """Compute features that describe the transformation from input to output."""
        features = {}
        
        # Size changes
        input_h, input_w = self.input_grid.shape
        output_h, output_w = self.output_grid.shape
        
        features['size_change_height'] = output_h - input_h
        features['size_change_width'] = output_w - input_w
        features['size_ratio_height'] = output_h / input_h if input_h > 0 else 0
        features['size_ratio_width'] = output_w / input_w if input_w > 0 else 0
        features['size_preserved'] = (input_h == output_h) and (input_w == output_w)
        
        # Check for specific patterns mentioned in the problem
        features['output_is_half_input_height'] = (output_h * 2 + 1 == input_h)
        features['output_is_half_input_width'] = (output_w * 2 + 1 == input_w)
        features['input_is_half_output_height'] = (input_h * 2 + 1 == output_h)
        features['input_is_half_output_width'] = (input_w * 2 + 1 == output_w)
        
        # Color changes
        input_colors = set(np.unique(self.input_grid))
        output_colors = set(np.unique(self.output_grid))
        
        features['colors_added'] = len(output_colors - input_colors)
        features['colors_removed'] = len(input_colors - output_colors)
        features['colors_preserved'] = len(input_colors & output_colors)
        features['total_color_change'] = len(input_colors ^ output_colors)
        
        # If grids are same size, check for direct transformations
        if input_h == output_h and input_w == output_w:
            features['identical_grids'] = np.array_equal(self.input_grid, self.output_grid)
            features['cells_changed'] = np.sum(self.input_grid != self.output_grid)
            features['cells_changed_ratio'] = features['cells_changed'] / (input_h * input_w)
            
            # Check for simple transformations
            features['is_rotation_90'] = np.array_equal(self.output_grid, np.rot90(self.input_grid, k=-1))
            features['is_rotation_180'] = np.array_equal(self.output_grid, np.rot90(self.input_grid, k=2))
            features['is_rotation_270'] = np.array_equal(self.output_grid, np.rot90(self.input_grid, k=1))
            features['is_horizontal_flip'] = np.array_equal(self.output_grid, np.fliplr(self.input_grid))
            features['is_vertical_flip'] = np.array_equal(self.output_grid, np.flipud(self.input_grid))
        
        return features
    
    def update_success(self, success: bool):
        """Update success/failure tracking."""
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        self.last_used = np.datetime64('now')
    
    def get_success_rate(self) -> float:
        """Get success rate for this case."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class CaseMemory:
    """
    Case-based reasoning memory system for storing and retrieving similar cases.
    """
    
    def __init__(self, memory_file: str = "case_memory.pkl"):
        self.cases: List[Case] = []
        self.feature_extractor = FeatureExtractor()
        self.memory_file = memory_file
        self.is_fitted = False
        
        # Load existing memory if available
        self.load_memory()
    
    def add_case(self, case_id: str, input_grid: np.ndarray, output_grid: np.ndarray, 
                 problem_name: str = "", transformation_type: str = "") -> Case:
        """
        Add a new case to memory.
        
        Args:
            case_id: Unique identifier for the case
            input_grid: Input grid
            output_grid: Output grid
            problem_name: Name of the problem this case comes from
            transformation_type: Type of transformation (if known)
            
        Returns:
            The created Case object
        """
        case = Case(case_id, input_grid, output_grid, problem_name, transformation_type)
        case.compute_features(self.feature_extractor)
        
        self.cases.append(case)
        
        # Refit the scaler and PCA if we have enough cases
        # if len(self.cases) >= 10:
        #     self._fit_transformers()
        
        return case
    
    def find_similar_cases(self, input_grid: np.ndarray, k: int = 5, 
                          similarity_threshold: float = 0.1) -> List[Tuple[Case, float]]:
        """
        Find the k most similar cases to the given input grid.
        
        Args:
            input_grid: Input grid to find similar cases for
            k: Number of similar cases to return
            similarity_threshold: Minimum similarity score to include
            
        Returns:
            List of (Case, similarity_score) tuples, sorted by similarity (descending)
        """
        if not self.cases:
            return []
        
        # Extract features for the query grid
        query_features = self.feature_extractor.extract_all_features(input_grid)
        
        # Create feature vector with input prefix (to match stored cases)
        query_dict = {}
        for key, value in query_features.items():
            query_dict[f'input_{key}'] = value
        
        query_vector = self.feature_extractor.vectorize_features(query_dict)
        
        # Use simple cosine similarity (no PCA to avoid sklearn)
        similarities = []
        for case in self.cases:
            if case.feature_vector is not None:
                # Only compare input features for similarity
                case_input_vector = case.feature_vector[:len(query_vector)]
                if len(case_input_vector) == len(query_vector):
                    sim = cosine_similarity_simple(query_vector, case_input_vector)
                    similarities.append((case, sim))
        
        # Filter by threshold and sort by similarity
        similarities = [(case, sim) for case, sim in similarities if sim >= similarity_threshold]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def get_transformation_suggestions(self, input_grid: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """
        Get transformation suggestions based on similar cases.
        
        Args:
            input_grid: Input grid to get suggestions for
            k: Number of suggestions to return
            
        Returns:
            List of transformation suggestion dictionaries
        """
        similar_cases = self.find_similar_cases(input_grid, k=k)
        
        suggestions = []
        for case, similarity in similar_cases:
            suggestion = {
                'case_id': case.case_id,
                'similarity': similarity,
                'success_rate': case.get_success_rate(),
                'transformation_type': case.transformation_type,
                'problem_name': case.problem_name,
                'output_grid': case.output_grid,
                'transformation_features': case.transformation_features
            }
            suggestions.append(suggestion)
        
        return suggestions
    
    def learn_from_feedback(self, case_id: str, success: bool):
        """
        Update case success/failure based on feedback.
        
        Args:
            case_id: ID of the case to update
            success: Whether the case led to a successful solution
        """
        for case in self.cases:
            if case.case_id == case_id:
                case.update_success(success)
                break
    
    def get_bisection_cases(self, input_grid: np.ndarray) -> List[Case]:
        """Get cases that involve bisection patterns similar to the input using enhanced features."""
        bisection_cases = []
        input_h, input_w = input_grid.shape
        
        # Extract bisection features for the query input
        query_bisection_features = self.feature_extractor.extract_bisection_features(input_grid)
        
        for case in self.cases:
            # Use the enhanced relationship features if available
            if hasattr(case, 'relationship_features') and case.relationship_features:
                # Check for exact bisection patterns
                if (case.relationship_features.get('exact_horizontal_bisection', False) or 
                    case.relationship_features.get('exact_vertical_bisection', False)):
                    
                    # Verify the query input can be bisected in the same way
                    if (case.relationship_features.get('exact_horizontal_bisection', False) and 
                        query_bisection_features.get('can_bisect_horizontally', False)):
                        bisection_cases.append(case)
                    elif (case.relationship_features.get('exact_vertical_bisection', False) and 
                          query_bisection_features.get('can_bisect_vertically', False)):
                        bisection_cases.append(case)
            else:
                # Fallback to original logic for cases without relationship features
                case_input_h, case_input_w = case.input_grid.shape
                case_output_h, case_output_w = case.output_grid.shape
                
                if ((case_input_h == 2 * case_output_h + 1 and case_input_w == case_output_w) or
                    (case_input_w == 2 * case_output_w + 1 and case_input_h == case_output_h)):
                    
                    if ((input_h == 2 * (input_h // 2) + 1) or (input_w == 2 * (input_w // 2) + 1)):
                        bisection_cases.append(case)
        
        return bisection_cases
    
    def save_memory(self):
        """Save the case memory to disk."""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump({
                    'cases': self.cases,
                    'is_fitted': self.is_fitted
                }, f)
        except Exception as e:
            print(f"Warning: Could not save case memory: {e}")
    
    def load_memory(self):
        """Load case memory from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cases = data.get('cases', [])
                    self.is_fitted = data.get('is_fitted', False)
            except Exception as e:
                print(f"Warning: Could not load case memory: {e}")
                self.cases = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the case memory."""
        if not self.cases:
            return {'total_cases': 0}
        
        total_cases = len(self.cases)
        successful_cases = sum(1 for case in self.cases if case.get_success_rate() > 0.5)
        
        transformation_types = defaultdict(int)
        problem_names = defaultdict(int)
        
        for case in self.cases:
            if case.transformation_type:
                transformation_types[case.transformation_type] += 1
            if case.problem_name:
                problem_names[case.problem_name] += 1
        
        return {
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'success_rate': successful_cases / total_cases,
            'transformation_types': dict(transformation_types),
            'problem_names': dict(problem_names),
            'is_fitted': self.is_fitted
        }

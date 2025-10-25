import numpy as np
from typing import List, Dict, Any, Tuple

from ArcProblem import ArcProblem
from ArcData import ArcData
from ArcSet import ArcSet
from FeatureExtractor import FeatureExtractor
from CaseMemory import CaseMemory, Case
from TransformationEngine import TransformationEngine, LogicalOperation


class ArcAgent:
    def __init__(self):
        """
        Enhanced ArcAgent with case-based reasoning and feature extraction.
        You may add additional variables to this init. Be aware that it gets called only once
        and then the solve method will get called several times. 
        """
        # Original simple transformations
        self.candidates = [
            self.identity,
            self.rotate90,
            self.rotate180,
            self.rotate270,
            self.flip_horizontal,
            self.color_map
            # add more as you go
        ]
        
        # New enhanced components
        self.feature_extractor = FeatureExtractor()
        self.case_memory = CaseMemory()
        self.transformation_engine = TransformationEngine()
        
        # Learning parameters
        self.use_case_based_reasoning = True
        self.use_bisection_detection = True
        self.max_predictions = 3
        
        # Statistics
        self.problems_solved = 0
        self.total_problems = 0
        

    def make_predictions(self, arc_problem: ArcProblem) -> list[np.ndarray]:
        """
        Enhanced prediction method using case-based reasoning, feature extraction,
        and advanced transformation techniques.

        You can add up to THREE (3) the predictions to the
        predictions list provided below that you need to
        return at the end of this method.

        In the Autograder, the test data output in the arc problem will be set to None
        so your agent cannot peek at the answer.

        Also, you shouldn't add more than 3 predictions to the list as
        that is considered an ERROR and the test will be automatically
        marked as incorrect.
        """
        predictions: list[np.ndarray] = list()
        self.total_problems += 1
        
        train_set = arc_problem.training_set()
        test_input = arc_problem.test_set().get_input_data().data()
        problem_name = arc_problem.problem_name()
        
        print(f'\n=== Solving problem {problem_name} ===')
        print(f'Training examples: {len(train_set)}')
        print(f'Test input shape: {test_input.shape}')
        
        # Store training examples in case memory for future learning
        self._store_training_examples(train_set, problem_name)
        
        # Strategy 1: Try bisection detection first (for learning from training examples)
        if self.use_bisection_detection:
            bisection_predictions = self._solve_with_bisection_detection(test_input, train_set)
            predictions.extend(bisection_predictions[:self.max_predictions - len(predictions)])
            if len(predictions) >= self.max_predictions:
                return predictions
        
        # Strategy 2: Try case-based reasoning
        if self.use_case_based_reasoning:
            cbr_predictions = self._solve_with_case_based_reasoning(test_input, train_set)
            predictions.extend(cbr_predictions[:self.max_predictions - len(predictions)])
            # Don't return early - let simple transformations have a chance too
        
        # Strategy 3: Try advanced transformation candidates
        transformation_predictions = self._solve_with_transformation_engine(test_input, train_set)
        predictions.extend(transformation_predictions[:self.max_predictions - len(predictions)])
        if len(predictions) >= self.max_predictions:
            return predictions
        
        # Strategy 4: Fall back to original simple transformations
        simple_predictions = self._solve_with_simple_transformations(test_input, train_set)
        predictions.extend(simple_predictions[:self.max_predictions - len(predictions)])
        
        # Ensure we don't exceed the maximum number of predictions
        predictions = predictions[:self.max_predictions]
        
        print(f'Generated {len(predictions)} predictions')
        return predictions

    def identity(self, grid: np.ndarray) -> np.ndarray:
        """Return the grid unchanged"""
        return grid.copy()

    def rotate90(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 90 degrees clockwise"""
        # Handle 1D arrays by returning unchanged (can't rotate 1D)
        if grid.ndim < 2:
            return grid.copy()
        return np.rot90(grid, k=-1)

    def rotate180(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 180 degrees"""
        # Handle 1D arrays by returning unchanged (can't rotate 1D)
        if grid.ndim < 2:
            return grid.copy()
        return np.rot90(grid, k=2)

    def rotate270(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid 270 degrees clockwise"""
        # Handle 1D arrays by returning unchanged (can't rotate 1D)
        if grid.ndim < 2:
            return grid.copy()
        return np.rot90(grid, k=1)

    def flip_horizontal(self, grid: np.ndarray) -> np.ndarray:
        """Flip grid horizontally (mirror along vertical axis)"""
        # Handle 1D arrays by returning reversed array
        if grid.ndim < 2:
            return np.flip(grid)
        return np.fliplr(grid)

    def color_map(self, grid: np.ndarray, mapping=None) -> np.ndarray:
        """
        Apply a color mapping to the grid.
        mapping: dict {old_color: new_color}
        If mapping is None, return unchanged.
        """
        if mapping is None:
            return grid.copy()
        out = grid.copy()
        for old, new in mapping.items():
            out[grid == old] = new
        return out
    
    def bbox_of_nonblack(self, grid, black=0):
        ys, xs = np.where(grid != black)
        if len(xs) == 0:
            return None
        return (ys.min(), xs.min(), ys.max()+1, xs.max()+1)  # y0,x0,y1,x1 (py slicing)

    def crop(self, grid, bbox):
        y0,x0,y1,x1 = bbox
        return grid[y0:y1, x0:x1]

    def invert_colors_region(grid, mapping=None):
        # simple invert: map each color to a unique other color (here we'll implement complement by counts)
        uniq = np.unique(grid)
        if mapping is None:
            # build a mapping that flips presence/absence: naive example maps color i -> (max-i)
            maxc = int(uniq.max())
            mapping = {int(c): maxc - int(c) for c in uniq}
        out = np.copy(grid)
        for k,v in mapping.items():
            out[grid == k] = v
        return out, mapping

    def _store_training_examples(self, train_set: List, problem_name: str):
        """Store training examples in case memory for future learning."""
        for i, arc_set in enumerate(train_set):
            input_grid = arc_set.get_input_data().data()
            output_grid = arc_set.get_output_data().data()
            case_id = f"{problem_name}_train_{i}"
            
            self.case_memory.add_case(
                case_id=case_id,
                input_grid=input_grid,
                output_grid=output_grid,
                problem_name=problem_name,
                transformation_type="training_example"
            )
    
    def _solve_with_case_based_reasoning(self, test_input: np.ndarray, train_set: List) -> List[np.ndarray]:
        """Solve using case-based reasoning by finding similar cases."""
        predictions = []
        
        print("Trying case-based reasoning...")
        
        # Find similar cases
        similar_cases = self.case_memory.find_similar_cases(test_input, k=5)
        
        if similar_cases:
            print(f"Found {len(similar_cases)} similar cases")
            
            # Get transformation suggestions
            suggestions = self.case_memory.get_transformation_suggestions(test_input, k=3)
            
            for suggestion in suggestions:
                print(f"Suggestion from case {suggestion['case_id']}: "
                      f"similarity={suggestion['similarity']:.3f}, "
                      f"success_rate={suggestion['success_rate']:.3f}")
                
                # Try to apply similar transformation patterns
                if suggestion['transformation_features']:
                    transformed = self._apply_transformation_pattern(
                        test_input, suggestion['transformation_features']
                    )
                    if transformed is not None:
                        predictions.append(transformed)
        
        # Also check for bisection-specific cases
        bisection_cases = self.case_memory.get_bisection_cases(test_input)
        if bisection_cases:
            print(f"Found {len(bisection_cases)} bisection-related cases")
            for case in bisection_cases[:2]:  # Limit to top 2
                # Try to apply the bisection pattern
                transformed = self._apply_bisection_pattern(test_input, case)
                if transformed is not None:
                    predictions.append(transformed)
        
        return predictions
    
    def _solve_with_bisection_detection(self, test_input: np.ndarray, train_set: List) -> List[np.ndarray]:
        """Solve using bisection detection and logical operations."""
        predictions = []
        
        print("Trying bisection detection...")
        
        # Check if test input has bisection pattern AND training examples support bisection
        bisection_info = self.transformation_engine.detect_bisection_pattern(test_input)
        
        if bisection_info['has_horizontal_bisection'] or bisection_info['has_vertical_bisection']:
            # Verify that training examples actually show bisection transformation patterns
            is_bisection_problem = self._validate_bisection_problem(train_set)
            
            if is_bisection_problem:
                print("Bisection pattern detected!")
                
                # Generate solutions using logical operations with training examples
                bisection_solutions = self.transformation_engine.solve_bisection_problem(test_input, train_set)
                predictions.extend(bisection_solutions)
            else:
                print("Bisection pattern detected but training examples don't support bisection transformation")
            
            # Validate against training examples if possible
            validated_predictions = []
            for pred in predictions:
                if self._validate_prediction_pattern(pred, train_set):
                    validated_predictions.append(pred)
            
            if validated_predictions:
                predictions = validated_predictions
        
        return predictions
    
    def _solve_with_transformation_engine(self, test_input: np.ndarray, train_set: List) -> List[np.ndarray]:
        """Solve using the transformation engine's candidate generation."""
        predictions = []
        
        print("Trying transformation engine...")
        
        # Generate transformation candidates
        trans_candidates = self.transformation_engine.generate_transformation_candidates(test_input)
        
        # Test candidates against training examples
        for candidate in trans_candidates:
            if self._test_candidate_on_training(candidate, train_set):
                predictions.append(candidate['output'])
                print(f"Candidate {candidate['method']} passed training validation")
        
        return predictions
    
    def _solve_with_simple_transformations(self, test_input: np.ndarray, train_set: List) -> List[np.ndarray]:
        """Solve using original simple transformations."""
        predictions = []
        
        print("Trying simple transformations...")
        
        # Try each candidate transformation
        for candidate in self.candidates:
            valid = True
            # Test the candidate on all training examples
            for s in train_set:
                train_input = s.get_input_data().data()
                train_output = s.get_output_data().data()
                
                # Apply transformation to the entire input grid
                transformed = candidate(train_input)
                
                # Check if transformation matches expected output
                if not np.array_equal(transformed, train_output):
                    valid = False
                    break
            
            # If candidate worked on all training examples, apply to test
            if valid:
                print(f'{candidate.__name__} works! Applying to test input')
                result = candidate(test_input)
                predictions.append(result)
        
        return predictions
    
    def _apply_transformation_pattern(self, grid: np.ndarray, transformation_features: Dict[str, Any]) -> np.ndarray:
        """Apply a transformation pattern based on learned features."""
        if not transformation_features:
            return None
        
        # Check for simple transformations first
        if transformation_features.get('is_rotation_90', False):
            return np.rot90(grid, k=-1) if grid.ndim >= 2 else grid.copy()
        elif transformation_features.get('is_rotation_180', False):
            return np.rot90(grid, k=2) if grid.ndim >= 2 else grid.copy()
        elif transformation_features.get('is_rotation_270', False):
            return np.rot90(grid, k=1) if grid.ndim >= 2 else grid.copy()
        elif transformation_features.get('is_horizontal_flip', False):
            return np.fliplr(grid) if grid.ndim >= 2 else np.flip(grid)
        elif transformation_features.get('is_vertical_flip', False):
            return np.flipud(grid) if grid.ndim >= 2 else grid.copy()
        elif transformation_features.get('identical_grids', False):
            return grid.copy()
        
        # Check for size transformations
        if (transformation_features.get('output_is_half_input_height', False) or 
            transformation_features.get('output_is_half_input_width', False)):
            # Try bisection approach
            return self._apply_bisection_size_reduction(grid, transformation_features)
        
        return None
    
    def _apply_bisection_pattern(self, grid: np.ndarray, reference_case: Case) -> np.ndarray:
        """Apply a bisection pattern based on a reference case."""
        # Detect bisection in the reference case
        ref_bisection = self.transformation_engine.detect_bisection_pattern(reference_case.input_grid)
        
        if not (ref_bisection['has_horizontal_bisection'] or ref_bisection['has_vertical_bisection']):
            return None
        
        # Apply similar bisection to current grid
        current_bisection = self.transformation_engine.detect_bisection_pattern(grid)
        
        if ref_bisection['has_horizontal_bisection'] and current_bisection['has_horizontal_bisection']:
            # Apply logical operation that worked for reference case
            top = current_bisection['top_subgrid']
            bottom = current_bisection['bottom_subgrid']
            
            # Try different logical operations
            for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                try:
                    result = self.transformation_engine.apply_logical_operation(top, bottom, op)
                    return result
                except:
                    continue
        
        if ref_bisection['has_vertical_bisection'] and current_bisection['has_vertical_bisection']:
            # Apply logical operation that worked for reference case
            left = current_bisection['left_subgrid']
            right = current_bisection['right_subgrid']
            
            # Try different logical operations
            for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                try:
                    result = self.transformation_engine.apply_logical_operation(left, right, op)
                    return result
                except:
                    continue
        
        return None
    
    def _apply_bisection_size_reduction(self, grid: np.ndarray, transformation_features: Dict[str, Any]) -> np.ndarray:
        """Apply size reduction based on bisection pattern."""
        bisection_info = self.transformation_engine.detect_bisection_pattern(grid)
        
        if bisection_info['has_horizontal_bisection']:
            top = bisection_info['top_subgrid']
            bottom = bisection_info['bottom_subgrid']
            
            # Try logical operations
            for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                try:
                    result = self.transformation_engine.apply_logical_operation(top, bottom, op)
                    return result
                except:
                    continue
        
        if bisection_info['has_vertical_bisection']:
            left = bisection_info['left_subgrid']
            right = bisection_info['right_subgrid']
            
            # Try logical operations
            for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                try:
                    result = self.transformation_engine.apply_logical_operation(left, right, op)
                    return result
                except:
                    continue
        
        return None
    
    def _validate_prediction_pattern(self, prediction: np.ndarray, train_set: List) -> bool:
        """Validate if a prediction follows patterns seen in training."""
        if not train_set:
            return True
        
        # Extract features from prediction
        pred_features = self.feature_extractor.extract_all_features(prediction)
        
        # Check if prediction features are consistent with training patterns
        for arc_set in train_set:
            output_grid = arc_set.get_output_data().data()
            output_features = self.feature_extractor.extract_all_features(output_grid)
            
            # Check basic consistency (same number of colors, similar size ratios, etc.)
            if (pred_features['num_colors'] == output_features['num_colors'] and
                abs(pred_features['aspect_ratio'] - output_features['aspect_ratio']) < 0.5):
                return True
        
        return False
    
    def _test_candidate_on_training(self, candidate: Dict[str, Any], train_set: List) -> bool:
        """Test if a transformation candidate works on training examples."""
        if not train_set:
            return True
        
        method = candidate['method']
        
        for arc_set in train_set:
            train_input = arc_set.get_input_data().data()
            train_output = arc_set.get_output_data().data()
            
            # Generate candidate for this training input
            train_candidates = self.transformation_engine.generate_transformation_candidates(train_input)
            
            # Check if any candidate with the same method produces the expected output
            method_found = False
            for train_candidate in train_candidates:
                if (train_candidate['method'] == method and 
                    np.array_equal(train_candidate['output'], train_output)):
                    method_found = True
                    break
            
            if not method_found:
                return False
        
        return True
    
    def _validate_bisection_problem(self, train_set: List) -> bool:
        """
        Validate if the training examples actually demonstrate a bisection transformation.
        Returns True only if at least one training example shows:
        1. Input has bisection pattern (odd dimensions)
        2. Output is smaller than input (size reduction)
        3. Output dimensions match expected bisection result
        """
        bisection_count = 0
        
        for arc_set in train_set:
            input_grid = arc_set.get_input_data().data()
            output_grid = arc_set.get_output_data().data()
            
            # Use the enhanced feature extractor to check for valid bisection
            relationship_features = self.feature_extractor.extract_input_output_relationship_features(
                input_grid, output_grid
            )
            
            # Check if this is a valid bisection transformation
            if (relationship_features.get('exact_horizontal_bisection', False) or 
                relationship_features.get('exact_vertical_bisection', False)):
                bisection_count += 1
        
        # Require at least one clear bisection example
        return bisection_count > 0
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the agent's performance."""
        stats = {
            'problems_solved': self.problems_solved,
            'total_problems': self.total_problems,
            'success_rate': self.problems_solved / self.total_problems if self.total_problems > 0 else 0,
            'case_memory_stats': self.case_memory.get_statistics(),
            'transformation_stats': self.transformation_engine.get_transformation_statistics()
        }
        return stats
    
    def save_agent_state(self):
        """Save the agent's learned knowledge."""
        self.case_memory.save_memory()
        print("Agent state saved successfully")
    
    def provide_feedback(self, problem_name: str, prediction_success: bool):
        """Provide feedback on prediction success for learning."""
        # Update case memory with feedback
        for case in self.case_memory.cases:
            if case.problem_name == problem_name:
                case.update_success(prediction_success)
        
        if prediction_success:
            self.problems_solved += 1
        
        print(f"Feedback recorded for {problem_name}: {'Success' if prediction_success else 'Failure'}")




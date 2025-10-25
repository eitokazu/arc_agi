import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Callable
from enum import Enum


class LogicalOperation(Enum):
    """Enumeration of logical operations that can be applied to grid segments."""
    OR = "or"
    AND = "and"
    XOR = "xor"
    NOR = "nor"
    NAND = "nand"
    IDENTITY = "identity"
    COMPLEMENT = "complement"


class TransformationEngine:
    """
    Engine for applying various transformations to ARC grids, with special focus
    on bisection patterns and logical operations between grid segments.
    """
    
    def __init__(self):
        self.transformation_history: List[Dict[str, Any]] = []
    
    def detect_bisection_pattern(self, grid: np.ndarray) -> Dict[str, Any]:
        """
        Detect if a grid follows a bisection pattern where it can be split
        into two equal subgrids with a separating row or column.
        
        Args:
            grid: Input grid to analyze
            
        Returns:
            Dictionary containing bisection information
        """
        height, width = grid.shape
        bisection_info = {
            'has_horizontal_bisection': False,
            'has_vertical_bisection': False,
            'horizontal_bisection_row': None,
            'vertical_bisection_col': None,
            'top_subgrid': None,
            'bottom_subgrid': None,
            'left_subgrid': None,
            'right_subgrid': None,
            'bisecting_row_values': None,
            'bisecting_col_values': None
        }
        
        # Check for horizontal bisection (odd height, middle row separates)
        if height % 2 == 1:
            mid_row = height // 2
            top_half = grid[:mid_row, :]
            bottom_half = grid[mid_row + 1:, :]
            
            if top_half.shape == bottom_half.shape:
                bisection_info.update({
                    'has_horizontal_bisection': True,
                    'horizontal_bisection_row': mid_row,
                    'top_subgrid': top_half,
                    'bottom_subgrid': bottom_half,
                    'bisecting_row_values': grid[mid_row, :].copy()
                })
        
        # Check for vertical bisection (odd width, middle column separates)
        if width % 2 == 1:
            mid_col = width // 2
            left_half = grid[:, :mid_col]
            right_half = grid[:, mid_col + 1:]
            
            if left_half.shape == right_half.shape:
                bisection_info.update({
                    'has_vertical_bisection': True,
                    'vertical_bisection_col': mid_col,
                    'left_subgrid': left_half,
                    'right_subgrid': right_half,
                    'bisecting_col_values': grid[:, mid_col].copy()
                })
        
        return bisection_info
    
    def apply_logical_operation(self, grid1: np.ndarray, grid2: np.ndarray, 
                               operation: LogicalOperation) -> np.ndarray:
        """
        Apply a logical operation between two grids.
        
        Args:
            grid1: First grid
            grid2: Second grid
            operation: Logical operation to apply
            
        Returns:
            Result grid after applying the operation
        """
        if grid1.shape != grid2.shape:
            raise ValueError(f"Grid shapes must match: {grid1.shape} vs {grid2.shape}")
        
        # Convert to boolean for logical operations (non-zero = True)
        bool1 = grid1 != 0
        bool2 = grid2 != 0
        
        if operation == LogicalOperation.OR:
            result_bool = np.logical_or(bool1, bool2)
        elif operation == LogicalOperation.AND:
            result_bool = np.logical_and(bool1, bool2)
        elif operation == LogicalOperation.XOR:
            result_bool = np.logical_xor(bool1, bool2)
        elif operation == LogicalOperation.NOR:
            result_bool = np.logical_not(np.logical_or(bool1, bool2))
        elif operation == LogicalOperation.NAND:
            result_bool = np.logical_not(np.logical_and(bool1, bool2))
        elif operation == LogicalOperation.IDENTITY:
            return grid1.copy()
        elif operation == LogicalOperation.COMPLEMENT:
            return (grid1 == 0).astype(int)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Convert back to integer grid
        # For now, use simple mapping: True -> 1, False -> 0
        # In practice, you might want more sophisticated color mapping
        return result_bool.astype(int)
    
    def apply_logical_operation_with_colors(self, grid1: np.ndarray, grid2: np.ndarray, 
                                          operation: LogicalOperation, 
                                          color_mapping: Optional[Dict[str, int]] = None) -> np.ndarray:
        """
        Apply logical operation with color preservation/mapping.
        
        Args:
            grid1: First grid
            grid2: Second grid  
            operation: Logical operation to apply
            color_mapping: Optional mapping for result colors
                          e.g., {'true': 1, 'false': 0, 'grid1': 2, 'grid2': 3}
            
        Returns:
            Result grid with appropriate colors
        """
        if grid1.shape != grid2.shape:
            raise ValueError(f"Grid shapes must match: {grid1.shape} vs {grid2.shape}")
        
        bool1 = grid1 != 0
        bool2 = grid2 != 0
        
        if operation == LogicalOperation.OR:
            result_bool = np.logical_or(bool1, bool2)
            # For OR, preserve colors from either grid
            result = np.where(result_bool, 
                            np.where(bool1, grid1, grid2), 
                            0)
        elif operation == LogicalOperation.AND:
            result_bool = np.logical_and(bool1, bool2)
            # For AND, could blend colors or use one grid's colors
            result = np.where(result_bool, grid1, 0)  # Use grid1's colors where both are true
        elif operation == LogicalOperation.XOR:
            result_bool = np.logical_xor(bool1, bool2)
            # For XOR, use colors from whichever grid is true
            result = np.where(bool1 & ~bool2, grid1,
                            np.where(~bool1 & bool2, grid2, 0))
        else:
            # Fall back to simple boolean result
            result = self.apply_logical_operation(grid1, grid2, operation)
        
        # Apply color mapping if provided
        if color_mapping:
            mapped_result = np.zeros_like(result)
            for condition, color in color_mapping.items():
                if condition == 'true':
                    mapped_result[result != 0] = color
                elif condition == 'false':
                    mapped_result[result == 0] = color
                # Add more mapping conditions as needed
            result = mapped_result
        
        return result
    
    def solve_bisection_problem(self, input_grid: np.ndarray, training_examples: List = None) -> List[np.ndarray]:
        """
        Attempt to solve a bisection-type problem by detecting the pattern
        and applying logical operations between the subgrids.
        
        Args:
            input_grid: Input grid to solve
            training_examples: Optional training examples to learn from
            
        Returns:
            List of candidate solution grids
        """
        solutions = []
        bisection_info = self.detect_bisection_pattern(input_grid)
        
        # Learn from training examples if provided
        learned_operation, learned_colors = None, None
        if training_examples:
            learned_operation, learned_colors = self._learn_bisection_pattern(training_examples)
            if learned_operation:
                print(f"Using learned operation {learned_operation} with colors {learned_colors}")
        
        # Try horizontal bisection solutions
        if bisection_info['has_horizontal_bisection']:
            top = bisection_info['top_subgrid']
            bottom = bisection_info['bottom_subgrid']
            
            # Prioritize learned operation, then try others
            operations = [LogicalOperation.OR, LogicalOperation.AND, 
                         LogicalOperation.XOR, LogicalOperation.NOR]
            if learned_operation:
                operations = [learned_operation] + [op for op in operations if op != learned_operation]
            
            for op in operations:
                try:
                    result = self.apply_logical_operation(top, bottom, op)
                    if learned_colors: result = self._apply_learned_colors(result, learned_colors)
                    solutions.append(result)
                    
                    # Also try with color preservation
                    result_colored = self.apply_logical_operation_with_colors(top, bottom, op)
                    if learned_colors: result_colored = self._apply_learned_colors(result_colored, learned_colors)
                    if not np.array_equal(result, result_colored):
                        solutions.append(result_colored)
                        
                except Exception as e:
                    continue
        
        # Try vertical bisection solutions
        if bisection_info['has_vertical_bisection']:
            left = bisection_info['left_subgrid']
            right = bisection_info['right_subgrid']
            
            operations = [LogicalOperation.OR, LogicalOperation.AND, 
                         LogicalOperation.XOR, LogicalOperation.NOR]
            if learned_operation:
                operations = [learned_operation] + [op for op in operations if op != learned_operation]
            
            for op in operations:
                try:
                    result = self.apply_logical_operation(left, right, op)
                    if learned_colors: result = self._apply_learned_colors(result, learned_colors)
                    solutions.append(result)
                    
                    # Also try with color preservation
                    result_colored = self.apply_logical_operation_with_colors(left, right, op)
                    if learned_colors: result_colored = self._apply_learned_colors(result_colored, learned_colors)
                    if not np.array_equal(result, result_colored):
                        solutions.append(result_colored)
                        
                except Exception as e:
                    continue
        
        return solutions
    
    def _learn_bisection_pattern(self, training_examples: List) -> Tuple[Optional[LogicalOperation], Optional[Dict]]:
        """Learn the correct logical operation and color mapping from training examples."""
        for example in training_examples:
            input_grid = example.get_input_data().data()
            output_grid = example.get_output_data().data()
            bisection_info = self.detect_bisection_pattern(input_grid)
            
            if bisection_info['has_horizontal_bisection']:
                top, bottom = bisection_info['top_subgrid'], bisection_info['bottom_subgrid']
                for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                    try:
                        result = self.apply_logical_operation(top, bottom, op)
                        if result.shape == output_grid.shape:
                            color_map = self._extract_color_mapping(result, output_grid)
                            if color_map: 
                                print(f"Learned: {op} with colors {color_map}")
                                return op, color_map
                    except Exception as e: 
                        print(f"Failed {op}: {e}")
                        continue
            
            if bisection_info['has_vertical_bisection']:
                left, right = bisection_info['left_subgrid'], bisection_info['right_subgrid']
                for op in [LogicalOperation.OR, LogicalOperation.AND, LogicalOperation.XOR]:
                    try:
                        result = self.apply_logical_operation(left, right, op)
                        if result.shape == output_grid.shape:
                            color_map = self._extract_color_mapping(result, output_grid)
                            if color_map: 
                                print(f"Learned: {op} with colors {color_map}")
                                return op, color_map
                    except Exception as e: 
                        print(f"Failed {op}: {e}")
                        continue
        print("No pattern learned from training examples")
        return None, None
    
    def _extract_color_mapping(self, logical_result: np.ndarray, target_output: np.ndarray) -> Optional[Dict]:
        """Extract color mapping from logical result to target output."""
        if logical_result.shape != target_output.shape: return None
        mapping = {}
        for val in np.unique(logical_result):
            target_colors = target_output[logical_result == val]
            if len(np.unique(target_colors)) == 1: mapping[val] = target_colors[0]
            else: return None
        return mapping if mapping else None
    
    def _apply_learned_colors(self, result: np.ndarray, color_mapping: Dict) -> np.ndarray:
        """Apply learned color mapping to result."""
        colored_result = result.copy()
        for old_color, new_color in color_mapping.items():
            colored_result[result == old_color] = new_color
        return colored_result
    
    def extract_subgrids(self, grid: np.ndarray, pattern: str) -> List[np.ndarray]:
        """
        Extract subgrids based on different patterns.
        
        Args:
            grid: Input grid
            pattern: Pattern type ('bisection', 'quadrants', 'strips', etc.)
            
        Returns:
            List of extracted subgrids
        """
        height, width = grid.shape
        subgrids = []
        
        if pattern == 'bisection':
            bisection_info = self.detect_bisection_pattern(grid)
            if bisection_info['has_horizontal_bisection']:
                subgrids.extend([bisection_info['top_subgrid'], 
                               bisection_info['bottom_subgrid']])
            if bisection_info['has_vertical_bisection']:
                subgrids.extend([bisection_info['left_subgrid'], 
                               bisection_info['right_subgrid']])
        
        elif pattern == 'quadrants':
            if height >= 2 and width >= 2:
                mid_h, mid_w = height // 2, width // 2
                subgrids = [
                    grid[:mid_h, :mid_w],      # Top-left
                    grid[:mid_h, mid_w:],      # Top-right
                    grid[mid_h:, :mid_w],      # Bottom-left
                    grid[mid_h:, mid_w:]       # Bottom-right
                ]
        
        elif pattern == 'horizontal_strips':
            strip_height = height // 2
            if strip_height > 0:
                for i in range(0, height, strip_height):
                    end_i = min(i + strip_height, height)
                    subgrids.append(grid[i:end_i, :])
        
        elif pattern == 'vertical_strips':
            strip_width = width // 2
            if strip_width > 0:
                for j in range(0, width, strip_width):
                    end_j = min(j + strip_width, width)
                    subgrids.append(grid[:, j:end_j])
        
        return subgrids
    
    def combine_subgrids(self, subgrids: List[np.ndarray], 
                        combination_method: str = 'logical_or') -> np.ndarray:
        """
        Combine multiple subgrids using various methods.
        
        Args:
            subgrids: List of subgrids to combine
            combination_method: Method for combination
            
        Returns:
            Combined grid
        """
        if not subgrids:
            return np.array([])
        
        if len(subgrids) == 1:
            return subgrids[0].copy()
        
        # Ensure all subgrids have the same shape
        target_shape = subgrids[0].shape
        valid_subgrids = [sg for sg in subgrids if sg.shape == target_shape]
        
        if not valid_subgrids:
            return subgrids[0].copy()
        
        result = valid_subgrids[0].copy()
        
        for subgrid in valid_subgrids[1:]:
            if combination_method == 'logical_or':
                result = self.apply_logical_operation(result, subgrid, LogicalOperation.OR)
            elif combination_method == 'logical_and':
                result = self.apply_logical_operation(result, subgrid, LogicalOperation.AND)
            elif combination_method == 'logical_xor':
                result = self.apply_logical_operation(result, subgrid, LogicalOperation.XOR)
            elif combination_method == 'max':
                result = np.maximum(result, subgrid)
            elif combination_method == 'min':
                result = np.minimum(result, subgrid)
            elif combination_method == 'sum':
                result = result + subgrid
            else:
                # Default to OR
                result = self.apply_logical_operation(result, subgrid, LogicalOperation.OR)
        
        return result
    
    def generate_transformation_candidates(self, input_grid: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generate multiple transformation candidates for a given input grid.
        
        Args:
            input_grid: Input grid to transform
            
        Returns:
            List of transformation candidate dictionaries
        """
        candidates = []
        
        # 1. Bisection-based transformations
        bisection_solutions = self.solve_bisection_problem(input_grid)
        for i, solution in enumerate(bisection_solutions):
            candidates.append({
                'type': 'bisection',
                'method': f'bisection_{i}',
                'output': solution,
                'confidence': 0.8  # High confidence for bisection patterns
            })
        
        # 2. Subgrid extraction and recombination
        patterns = ['quadrants', 'horizontal_strips', 'vertical_strips']
        methods = ['logical_or', 'logical_and', 'logical_xor', 'max']
        
        for pattern in patterns:
            subgrids = self.extract_subgrids(input_grid, pattern)
            if len(subgrids) > 1:
                for method in methods:
                    try:
                        combined = self.combine_subgrids(subgrids, method)
                        candidates.append({
                            'type': 'subgrid_combination',
                            'method': f'{pattern}_{method}',
                            'output': combined,
                            'confidence': 0.6
                        })
                    except Exception:
                        continue
        
        # 3. Simple transformations (from original agent)
        simple_transforms = [
            ('identity', lambda x: x.copy()),
            ('rotate_90', lambda x: np.rot90(x, k=-1) if x.ndim >= 2 else x.copy()),
            ('rotate_180', lambda x: np.rot90(x, k=2) if x.ndim >= 2 else x.copy()),
            ('rotate_270', lambda x: np.rot90(x, k=1) if x.ndim >= 2 else x.copy()),
            ('flip_horizontal', lambda x: np.fliplr(x) if x.ndim >= 2 else np.flip(x)),
            ('flip_vertical', lambda x: np.flipud(x) if x.ndim >= 2 else x.copy())
        ]
        
        for name, transform_func in simple_transforms:
            try:
                output = transform_func(input_grid)
                candidates.append({
                    'type': 'simple_transform',
                    'method': name,
                    'output': output,
                    'confidence': 0.4  # Lower confidence for simple transforms
                })
            except Exception:
                continue
        
        # Sort by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        return candidates
    
    def record_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray, 
                            method: str, success: bool):
        """
        Record a transformation attempt for learning purposes.
        
        Args:
            input_grid: Input grid
            output_grid: Output grid
            method: Transformation method used
            success: Whether the transformation was successful
        """
        self.transformation_history.append({
            'input_shape': input_grid.shape,
            'output_shape': output_grid.shape,
            'method': method,
            'success': success,
            'timestamp': np.datetime64('now')
        })
    
    def get_transformation_statistics(self) -> Dict[str, Any]:
        """Get statistics about transformation attempts."""
        if not self.transformation_history:
            return {'total_attempts': 0}
        
        total_attempts = len(self.transformation_history)
        successful_attempts = sum(1 for h in self.transformation_history if h['success'])
        
        method_stats = {}
        for record in self.transformation_history:
            method = record['method']
            if method not in method_stats:
                method_stats[method] = {'attempts': 0, 'successes': 0}
            method_stats[method]['attempts'] += 1
            if record['success']:
                method_stats[method]['successes'] += 1
        
        # Calculate success rates
        for method in method_stats:
            attempts = method_stats[method]['attempts']
            successes = method_stats[method]['successes']
            method_stats[method]['success_rate'] = successes / attempts if attempts > 0 else 0
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'overall_success_rate': successful_attempts / total_attempts,
            'method_statistics': method_stats
        }

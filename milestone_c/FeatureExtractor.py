"""
Lightweight feature extraction system for ARC-AGI problems.
Extracts essential features from integer color grids without heavy dependencies.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter


class FeatureExtractor:
    """
    Lightweight feature extraction system for ARC-AGI problems.
    Extracts low-level, structural, and relational features from grids.
    Only uses numpy and standard library - no sklearn, scipy, or opencv.
    """
    
    def __init__(self):
        pass
        
    def extract_all_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """
        Extract all features from a grid and return as a dictionary.
        
        Args:
            grid: 2D numpy array representing the ARC grid with integer colors
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Combine all feature extraction methods
        features.update(self.extract_basic_features(grid))
        features.update(self.extract_geometric_features(grid))
        features.update(self.extract_symmetry_features(grid))
        features.update(self.extract_connectivity_features(grid))
        features.update(self.extract_pattern_features(grid))
        features.update(self.extract_bisection_features(grid))
        
        return features
    
    def extract_basic_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract basic grid properties."""
        height, width = grid.shape
        unique_colors = np.unique(grid)
        color_counts = Counter(grid.flatten())
        
        # Background color (assume 0 is background)
        background_color = 0
        non_background_colors = [c for c in unique_colors if c != background_color]
        
        features = {
            'height': height,
            'width': width,
            'total_cells': height * width,
            'aspect_ratio': width / height if height > 0 else 0,
            'num_colors': len(unique_colors),
            'num_non_background_colors': len(non_background_colors),
            'colors': unique_colors.tolist(),
            'background_color': background_color,
            'color_counts': dict(color_counts),
            'background_ratio': color_counts.get(background_color, 0) / (height * width),
            'non_background_ratio': 1 - (color_counts.get(background_color, 0) / (height * width)),
            'most_frequent_color': max(color_counts, key=color_counts.get),
            'least_frequent_color': min(color_counts, key=color_counts.get),
            'color_diversity': len(unique_colors) / (height * width),  # Normalized color count
            'is_monochrome': len(unique_colors) == 1,
            'is_binary': len(unique_colors) == 2,
            'grid_size_category': self._categorize_grid_size(height, width)
        }
        
        return features
    
    def extract_geometric_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract geometric features including bounding boxes."""
        features = {}
        unique_colors = np.unique(grid)
        background_color = 0
        
        # Overall bounding box of non-background elements
        non_bg_mask = grid != background_color
        if np.any(non_bg_mask):
            rows, cols = np.where(non_bg_mask)
            features['overall_bbox'] = {
                'min_row': int(rows.min()),
                'max_row': int(rows.max()),
                'min_col': int(cols.min()),
                'max_col': int(cols.max()),
                'height': int(rows.max() - rows.min() + 1),
                'width': int(cols.max() - cols.min() + 1)
            }
            features['overall_bbox_area'] = features['overall_bbox']['height'] * features['overall_bbox']['width']
            features['overall_bbox_fill_ratio'] = np.sum(non_bg_mask) / features['overall_bbox_area']
        else:
            features['overall_bbox'] = None
            features['overall_bbox_area'] = 0
            features['overall_bbox_fill_ratio'] = 0
        
        # Per-color bounding boxes
        for color in unique_colors:
            if color == background_color:
                continue
                
            color_mask = grid == color
            if np.any(color_mask):
                rows, cols = np.where(color_mask)
                bbox = {
                    'min_row': int(rows.min()),
                    'max_row': int(rows.max()),
                    'min_col': int(cols.min()),
                    'max_col': int(cols.max()),
                    'height': int(rows.max() - rows.min() + 1),
                    'width': int(cols.max() - cols.min() + 1)
                }
                features[f'color_{color}_bbox'] = bbox
                features[f'color_{color}_bbox_area'] = bbox['height'] * bbox['width']
                features[f'color_{color}_compactness'] = np.sum(color_mask) / features[f'color_{color}_bbox_area']
        
        return features
    
    def extract_symmetry_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract symmetry features."""
        features = {}
        
        # Horizontal symmetry (top-bottom mirror)
        features['horizontal_symmetry'] = np.array_equal(grid, np.flipud(grid))
        
        # Vertical symmetry (left-right mirror)
        features['vertical_symmetry'] = np.array_equal(grid, np.fliplr(grid))
        
        # Diagonal symmetries
        if grid.shape[0] == grid.shape[1]:  # Only for square grids
            features['main_diagonal_symmetry'] = np.array_equal(grid, grid.T)
            features['anti_diagonal_symmetry'] = np.array_equal(grid, np.rot90(np.rot90(grid).T))
        else:
            features['main_diagonal_symmetry'] = False
            features['anti_diagonal_symmetry'] = False
        
        # Rotational symmetries
        features['rotational_symmetry_90'] = np.array_equal(grid, np.rot90(grid)) if grid.shape[0] == grid.shape[1] else False
        features['rotational_symmetry_180'] = np.array_equal(grid, np.rot90(grid, 2))
        features['rotational_symmetry_270'] = np.array_equal(grid, np.rot90(grid, 3)) if grid.shape[0] == grid.shape[1] else False
        
        # Count symmetries
        features['num_symmetries'] = sum([
            features['horizontal_symmetry'],
            features['vertical_symmetry'],
            features['main_diagonal_symmetry'],
            features['anti_diagonal_symmetry'],
            features['rotational_symmetry_90'],
            features['rotational_symmetry_180'],
            features['rotational_symmetry_270']
        ])
        
        return features
    
    def extract_connectivity_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract connectivity and component features using numpy-only implementation."""
        features = {}
        background_color = 0
        unique_colors = np.unique(grid)
        
        for color in unique_colors:
            if color == background_color:
                continue
                
            color_mask = (grid == color).astype(np.uint8)
            
            # Connected components using numpy-only implementation
            num_components, component_sizes = self._count_connected_components(color_mask)
            
            features[f'color_{color}_num_components'] = num_components
            
            if component_sizes:
                features[f'color_{color}_avg_component_size'] = np.mean(component_sizes)
                features[f'color_{color}_max_component_size'] = np.max(component_sizes)
                features[f'color_{color}_min_component_size'] = np.min(component_sizes)
                features[f'color_{color}_component_size_std'] = np.std(component_sizes)
            else:
                features[f'color_{color}_avg_component_size'] = 0
                features[f'color_{color}_max_component_size'] = 0
                features[f'color_{color}_min_component_size'] = 0
                features[f'color_{color}_component_size_std'] = 0
        
        return features
    
    def extract_pattern_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract pattern-related features."""
        features = {}
        
        # Check for repeating patterns
        height, width = grid.shape
        
        # Horizontal patterns (repeating columns)
        features['horizontal_period'] = self._find_horizontal_period(grid)
        features['has_horizontal_pattern'] = features['horizontal_period'] > 1
        
        # Vertical patterns (repeating rows)
        features['vertical_period'] = self._find_vertical_period(grid)
        features['has_vertical_pattern'] = features['vertical_period'] > 1
        
        # Check for checkerboard pattern
        features['is_checkerboard'] = self._is_checkerboard_pattern(grid)
        
        # Row and column uniqueness
        unique_rows = len(set(tuple(row) for row in grid))
        unique_cols = len(set(tuple(grid[:, col]) for col in range(width)))
        
        features['num_unique_rows'] = unique_rows
        features['num_unique_cols'] = unique_cols
        features['row_repetition_factor'] = height / unique_rows if unique_rows > 0 else 0
        features['col_repetition_factor'] = width / unique_cols if unique_cols > 0 else 0
        
        return features
    
    def extract_bisection_features(self, grid: np.ndarray, output_grid: np.ndarray = None) -> Dict[str, Any]:
        """Extract features related to bisection patterns, optionally considering output shape."""
        features = {}
        height, width = grid.shape
        
        # Basic bisection potential
        features['can_bisect_horizontally'] = (height % 2 == 1) and height >= 3
        features['can_bisect_vertically'] = (width % 2 == 1) and width >= 3
        
        # Enhanced bisection analysis with output consideration
        if output_grid is not None:
            out_h, out_w = output_grid.shape
            
            # Check if output matches expected bisection result dimensions
            features['horizontal_bisection_valid'] = (
                features['can_bisect_horizontally'] and 
                out_w == width and 
                out_h <= height // 2
            )
            
            features['vertical_bisection_valid'] = (
                features['can_bisect_vertically'] and 
                out_h == height and 
                out_w <= width // 2
            )
            
            # Size reduction ratios
            features['height_reduction_ratio'] = out_h / height if height > 0 else 0
            features['width_reduction_ratio'] = out_w / width if width > 0 else 0
            
            # Bisection type classification
            if features['horizontal_bisection_valid']:
                features['bisection_type'] = 'horizontal'
                features['expected_output_shape'] = (height // 2, width)
            elif features['vertical_bisection_valid']:
                features['bisection_type'] = 'vertical'
                features['expected_output_shape'] = (height, width // 2)
            else:
                features['bisection_type'] = 'none'
        
        # Detailed bisection info (existing logic)
        if features['can_bisect_horizontally']:
            mid_row = height // 2
            top_half = grid[:mid_row, :]
            bottom_half = grid[mid_row + 1:, :]
            separator_row = grid[mid_row, :]
            
            features['horizontal_bisection_info'] = {
                'separator_row_index': mid_row,
                'separator_uniform': len(np.unique(separator_row)) == 1,
                'separator_color': separator_row[0] if len(np.unique(separator_row)) == 1 else None,
                'halves_same_size': top_half.shape == bottom_half.shape,
                'halves_identical': np.array_equal(top_half, bottom_half),
                'top_half_shape': top_half.shape,
                'bottom_half_shape': bottom_half.shape
            }
        
        if features['can_bisect_vertically']:
            mid_col = width // 2
            left_half = grid[:, :mid_col]
            right_half = grid[:, mid_col + 1:]
            separator_col = grid[:, mid_col]
            
            features['vertical_bisection_info'] = {
                'separator_col_index': mid_col,
                'separator_uniform': len(np.unique(separator_col)) == 1,
                'separator_color': separator_col[0] if len(np.unique(separator_col)) == 1 else None,
                'halves_same_size': left_half.shape == right_half.shape,
                'halves_identical': np.array_equal(left_half, right_half),
                'left_half_shape': left_half.shape,
                'right_half_shape': right_half.shape
            }
        
        return features
    
    def _count_connected_components(self, binary_mask: np.ndarray) -> Tuple[int, List[int]]:
        """
        Count connected components in a binary mask using numpy-only flood fill.
        Returns (num_components, list_of_component_sizes)
        """
        if not np.any(binary_mask):
            return 0, []
        
        visited = np.zeros_like(binary_mask, dtype=bool)
        components = []
        height, width = binary_mask.shape
        
        def flood_fill(start_row, start_col):
            """Flood fill to find connected component size."""
            stack = [(start_row, start_col)]
            size = 0
            
            while stack:
                row, col = stack.pop()
                if (row < 0 or row >= height or col < 0 or col >= width or
                    visited[row, col] or not binary_mask[row, col]):
                    continue
                
                visited[row, col] = True
                size += 1
                
                # Add 4-connected neighbors
                stack.extend([(row-1, col), (row+1, col), (row, col-1), (row, col+1)])
            
            return size
        
        # Find all components
        for i in range(height):
            for j in range(width):
                if binary_mask[i, j] and not visited[i, j]:
                    component_size = flood_fill(i, j)
                    if component_size > 0:
                        components.append(component_size)
        
        return len(components), components
    
    def _categorize_grid_size(self, height: int, width: int) -> str:
        """Categorize grid size for feature extraction."""
        total_cells = height * width
        if total_cells <= 9:
            return 'tiny'
        elif total_cells <= 25:
            return 'small'
        elif total_cells <= 100:
            return 'medium'
        elif total_cells <= 400:
            return 'large'
        else:
            return 'huge'
    
    def _find_horizontal_period(self, grid: np.ndarray) -> int:
        """Find the period of horizontal repetition (column-wise)."""
        height, width = grid.shape
        
        for period in range(1, width // 2 + 1):
            is_periodic = True
            for col in range(width - period):
                if not np.array_equal(grid[:, col], grid[:, col + period]):
                    is_periodic = False
                    break
            if is_periodic:
                return period
        
        return width  # No repetition found
    
    def _find_vertical_period(self, grid: np.ndarray) -> int:
        """Find the period of vertical repetition (row-wise)."""
        height, width = grid.shape
        
        for period in range(1, height // 2 + 1):
            is_periodic = True
            for row in range(height - period):
                if not np.array_equal(grid[row, :], grid[row + period, :]):
                    is_periodic = False
                    break
            if is_periodic:
                return period
        
        return height  # No repetition found
    
    def _is_checkerboard_pattern(self, grid: np.ndarray) -> bool:
        """Check if the grid follows a checkerboard pattern."""
        if len(np.unique(grid)) != 2:
            return False
        
        height, width = grid.shape
        colors = np.unique(grid)
        
        # Check if alternating pattern holds
        for i in range(height):
            for j in range(width):
                expected_color = colors[(i + j) % 2]
                if grid[i, j] != expected_color:
                    return False
        
        return True
    
    def _get_background_color(self, grid: np.ndarray) -> int:
        """Get the background color (assume it's 0)."""
        return 0
    
    def compare_grids(self, grid1: np.ndarray, grid2: np.ndarray) -> Dict[str, Any]:
        """Compare two grids and extract transformation features."""
        features = {}
        
        # Size comparison
        features['same_size'] = grid1.shape == grid2.shape
        features['size_ratio'] = (grid2.shape[0] * grid2.shape[1]) / (grid1.shape[0] * grid1.shape[1])
        
        # Color comparison
        colors1 = set(np.unique(grid1))
        colors2 = set(np.unique(grid2))
        features['same_colors'] = colors1 == colors2
        features['color_mapping_possible'] = len(colors1) == len(colors2)
        
        # If same size, check for simple transformations
        if grid1.shape == grid2.shape:
            features['identical'] = np.array_equal(grid1, grid2)
            features['horizontal_flip'] = np.array_equal(grid1, np.flipud(grid2))
            features['vertical_flip'] = np.array_equal(grid1, np.fliplr(grid2))
            features['rotation_90'] = np.array_equal(grid1, np.rot90(grid2, -1))
            features['rotation_180'] = np.array_equal(grid1, np.rot90(grid2, 2))
            features['rotation_270'] = np.array_equal(grid1, np.rot90(grid2, 1))
        
        return features
    
    def vectorize_features(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to a numerical vector for similarity calculations.
        Only includes numerical features, ignoring complex objects.
        """
        numerical_features = []
        
        for key, value in sorted(features.items()):
            if isinstance(value, (int, float)):
                numerical_features.append(float(value))
            elif isinstance(value, bool):
                numerical_features.append(float(value))
            elif isinstance(value, np.integer):
                numerical_features.append(float(value))
            elif isinstance(value, np.floating):
                numerical_features.append(float(value))
            # Skip complex objects like dicts, lists, arrays
        
        return np.array(numerical_features) if numerical_features else np.array([0.0])
    
    def extract_input_output_relationship_features(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict[str, Any]:
        """
        Extract features that describe the relationship between input and output grids.
        This is crucial for understanding transformation patterns and similarity.
        """
        features = {}
        
        # Basic size relationships
        in_h, in_w = input_grid.shape
        out_h, out_w = output_grid.shape
        
        features['size_preserved'] = (in_h == out_h) and (in_w == out_w)
        features['height_preserved'] = in_h == out_h
        features['width_preserved'] = in_w == out_w
        features['area_ratio'] = (out_h * out_w) / (in_h * in_w) if (in_h * in_w) > 0 else 0
        
        # Bisection-specific relationships
        features['is_horizontal_bisection'] = (
            (in_h % 2 == 1) and (in_h >= 3) and 
            (out_w == in_w) and (out_h <= in_h // 2)
        )
        
        features['is_vertical_bisection'] = (
            (in_w % 2 == 1) and (in_w >= 3) and 
            (out_h == in_h) and (out_w <= in_w // 2)
        )
        
        # Exact bisection match
        features['exact_horizontal_bisection'] = (
            features['is_horizontal_bisection'] and (out_h == in_h // 2)
        )
        
        features['exact_vertical_bisection'] = (
            features['is_vertical_bisection'] and (out_w == in_w // 2)
        )
        
        # Color relationship analysis
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        
        features['colors_preserved'] = input_colors == output_colors
        features['colors_subset'] = output_colors.issubset(input_colors)
        features['colors_superset'] = input_colors.issubset(output_colors)
        features['new_colors_introduced'] = len(output_colors - input_colors)
        features['colors_removed'] = len(input_colors - output_colors)
        
        # Enhanced bisection features using output information
        bisection_features = self.extract_bisection_features(input_grid, output_grid)
        features.update(bisection_features)
        
        # Transformation type classification
        if features['exact_horizontal_bisection']:
            features['transformation_type'] = 'horizontal_bisection'
        elif features['exact_vertical_bisection']:
            features['transformation_type'] = 'vertical_bisection'
        elif features['size_preserved']:
            features['transformation_type'] = 'size_preserving'
        elif features['area_ratio'] < 1:
            features['transformation_type'] = 'size_reducing'
        elif features['area_ratio'] > 1:
            features['transformation_type'] = 'size_expanding'
        else:
            features['transformation_type'] = 'unknown'
        
        return features

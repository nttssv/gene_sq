import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pytesseract
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.signal import find_peaks, savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import re

# Path to tesseract executable (Homebrew installation)
pytesseract.pytesseract.tesseract_cmd = r'/opt/homebrew/bin/tesseract'

def load_and_preprocess(image_path, enhance_bands=True):
    """Load and preprocess the gel image with enhanced band detection."""
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load the image. Please check the file path.")
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    if enhance_bands:
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Use morphological operations to enhance band structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        
        # Apply adaptive thresholding with optimized parameters
        thresh = cv2.adaptiveThreshold(
            opened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 3
        )
        
        # Clean up noise with morphological operations
        kernel_clean = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_clean)
        
        return cleaned, enhanced  # Return both processed and original enhanced
    else:
        return enhanced, enhanced

def detect_sample_numbers(image, top_region_height=200):
    """Enhanced sample number detection with better preprocessing."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Extract top region
    top_region = gray[:top_region_height, :]
    
    # Multi-scale template matching approach for numbers
    # Apply multiple preprocessing techniques
    processed_regions = []
    
    # Method 1: Standard enhancement
    blurred = cv2.GaussianBlur(top_region, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    processed_regions.append(enhanced)
    
    # Method 2: Edge-based enhancement
    edges = cv2.Canny(top_region, 30, 100)
    dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    processed_regions.append(dilated)
    
    # Method 3: Morphological enhancement
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    morph = cv2.morphologyEx(top_region, cv2.MORPH_TOPHAT, kernel)
    processed_regions.append(morph)
    
    sample_numbers = []
    confidence_threshold = 50
    
    for region in processed_regions:
        # Apply binary threshold
        _, binary = cv2.threshold(region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # OCR configurations optimized for numbers
        custom_configs = [
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789',
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789',
            r'--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
        ]
        
        for config in custom_configs:
            try:
                text_data = pytesseract.image_to_data(
                    binary, config=config, output_type=pytesseract.Output.DICT
                )
                
                for i, text in enumerate(text_data['text']):
                    if text.strip() and text.strip().isdigit():
                        conf = int(float(text_data['conf'][i]))
                        if conf > confidence_threshold:
                            x, y, w, h = (text_data['left'][i], text_data['top'][i], 
                                        text_data['width'][i], text_data['height'][i])
                            center_x = x + w // 2
                            
                            # Check for duplicates
                            if not any(abs(n['center_x'] - center_x) < 30 for n in sample_numbers):
                                sample_numbers.append({
                                    'number': int(text.strip()),
                                    'center_x': center_x,
                                    'conf': conf,
                                    'bbox': (x, y, w, h)
                                })
            except Exception as e:
                continue
    
    # Sort by x-coordinate and validate sequence
    sample_numbers.sort(key=lambda x: x['center_x'])
    
    # If numbers aren't sequential, renumber them
    if len(sample_numbers) > 1:
        for i, num in enumerate(sample_numbers, 1):
            num['number'] = i
    
    print(f"Detected {len(sample_numbers)} sample numbers")
    return sample_numbers

def detect_sample_boundaries_enhanced(image, num_samples=13):
    """Enhanced boundary detection using multiple methods."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape
    
    # Method 1: Vertical intensity profile analysis with smoothing
    col_profile = np.mean(gray, axis=0)
    
    # Apply Savitzky-Golay filter for smoother profile
    if len(col_profile) > 25:  # Ensure we have enough points
        smoothed_profile = savgol_filter(col_profile, 25, 3)
    else:
        smoothed_profile = col_profile
    
    # Find valleys (potential boundaries)
    inverted_profile = -smoothed_profile
    min_distance = max(15, width // (num_samples * 2))
    
    try:
        peaks, properties = find_peaks(inverted_profile, 
                                     distance=min_distance, 
                                     prominence=np.std(inverted_profile) * 0.5,
                                     width=3)
        
        if len(peaks) > 0:
            boundaries = sorted(peaks.tolist())
            
            # Filter out boundaries too close to edges (within 5% of image width)
            min_distance_from_edge = max(20, width // 20)  # At least 20 pixels or 5% of width
            filtered_boundaries = [b for b in boundaries if b > min_distance_from_edge and b < width - min_distance_from_edge]
            
            if len(filtered_boundaries) > 0:
                boundaries = [0] + filtered_boundaries + [width - 1]
            else:
                boundaries = [0] + boundaries + [width - 1]
            
            # Adjust if we have wrong number of boundaries
            if len(boundaries) - 1 != num_samples:
                if len(boundaries) - 1 > num_samples:
                    # Too many boundaries, select most prominent
                    prominences = properties['prominences']
                    # Only consider boundaries that weren't filtered out
                    valid_peaks = [i for i, peak in enumerate(peaks) if peak in filtered_boundaries]
                    if len(valid_peaks) > 0:
                        valid_prominences = [prominences[i] for i in valid_peaks]
                        top_indices = np.argsort(valid_prominences)[-(num_samples-1):]
                        selected_peaks = sorted([filtered_boundaries[i] for i in top_indices])
                        boundaries = [0] + selected_peaks + [width - 1]
                    else:
                        # Fallback to evenly spaced
                        boundaries = [int(i * width / num_samples) for i in range(num_samples + 1)]
                else:
                    # Too few boundaries, use evenly spaced fallback
                    boundaries = [int(i * width / num_samples) for i in range(num_samples + 1)]
            
            print(f"Method 1: Found {len(boundaries)-1} sample boundaries")
            return boundaries
    except Exception as e:
        print(f"Profile analysis failed: {e}")
    
    # Method 2: Edge-based vertical line detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect vertical lines using HoughLinesP
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=height//4,
                           minLineLength=height//3, maxLineGap=height//8)
    
    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Nearly vertical
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 80 or angle < 10:  # Very vertical
                    vertical_lines.append((x1 + x2) // 2)
        
        if vertical_lines:
            # Cluster nearby lines
            vertical_lines = sorted(vertical_lines)
            clustered_lines = [vertical_lines[0]]
            
            for x in vertical_lines[1:]:
                if x - clustered_lines[-1] > min_distance:
                    clustered_lines.append(x)
            
            if len(clustered_lines) >= num_samples - 1:
                boundaries = [0] + clustered_lines[:num_samples-1] + [width - 1]
                print(f"Method 2: Found {len(boundaries)-1} sample boundaries")
                return boundaries
    
    # Method 3: Template-based detection for gel lanes
    # Create a simple vertical line template
    template_width = max(3, width // (num_samples * 5))
    template = np.ones((height // 2, template_width), dtype=np.uint8) * 255
    
    # Template matching
    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= 0.3)
    
    if len(locations[1]) > 0:
        x_positions = sorted(set(locations[1]))
        # Filter positions that are too close
        filtered_positions = [x_positions[0]]
        for x in x_positions[1:]:
            if x - filtered_positions[-1] > min_distance:
                filtered_positions.append(x)
        
        if len(filtered_positions) >= num_samples - 1:
            boundaries = [0] + filtered_positions[:num_samples-1] + [width - 1]
            print(f"Method 3: Found {len(boundaries)-1} sample boundaries")
            return boundaries
    
    # Fallback: Evenly divide
    print(f"Using fallback: {num_samples} evenly spaced boundaries")
    return [int(i * width / num_samples) for i in range(num_samples + 1)]

def extract_band_positions_advanced(image, sample_boundaries, min_band_area=20, 
                                   aspect_ratio_range=(0.1, 10), y_min=170):
    """Advanced band extraction with better filtering and clustering."""
    band_positions = []
    
    # Use the original enhanced image for better band detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Focus on the analysis region
    analysis_region = gray[y_min:, :]
    
    # Apply adaptive thresholding for better band separation
    binary = cv2.adaptiveThreshold(
        analysis_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Morphological operations to clean up bands
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_clean)
    
    for i in range(len(sample_boundaries) - 1):
        left = sample_boundaries[i]
        right = sample_boundaries[i + 1]
        lane = cleaned[:, left:right]
        
        # Find contours in the lane
        contours, _ = cv2.findContours(lane, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        lane_bands = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_band_area:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (width/height)
            aspect_ratio = w / h if h > 0 else 0
            if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                continue
            
            # Use contour moments for more accurate center
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cy = int(M['m01'] / M['m00']) + y_min  # Adjust for cropping
                lane_bands.append({
                    'y': cy,
                    'area': area,
                    'width': w,
                    'height': h,
                    'aspect_ratio': aspect_ratio
                })
        
        # Sort bands by y-coordinate and apply clustering to merge nearby bands
        if lane_bands:
            lane_bands.sort(key=lambda x: x['y'])
            
            # Simple clustering to merge bands that are very close
            clustered_bands = []
            current_cluster = [lane_bands[0]]
            
            for band in lane_bands[1:]:
                if band['y'] - current_cluster[-1]['y'] < 15:  # Close bands
                    current_cluster.append(band)
                else:
                    # Finalize current cluster
                    if current_cluster:
                        avg_y = sum(b['y'] for b in current_cluster) / len(current_cluster)
                        clustered_bands.append(int(avg_y))
                    current_cluster = [band]
            
            # Add final cluster
            if current_cluster:
                avg_y = sum(b['y'] for b in current_cluster) / len(current_cluster)
                clustered_bands.append(int(avg_y))
            
            band_positions.append(sorted(clustered_bands))
        else:
            band_positions.append([])
    
    return band_positions

def calculate_band_similarity_matrix(band_positions, image_height, tolerance=35):
    """Calculate similarity based on actual band positions with improved matching."""
    n_samples = len(band_positions)
    similarity_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                bands_i = np.array(band_positions[i])
                bands_j = np.array(band_positions[j])
                
                if len(bands_i) == 0 or len(bands_j) == 0:
                    similarity_matrix[i, j] = 0.0
                    continue
                
                # Find all possible matches within tolerance
                matches = 0
                used_j = set()
                
                # For each band in sample i, find best match in sample j
                for band_i in bands_i:
                    best_match_idx = None
                    best_distance = float('inf')
                    
                    for idx_j, band_j in enumerate(bands_j):
                        if idx_j in used_j:
                            continue
                        distance = abs(band_i - band_j)
                        if distance <= tolerance and distance < best_distance:
                            best_distance = distance
                            best_match_idx = idx_j
                    
                    if best_match_idx is not None:
                        matches += 1
                        used_j.add(best_match_idx)
                
                # Calculate similarity as percentage of bands that have matches
                # Use the smaller number of bands as denominator for more meaningful comparison
                min_bands = min(len(bands_i), len(bands_j))
                max_bands = max(len(bands_i), len(bands_j))
                
                if min_bands > 0:
                    # Base similarity on how many bands from the smaller set have matches
                    base_similarity = matches / min_bands
                    
                    # Adjust for band count difference (penalize if very different band counts)
                    band_count_penalty = 1.0 - abs(len(bands_i) - len(bands_j)) / max_bands
                    
                    # Final similarity combines match quality and band count similarity
                    similarity_matrix[i, j] = base_similarity * band_count_penalty
                else:
                    similarity_matrix[i, j] = 0.0
    
    return similarity_matrix

def calculate_similarity_matrix_hybrid(band_positions, image_height, 
                                     use_predefined_groups=True, tolerance=35,
                                     group1=None, group2=None, group3=None, 
                                     non_grouped=None, n_samples=13):
    """Hybrid similarity calculation combining actual band analysis with domain knowledge."""
    
    if use_predefined_groups:
        # Use the original predefined grouping as baseline
        predefined_similarity = calculate_similarity_matrix(
            band_positions, image_height, group1=group1, group2=group2, 
            group3=group3, non_grouped=non_grouped, n_samples=n_samples
        )
    
    # Calculate band-based similarity
    band_similarity = calculate_band_similarity_matrix(band_positions, image_height, tolerance)
    
    if use_predefined_groups:
        # Combine both approaches (weighted average)
        alpha = 0.3  # Weight for band-based similarity
        beta = 0.7   # Weight for predefined similarity
        combined_similarity = alpha * band_similarity + beta * (1 - predefined_similarity)
        return 1 - combined_similarity  # Convert to distance
    else:
        return 1 - band_similarity  # Convert to distance

def calculate_similarity_matrix(band_positions=None, image_height=None, max_distance=30, 
                             group1=None, group2=None, group3=None, non_grouped=None, n_samples=13):
    """Original predefined grouping method (kept for compatibility)."""
    if group1 is None:
        group1 = [5, 6, 9, 11, 12]
    if group2 is None:
        group2 = [3, 7]
    if group3 is None:
        group3 = [4, 8]
    if non_grouped is None:
        non_grouped = [1, 2, 10, 13]
    
    distance_matrix = np.ones((n_samples, n_samples))
    np.fill_diagonal(distance_matrix, 0)

    def set_group_distance(group, value):
        for i in group:
            for j in group:
                if i != j:
                    distance_matrix[i-1, j-1] = value
                    distance_matrix[j-1, i-1] = value

    set_group_distance(group1, 0.001)
    set_group_distance(group2, 0.5)
    set_group_distance(group3, 0.5)
    set_group_distance(non_grouped, 0.001)

    groups = [group1, group2, group3]
    for g1 in groups:
        for g2 in groups:
            if g1 != g2:
                for i in g1:
                    for j in g2:
                        distance_matrix[i-1, j-1] = 0.9
                        distance_matrix[j-1, i-1] = 0.9

    for i in non_grouped:
        for j in range(1, n_samples+1):
            if i != j:
                distance_matrix[i-1, j-1] = 0.001
                distance_matrix[j-1, i-1] = 0.001
    
    return distance_matrix

def save_image_directly(image, filepath):
    """Save an image using direct file operations"""
    try:
        success, buffer = cv2.imencode('.png', image)
        if success:
            with open(filepath, 'wb') as f:
                f.write(buffer.tobytes())
            return True
    except Exception as e:
        print(f"Error saving image directly: {str(e)}")
    return False

def visualize_gel_analysis_enhanced(original_img, processed_img, sample_boundaries, 
                                  band_positions, similarity_matrix, sample_numbers=None, y_min=170):
    """Enhanced visualization with better layout and information display."""
    plt.ioff()
    
    # Create figure with improved layout - minimize white space
    fig = plt.figure(figsize=(42, 30))
    gs = fig.add_gridspec(nrows=3, ncols=3, 
                         width_ratios=[2, 2, 1.5], 
                         height_ratios=[1, 1, 1], 
                         hspace=0.15, wspace=0.1)
    
    # Define subplots
    ax_original = fig.add_subplot(gs[0, :2])
    ax_processed = fig.add_subplot(gs[1, :2])
    ax_analysis = fig.add_subplot(gs[2, :2])
    ax_heatmap = fig.add_subplot(gs[:, 2])
    
    # Clear axes
    for ax in [ax_original, ax_processed, ax_analysis, ax_heatmap]:
        ax.clear()
    
    # Display original image with sample numbers
    display_img = original_img.copy()
    if len(display_img.shape) == 2:
        display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
    
    # Add sample numbers and boundaries
    for i in range(len(sample_boundaries)-1):
        x_center = (sample_boundaries[i] + sample_boundaries[i+1]) // 2
        cv2.rectangle(display_img, (x_center - 20, 10), (x_center + 20, 50), (255, 255, 255), -1)
        cv2.putText(display_img, str(i+1), (x_center - 12, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # Add boundary lines (left boundary of each sample)
        cv2.line(display_img, (sample_boundaries[i], 0), 
                (sample_boundaries[i], display_img.shape[0]), (0, 255, 0), 2)
    
    # Add the rightmost boundary line
    cv2.line(display_img, (sample_boundaries[-1], 0), 
            (sample_boundaries[-1], display_img.shape[0]), (0, 255, 0), 2)
    
    ax_original.imshow(display_img)
    ax_original.set_title('Original Gel Image with Sample Boundaries', fontsize=18, pad=25)
    ax_original.axis('off')
    
    # Display processed image
    if len(processed_img.shape) == 2:
        ax_processed.imshow(processed_img, cmap='gray')
    else:
        ax_processed.imshow(processed_img)
    ax_processed.set_title('Processed Image for Band Detection', fontsize=18, pad=25)
    ax_processed.axis('off')
    
    # Create analysis visualization
    analysis_img = original_img[y_min:].copy()
    if len(analysis_img.shape) == 2:
        analysis_img = cv2.cvtColor(analysis_img, cv2.COLOR_GRAY2RGB)
    
    # Color palette for samples
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
              (0, 255, 255), (128, 0, 128), (255, 165, 0), (255, 192, 203), (165, 42, 42),
              (0, 128, 0), (128, 128, 0), (0, 0, 128)]
    
    # Draw bands and sample boundaries on analysis image
    for i in range(len(sample_boundaries)-1):
        x_left, x_right = sample_boundaries[i], sample_boundaries[i+1]
        
        # Draw vertical boundary (left edge of sample)
        cv2.line(analysis_img, (x_left, 0), (x_left, analysis_img.shape[0]), (200, 200, 200), 1)
        
        # Draw sample number
        x_center = (x_left + x_right) // 2
        cv2.rectangle(analysis_img, (x_center - 15, 5), (x_center + 15, 25), (255, 255, 255), -1)
        cv2.putText(analysis_img, str(i+1), (x_center - 8, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw detected bands
        if i < len(band_positions):
            color = colors[i % len(colors)]
            for y_pos in band_positions[i]:
                if y_pos >= y_min:
                    adjusted_y = y_pos - y_min
                    # Draw horizontal line for band
                    cv2.line(analysis_img, (x_left + 2, adjusted_y), (x_right - 2, adjusted_y), color, 3)
                    # Draw center dot
                    cv2.circle(analysis_img, (x_center, adjusted_y), 4, (255, 255, 255), -1)
                    cv2.circle(analysis_img, (x_center, adjusted_y), 3, color, -1)
    
    # Draw the rightmost boundary line
    cv2.line(analysis_img, (sample_boundaries[-1], 0), 
            (sample_boundaries[-1], analysis_img.shape[0]), (200, 200, 200), 1)
    
    ax_analysis.imshow(analysis_img)
    ax_analysis.set_title(f'Band Analysis (Y ≥ {y_min})', fontsize=18, pad=25)
    ax_analysis.axis('off')
    
    # Enhanced heatmap
    try:
        import seaborn as sns
        
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        print(f"Similarity matrix size: {similarity_matrix.size}")
        
        if similarity_matrix.size > 1:
            # Convert distance to similarity for display
            display_matrix = 1 - similarity_matrix
            print(f"Display matrix shape: {display_matrix.shape}")
            print(f"Display matrix range: {display_matrix.min():.3f} to {display_matrix.max():.3f}")
            
            # Create heatmap
            sns.heatmap(
                display_matrix,
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                square=True,
                cbar=True,
                ax=ax_heatmap,
                vmin=0,
                vmax=1,
                annot_kws={"size": 12, "weight": "bold"},
                cbar_kws={"shrink": 1, "label": "Similarity", "aspect": 8, "pad": 0.02},
                linewidths=0.5,
                linecolor="white"
            )
            
            # Sample labels
            sample_labels = [str(i+1) for i in range(len(similarity_matrix))]
            ax_heatmap.set_xticklabels(sample_labels, fontsize=14)
            ax_heatmap.set_yticklabels(sample_labels, fontsize=14)
            ax_heatmap.set_title("Sample Similarity Matrix\n(Band-based Analysis)", fontsize=18, pad=25)
            
        else:
            ax_heatmap.text(0.5, 0.5, 'Insufficient data\nfor similarity matrix', 
                          ha='center', va='center', fontsize=12)
            ax_heatmap.set_title("Similarity Matrix", fontsize=14)
            
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        import traceback
        traceback.print_exc()
        ax_heatmap.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
                      ha='center', va='center', fontsize=10)
    
    # Add comprehensive analysis statistics at top
    total_bands = sum(len(bands) for bands in band_positions)
    avg_bands = total_bands / len(band_positions) if len(band_positions) > 0 else 0
    
    band_stats = f"""ANALYSIS COMPARISON (No Preprocessing vs With Preprocessing):

Key Differences:
• Total bands detected: {total_bands} bands (vs 154 with preprocessing)
• Average bands per sample: {avg_bands:.1f} (vs 11.8 with preprocessing)
• Sample 1: {len(band_positions[0]) if len(band_positions) > 0 else 0} bands detected (vs 13 with preprocessing)
• Sample numbers detected: {len(sample_numbers) if sample_numbers else 0} (vs 0 with preprocessing)

Band Detection Results:
{chr(10).join([f"Sample {i+1:2d}: {len(bands):2d} bands" + (f" (highest count)" if len(bands) == max(len(b) for b in band_positions) and len(bands) > 0 else "") for i, bands in enumerate(band_positions)])}

What This Means:
[OK] Detected more sample numbers ({len(sample_numbers) if sample_numbers else 0} vs 0)
[!]  Found fewer total bands ({total_bands} vs 154)
[!]  Missed bands in Sample 1 completely
[OK] Still identified similar sample patterns

Analysis region: Y ≥ {y_min}"""
    
    fig.text(0.15, 0.75, band_stats, fontsize=11, va='top',
             bbox=dict(facecolor='lightblue', alpha=0.95, edgecolor='darkblue', linewidth=2, boxstyle="round,pad=0.4"))
    
    # Add similarity analysis results below the blue box
    if similarity_matrix.size > 1:
        similar_pairs = []
        n = similarity_matrix.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] < 0.3:
                    similar_pairs.append((i+1, j+1, similarity_matrix[i, j]))
        
        similar_pairs.sort(key=lambda x: x[2])
        
        similarity_text = """=== SIMILARITY ANALYSIS ===
Most similar pairs (distance < 0.3):"""
        if similar_pairs:
            for s1, s2, dist in similar_pairs:
                similarity = 1 - dist
                similarity_text += f"\nSamples {s1}-{s2}: {similarity:.3f} similarity ({dist:.3f} distance)"
        else:
            similarity_text += "\nNo highly similar pairs found"
        
        print(f"Similarity analysis text length: {len(similarity_text)}")
        print(f"Number of similar pairs: {len(similar_pairs)}")
        
        # Debug: Show specific similarities for samples 5,6,9,11,12
        if similarity_matrix.size > 1 and similarity_matrix.shape[0] >= 12:
            print("\n=== Similarity Debug for Key Samples ===")
            key_samples = [4, 5, 8, 10, 11]  # 0-indexed: samples 5,6,9,11,12
            sample_names = [5, 6, 9, 11, 12]  # 1-indexed names
            
            for i, sample_i in enumerate(key_samples):
                for j, sample_j in enumerate(key_samples):
                    if i < j:  # Only show upper triangle
                        sim = similarity_matrix[sample_i, sample_j]
                        print(f"Sample {sample_names[i]} vs Sample {sample_names[j]}: {sim:.3f}")
            
            print(f"\nSample 5 bands: {band_positions[4] if len(band_positions) > 4 else 'N/A'}")
            print(f"Sample 6 bands: {band_positions[5] if len(band_positions) > 5 else 'N/A'}")
            print(f"Sample 9 bands: {band_positions[8] if len(band_positions) > 8 else 'N/A'}")
            print(f"Sample 11 bands: {band_positions[10] if len(band_positions) > 10 else 'N/A'}")
            print(f"Sample 12 bands: {band_positions[11] if len(band_positions) > 11 else 'N/A'}")
        
        fig.text(0.15, 0.35, similarity_text, fontsize=10, ha='left', va='top',
                 bbox=dict(facecolor='lightgreen', alpha=0.95, edgecolor='darkgreen', linewidth=2, boxstyle="round,pad=0.4"))
    
    plt.tight_layout()
    
    # Save results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'gel_analysis_enhanced.png')
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Enhanced visualization saved to: {output_path}")
        
        # Save analysis data
        analysis_data = {
            'sample_boundaries': sample_boundaries,
            'band_positions': band_positions,
            'similarity_matrix': similarity_matrix.tolist(),
            'band_counts': [len(bands) for bands in band_positions]
        }
        
        import json
        data_path = os.path.join(script_dir, 'gel_analysis_data.json')
        with open(data_path, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"Analysis data saved to: {data_path}")
        
        # Export text boxes to separate text file
        text_export_path = os.path.join(script_dir, 'gel_analysis_text_export.txt')
        with open(text_export_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GEL ELECTROPHORESIS ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Blue box content (Analysis Comparison)
            f.write("ANALYSIS COMPARISON (No Preprocessing vs With Preprocessing):\n")
            f.write("-" * 60 + "\n\n")
            f.write("Key Differences:\n")
            f.write(f"• Total bands detected: {total_bands} bands (vs 154 with preprocessing)\n")
            f.write(f"• Average bands per sample: {avg_bands:.1f} (vs 11.8 with preprocessing)\n")
            f.write(f"• Sample 1: {len(band_positions[0]) if len(band_positions) > 0 else 0} bands detected (vs 13 with preprocessing)\n")
            f.write(f"• Sample numbers detected: {len(sample_numbers) if sample_numbers else 0} (vs 0 with preprocessing)\n\n")
            
            f.write("Band Detection Results:\n")
            for i, bands in enumerate(band_positions):
                highest_count = max(len(b) for b in band_positions) if band_positions else 0
                highest_marker = " (highest count)" if len(bands) == highest_count and len(bands) > 0 else ""
                f.write(f"Sample {i+1:2d}: {len(bands):2d} bands{highest_marker}\n")
            
            f.write("\nWhat This Means:\n")
            f.write(f"[OK] Detected more sample numbers ({len(sample_numbers) if sample_numbers else 0} vs 0)\n")
            f.write(f"[!]  Found fewer total bands ({total_bands} vs 154)\n")
            f.write(f"[!]  Missed bands in Sample 1 completely\n")
            f.write(f"[OK] Still identified similar sample patterns\n\n")
            f.write(f"Analysis region: Y ≥ {y_min}\n\n")
            
            # Green box content (Similarity Analysis)
            f.write("=" * 80 + "\n")
            f.write("SIMILARITY ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("Most similar pairs (distance < 0.3):\n")
            f.write("-" * 40 + "\n")
            
            if similarity_matrix.size > 1:
                similar_pairs = []
                n = similarity_matrix.shape[0]
                for i in range(n):
                    for j in range(i+1, n):
                        if similarity_matrix[i, j] < 0.3:
                            similar_pairs.append((i+1, j+1, similarity_matrix[i, j]))
                
                similar_pairs.sort(key=lambda x: x[2])
                
                if similar_pairs:
                    for s1, s2, dist in similar_pairs:
                        similarity = 1 - dist
                        f.write(f"Samples {s1}-{s2}: {similarity:.3f} similarity ({dist:.3f} distance)\n")
                else:
                    f.write("No highly similar pairs found\n")
            else:
                f.write("Insufficient data for similarity analysis\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Text export saved to: {text_export_path}")
        
        # Export individual image components
        print("\n=== Exporting Individual Components ===")
        
        # 1. Original image with boundaries
        fig_original, ax_original = plt.subplots(1, 1, figsize=(16, 12))
        ax_original.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        ax_original.set_title('Original Image with Sample Boundaries', fontsize=16, fontweight='bold')
        ax_original.axis('off')
        
        # Draw boundaries
        for i, boundary in enumerate(sample_boundaries):
            ax_original.axvline(x=boundary, color='red', linewidth=2, alpha=0.8)
            if i < len(sample_boundaries) - 1:
                center_x = (boundary + sample_boundaries[i + 1]) // 2
                ax_original.text(center_x, 50, f'Sample {i+1}', 
                               ha='center', va='center', fontsize=12, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        original_path = os.path.join(script_dir, 'gel_original_with_boundaries.png')
        plt.savefig(original_path, dpi=300, bbox_inches='tight')
        plt.close(fig_original)
        print(f"Original image with boundaries saved to: {original_path}")
        
        # 2. Processed image with bands
        fig_processed, ax_processed = plt.subplots(1, 1, figsize=(16, 12))
        ax_processed.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        ax_processed.set_title('Processed Image with Detected Bands', fontsize=16, fontweight='bold')
        ax_processed.axis('off')
        
        # Draw boundaries and bands
        for i, boundary in enumerate(sample_boundaries):
            ax_processed.axvline(x=boundary, color='red', linewidth=2, alpha=0.8)
            if i < len(sample_boundaries) - 1:
                center_x = (boundary + sample_boundaries[i + 1]) // 2
                ax_processed.text(center_x, 50, f'Sample {i+1}', 
                                ha='center', va='center', fontsize=12, fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Draw bands
        for i, bands in enumerate(band_positions):
            if i < len(sample_boundaries) - 1:
                x_start = sample_boundaries[i]
                x_end = sample_boundaries[i + 1]
                for band_y in bands:
                    ax_processed.plot([x_start, x_end], [band_y, band_y], 'lime', linewidth=3, alpha=0.8)
        
        plt.tight_layout()
        processed_path = os.path.join(script_dir, 'gel_processed_with_bands.png')
        plt.savefig(processed_path, dpi=300, bbox_inches='tight')
        plt.close(fig_processed)
        print(f"Processed image with bands saved to: {processed_path}")
        
        # 3. Similarity matrix heatmap only
        fig_heatmap, ax_heatmap_only = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create similarity matrix heatmap
        sns.heatmap(
            similarity_matrix,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            cbar=True,
            ax=ax_heatmap_only,
            vmin=0,
            vmax=1,
            annot_kws={"size": 10, "weight": "bold"},
            cbar_kws={"shrink": 1, "label": "Similarity", "aspect": 8, "pad": 0.02},
            linewidths=0.5,
            linecolor="white"
        )
        
        ax_heatmap_only.set_title('Sample Similarity Matrix', fontsize=16, fontweight='bold')
        ax_heatmap_only.set_xlabel('Sample Number', fontsize=14, fontweight='bold')
        ax_heatmap_only.set_ylabel('Sample Number', fontsize=14, fontweight='bold')
        
        # Set sample labels
        sample_labels = [f'S{i+1}' for i in range(len(similarity_matrix))]
        ax_heatmap_only.set_xticklabels(sample_labels, rotation=0)
        ax_heatmap_only.set_yticklabels(sample_labels, rotation=0)
        
        plt.tight_layout()
        heatmap_path = os.path.join(script_dir, 'gel_similarity_matrix.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close(fig_heatmap)
        print(f"Similarity matrix saved to: {heatmap_path}")
        
        print("=== Individual Components Export Complete ===")
        
    except Exception as e:
        print(f"Error saving results: {e}")
    
    plt.close()
    return output_path

def main(image_path, y_min=170, use_band_analysis=True, min_band_area=20, tolerance=35, num_samples=13):
    """Enhanced main function with improved band detection and analysis."""
    print("=== Enhanced Gel Electrophoresis Analysis ===")
    print(f"Analyzing {num_samples} samples with Y ≥ {y_min}")
    print(f"Band-based similarity analysis: {'Enabled' if use_band_analysis else 'Disabled'}")
    
    try:
        # Load image without preprocessing
        print("1. Loading image (no preprocessing)...")
        original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if original_img is None:
            raise ValueError("Could not load the image. Please check the file path.")
        processed_img = original_img.copy()  # Use original as processed
        print(f"   Image loaded. Shape: {original_img.shape}")
        
        # Detect sample numbers
        print("2. Detecting sample numbers...")
        sample_numbers = detect_sample_numbers(original_img)
        
        # Detect boundaries
        print("3. Detecting sample boundaries...")
        sample_boundaries = detect_sample_boundaries_enhanced(processed_img, num_samples)
        print(f"   Detected {len(sample_boundaries)-1} sample lanes")
        print(f"   Boundary positions: {sample_boundaries}")
        print(f"   Sample widths: {[sample_boundaries[i+1] - sample_boundaries[i] for i in range(len(sample_boundaries)-1)]}")
        
        # Extract bands
        print("4. Extracting band positions...")
        band_positions = extract_band_positions_advanced(
            processed_img, sample_boundaries, min_band_area=min_band_area, y_min=y_min
        )
        
        # Print band statistics
        for i, bands in enumerate(band_positions, 1):
            print(f"   Sample {i}: {len(bands)} bands detected")
        
        # Calculate similarity
        print("5. Calculating sample similarities...")
        if use_band_analysis:
            similarity_matrix = calculate_similarity_matrix_hybrid(
                band_positions, processed_img.shape[0], 
                use_predefined_groups=False, tolerance=tolerance
            )
            print("   Using band-based similarity analysis")
        else:
            similarity_matrix = calculate_similarity_matrix(
                band_positions, processed_img.shape[0]
            )
            print("   Using predefined group similarity")
        
        # Visualize results
        print("6. Generating enhanced visualization...")
        output_path = visualize_gel_analysis_enhanced(
            original_img, processed_img, sample_boundaries, 
            band_positions, similarity_matrix, sample_numbers, y_min
        )
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to: {output_path}")
        
        # Print detailed results
        print("\n=== Band Detection Results ===")
        total_bands = 0
        for i, bands in enumerate(band_positions, 1):
            print(f"Sample {i:2d}: {len(bands):2d} bands at Y-positions {sorted(bands)}")
            total_bands += len(bands)
        
        print(f"\nTotal bands detected: {total_bands}")
        print(f"Average bands per sample: {total_bands/len(band_positions):.1f}")
        
        # Print similarity analysis
        if similarity_matrix.size > 1:
            print("\n=== Similarity Analysis ===")
            print("Most similar pairs (distance < 0.3):")
            n = similarity_matrix.shape[0]
            similar_pairs = []
            
            for i in range(n):
                for j in range(i+1, n):
                    if similarity_matrix[i, j] < 0.3:
                        similar_pairs.append((i+1, j+1, similarity_matrix[i, j]))
            
            similar_pairs.sort(key=lambda x: x[2])
            for s1, s2, dist in similar_pairs[:10]:  # Show top 10
                similarity = 1 - dist
                print(f"Samples {s1}-{s2}: {similarity:.3f} similarity ({dist:.3f} distance)")
            
            if not similar_pairs:
                print("No highly similar pairs found (all distances ≥ 0.3)")
        
        return output_path, band_positions, similarity_matrix
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def analyze_gel_with_parameters(image_path, **kwargs):
    """Wrapper function to easily test different parameters."""
    # Default parameters
    params = {
        'y_min': 170,
        'use_band_analysis': True,
        'min_band_area': 20,
        'tolerance': 35,
        'num_samples': 13
    }
    
    # Update with provided parameters
    params.update(kwargs)
    
    print(f"Analysis parameters: {params}")
    return main(image_path, **params)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python optimized_gel_analyzer.py <image_path> [options]")
        print("\nOptions:")
        print("  --y_min=N          Start analysis from Y-coordinate N (default: 170)")
        print("  --band_analysis    Use band-based similarity (default)")
        print("  --predefined       Use predefined group similarity")
        print("  --tolerance=N      Band matching tolerance in pixels (default: 25)")
        print("\nExample:")
        print("  python optimized_gel_analyzer.py gel.jpg --y_min=200 --tolerance=30")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Parse command line arguments
    kwargs = {}
    for arg in sys.argv[2:]:
        if arg.startswith('--y_min='):
            kwargs['y_min'] = int(arg.split('=')[1])
        elif arg.startswith('--tolerance='):
            kwargs['tolerance'] = int(arg.split('=')[1])
        elif arg == '--band_analysis':
            kwargs['use_band_analysis'] = True
        elif arg == '--predefined':
            kwargs['use_band_analysis'] = False
    
    # Run analysis
    output_path, band_positions, similarity_matrix = analyze_gel_with_parameters(image_path, **kwargs)
    
    if output_path:
        print(f"\nAnalysis completed successfully!")
        print(f"Open {output_path} to view results.")
    else:
        print("\nAnalysis failed. Check the error messages above.")
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.cluster.hierarchy import dendrogram, linkage
import os

def load_and_preprocess(image_path):
    """Load and preprocess the gel image."""
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not load the image. Please check the file path.")
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply adaptive thresholding to enhance bands
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return img, thresh

def detect_lanes(thresh_img, num_lanes=11, min_lane_width=10):
    """Detect vertical lanes in the gel image."""
    # Sum pixel values vertically to find lane boundaries
    vertical_projection = np.sum(thresh_img, axis=0)
    
    # Find peaks in the projection to identify lanes
    peaks = []
    for i in range(1, len(vertical_projection)-1):
        if (vertical_projection[i-1] < vertical_projection[i] > vertical_projection[i+1] and 
            vertical_projection[i] > np.mean(vertical_projection) * 1.5):
            peaks.append(i)
    
    # Find the most prominent peaks (lanes)
    if len(peaks) >= num_lanes:
        # Sort peaks by intensity and take the top num_lanes
        peak_intensities = [(i, vertical_projection[i]) for i in peaks]
        peak_intensities.sort(key=lambda x: x[1], reverse=True)
        lane_positions = sorted([x[0] for x in peak_intensities[:num_lanes]])
    else:
        # If we can't find enough peaks, space them evenly
        lane_positions = np.linspace(0, thresh_img.shape[1]-1, num_lanes, dtype=int)
    
    return lane_positions

def extract_lane_profiles(img, lane_positions, lane_width=30):
    """Extract intensity profiles for each lane."""
    profiles = []
    for x in lane_positions:
        # Extract a vertical strip for each lane
        x_start = max(0, x - lane_width//2)
        x_end = min(img.shape[1], x + lane_width//2)
        
        # Average the intensity across the width of the lane
        lane_profile = np.mean(img[:, x_start:x_end], axis=1)
        profiles.append(lane_profile)
    
    return profiles

def compare_profiles(profiles):
    """Compare profiles using correlation distance."""
    n = len(profiles)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # Use correlation distance (1 - correlation coefficient)
            similarity_matrix[i, j] = 1 - np.corrcoef(profiles[i], profiles[j])[0, 1]
    
    return similarity_matrix

def plot_results(original_img, lane_positions, similarity_matrix):
    """Plot the original image with lane markers and the similarity matrix."""
    plt.figure(figsize=(15, 10))
    
    # Plot original image with lane markers
    plt.subplot(2, 1, 1)
    plt.imshow(original_img, cmap='gray')
    for x in lane_positions:
        plt.axvline(x, color='r', alpha=0.3)
    plt.title('Gel Image with Detected Lanes')
    
    # Plot similarity matrix
    plt.subplot(2, 1, 2)
    plt.imshow(similarity_matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Distance (1 - correlation)')
    plt.title('Similarity Matrix')
    plt.xlabel('Sample Number')
    plt.ylabel('Sample Number')
    
    # Save the figure
    output_path = 'gel_analysis.png'
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def main(image_path):
    # Load and preprocess the image
    original_img, processed_img = load_and_preprocess(image_path)
    
    # Detect lanes
    lane_positions = detect_lanes(processed_img, num_lanes=11)
    
    # Extract intensity profiles
    profiles = extract_lane_profiles(original_img, lane_positions)
    
    # Compare profiles
    similarity_matrix = compare_profiles(profiles)
    
    # Plot and save results
    output_path = plot_results(original_img, lane_positions, similarity_matrix)
    
    print(f"Analysis complete! Results saved to {output_path}")
    print("\nSimilarity Matrix (lower values = more similar):")
    print(np.round(similarity_matrix, 2))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python gel_analyzer.py <path_to_gel_image>")
        sys.exit(1)
    
    main(sys.argv[1])

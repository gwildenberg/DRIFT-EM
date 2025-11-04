#!/usr/bin/env python3

import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2

def load_magc_file(magc_path):
    """
    Load polygons from a .magc file.
    .magc files are INI-style files containing polygon coordinates.
    """
    try:
        polygons = []
        current_section = None
        
        with open(magc_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Check if this is a section header
            if line.startswith('[section.') and line.endswith(']'):
                current_section = line[1:-1]  # Remove the brackets
            
            # If we're in a section and this is a polygon line
            elif current_section is not None and line.startswith('polygon ='):
                # Extract polygon coordinates
                coords_str = line.split('=')[1].strip()
                coords = [float(x) for x in coords_str.split(',')]
                
                # Group coordinates into (x,y) pairs
                points = [(coords[j], coords[j+1]) for j in range(0, len(coords), 2)]
                
                # Add the polygon to our list
                polygons.append(points)
        
        print(f"Loaded {len(polygons)} polygons from {magc_path}")
        return polygons
    except Exception as e:
        print(f"Error loading .magc file: {e}")
        return []

def extract_red_outline(outline_image_path):
    """
    Extract the red outline from an image.
    Returns the outline as a binary mask.
    """
    try:
        # Load the image using OpenCV (which loads in BGR format)
        img = cv2.imread(outline_image_path)
        if img is None:
            raise ValueError(f"Could not load image: {outline_image_path}")
        
        # Create mask to extract red areas
        # Red pixels have high R and low G/B values
        # Define lower and upper bounds for "red" color in BGR format
        lower_red = np.array([0, 0, 180])  # BGR: low blue, low green, high red
        upper_red = np.array([80, 80, 255])  # BGR: some blue/green allowed, high red
        
        # Create a mask that isolates red pixels
        mask = cv2.inRange(img, lower_red, upper_red)
        
        # Find contours in the mask to get the outline shape
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("No red outline found in the image")
            return None
        
        # Find the largest contour (which should be our outline)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Convert the contour to polygon points (divide by 10 to match MAGC format)
        outline_points = []
        for point in largest_contour.reshape(-1, 2):
            outline_points.append((float(point[0]), float(point[1])))
        
        print(f"Extracted outline with {len(outline_points)} points")
        return outline_points
    except Exception as e:
        print(f"Error extracting red outline: {e}")
        return None

def calculate_polygon_overlap(green_polygons, red_polygons, outline_polygon, image_size):
    """
    Calculate overlap between green and red polygons within the outline region.
    Returns overlap statistics.
    """
    try:
        # Create blank images to draw the polygons for overlap calculation
        width, height = image_size
        green_mask = np.zeros((height, width), dtype=np.uint8)
        red_mask = np.zeros((height, width), dtype=np.uint8)
        outline_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Draw the green polygons
        for polygon in green_polygons:
            # Scale points
            scaled_points = [(int(x/10), int(y/10)) for x, y in polygon]
            # Convert to numpy array and proper shape for cv2
            points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(green_mask, [points], 255)
        
        # Draw the red polygons
        for polygon in red_polygons:
            # Scale points
            scaled_points = [(int(x/10), int(y/10)) for x, y in polygon]
            # Convert to numpy array and proper shape for cv2
            points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(red_mask, [points], 255)
        
        # Draw the outline polygon
        scaled_points = [(int(x), int(y)) for x, y in outline_polygon]
        points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(outline_mask, [points], 255)
        
        # Calculate areas
        total_area_outline = cv2.countNonZero(outline_mask)
        
        # Calculate areas within the outline
        green_in_outline = cv2.bitwise_and(green_mask, green_mask, mask=outline_mask)
        red_in_outline = cv2.bitwise_and(red_mask, red_mask, mask=outline_mask)
        
        # Calculate overlap
        overlap_mask = cv2.bitwise_and(green_in_outline, red_in_outline)
        
        # Calculate all the metrics
        green_area = cv2.countNonZero(green_in_outline)
        red_area = cv2.countNonZero(red_in_outline)
        overlap_area = cv2.countNonZero(overlap_mask)
        
        # Calculate union (red OR green)
        union_mask = cv2.bitwise_or(green_in_outline, red_in_outline)
        union_area = cv2.countNonZero(union_mask)
        
        # Calculate Intersection over Union (IoU)
        iou = overlap_area / union_area if union_area > 0 else 0
        
        # Calculate Dice coefficient
        dice = (2 * overlap_area) / (green_area + red_area) if (green_area + red_area) > 0 else 0
        
        # Calculate precision and recall
        precision = overlap_area / red_area if red_area > 0 else 0
        recall = overlap_area / green_area if green_area > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'total_outline_area': total_area_outline,
            'green_area': green_area,
            'red_area': red_area,
            'overlap_area': overlap_area,
            'union_area': union_area,
            'green_coverage': green_area / total_area_outline if total_area_outline > 0 else 0,
            'red_coverage': red_area / total_area_outline if total_area_outline > 0 else 0,
            'overlap_percentage': overlap_area / total_area_outline if total_area_outline > 0 else 0,
            'iou': iou,
            'dice_coefficient': dice,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        print(f"Error calculating polygon overlap: {e}")
        return None

def create_visualization(image_path, green_polygons, red_polygons, outline_points, output_path):
    """
    Create a visualization of the polygons and outline using OpenCV for better control.
    """
    try:
        # Load the image using OpenCV
        img = cv2.imread(image_path)
        
        # Get image dimensions
        img_height, img_width = img.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Create separate masks for green and red polygons
        green_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        red_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Draw green polygons on mask
        for polygon in green_polygons:
            # Scale points
            scaled_points = [(int(x/10), int(y/10)) for x, y in polygon]
            # Convert to numpy array for OpenCV
            points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            # Draw polygon on mask
            cv2.fillPoly(green_mask, [points], 255)
        
        # Draw red polygons on mask
        for polygon in red_polygons:
            # Scale points
            scaled_points = [(int(x/10), int(y/10)) for x, y in polygon]
            # Convert to numpy array for OpenCV
            points = np.array(scaled_points, dtype=np.int32).reshape((-1, 1, 2))
            # Draw polygon on mask
            cv2.fillPoly(red_mask, [points], 255)
        
        # Create colored masks
        green_colored = np.zeros_like(img)
        green_colored[:, :, 1] = green_mask  # Green channel
        
        red_colored = np.zeros_like(img)
        red_colored[:, :, 2] = red_mask  # Red channel
        
        # Create yellow mask for overlapping regions
        overlap_mask = cv2.bitwise_and(green_mask, red_mask)
        yellow_colored = np.zeros_like(img)
        yellow_colored[:, :, 0] = 0        # Blue channel = 0
        yellow_colored[:, :, 1] = overlap_mask  # Green channel
        yellow_colored[:, :, 2] = overlap_mask  # Red channel
        
        # Create the final image
        # Start with the original image
        result = img.copy()
        
        # Add green areas (only where there is no overlap)
        green_only_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(overlap_mask))
        green_only_colored = np.zeros_like(img)
        green_only_colored[:, :, 1] = green_only_mask
        
        # Add red areas (only where there is no overlap)
        red_only_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(overlap_mask))
        red_only_colored = np.zeros_like(img)
        red_only_colored[:, :, 2] = red_only_mask
        
        # Alpha blending for transparency
        alpha = 0.5  # 50% opacity
        
        # Apply colored overlays
        result = cv2.addWeighted(result, 1, green_only_colored, alpha, 0)
        result = cv2.addWeighted(result, 1, red_only_colored, alpha, 0)
        result = cv2.addWeighted(result, 1, yellow_colored, alpha, 0)
        
        # Draw the pink outline
        if len(outline_points) > 2:
            # Convert outline points to integers
            outline_points_int = [(int(x), int(y)) for x, y in outline_points]
            # Convert to numpy array
            outline_points_array = np.array(outline_points_int, dtype=np.int32).reshape((-1, 1, 2))
            # Draw polyline
            cv2.polylines(result, [outline_points_array], True, (255, 0, 255), 2)
        
        # Save the output image
        cv2.imwrite(output_path, result)
        print(f"Visualization saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # Paths
    image_path = "/Users/kevin/PycharmProjects/atlas/5x_gregc_ORG_10p.tif"
    green_magc_path = "/Users/kevin/PycharmProjects/atlas/Gregc5x.magc"
    red_magc_path = "/Users/kevin/PycharmProjects/atlas/5x_gregc_ORG_10p_20250326_160314.magc"
    outline_image_path = "/Users/kevin/PycharmProjects/atlas/5x_gregc_ORG_10p_with_polygons_outline.tif"
    output_path = "/Users/kevin/PycharmProjects/atlas/polygon_overlap_analysis.tif"
    
    # 1. Load polygons from MAGC files
    green_polygons = load_magc_file(green_magc_path)
    red_polygons = load_magc_file(red_magc_path)
    
    # 2. Extract red outline from outline image
    outline_points = extract_red_outline(outline_image_path)
    
    if outline_points:
        # 3. Create visualization
        success = create_visualization(image_path, green_polygons, red_polygons, outline_points, output_path)
        
        # 4. Calculate overlap metrics
        if success:
            # Open the image to get its dimensions
            img = Image.open(image_path)
            img_size = img.size
            
            overlap_metrics = calculate_polygon_overlap(green_polygons, red_polygons, outline_points, img_size)
            
            if overlap_metrics:
                print("\nPolygon Overlap Analysis:")
                print(f"Total outline area: {overlap_metrics['total_outline_area']} pixels")
                print(f"Green polygon area: {overlap_metrics['green_area']} pixels ({overlap_metrics['green_coverage']*100:.2f}% of outline)")
                print(f"Red polygon area: {overlap_metrics['red_area']} pixels ({overlap_metrics['red_coverage']*100:.2f}% of outline)")
                print(f"Overlap area: {overlap_metrics['overlap_area']} pixels ({overlap_metrics['overlap_percentage']*100:.2f}% of outline)")
                print("\nMetrics:")
                print(f"Intersection over Union (IoU): {overlap_metrics['iou']:.4f}")
                print(f"Dice Coefficient: {overlap_metrics['dice_coefficient']:.4f}")
                print(f"Precision: {overlap_metrics['precision']:.4f}")
                print(f"Recall: {overlap_metrics['recall']:.4f}")
                print(f"F1 Score: {overlap_metrics['f1_score']:.4f}")
            else:
                print("Failed to calculate overlap metrics.")
        else:
            print("Failed to create visualization.")
    else:
        print("Failed to extract outline. Analysis cannot continue.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import random  # For selecting random contours
from scipy.optimize import minimize  # For advanced optimization
import datetime  # For timestamped output filenames
import os  # For path operations
import pandas as pd  # For creating Excel files

# Set fixed random seed for reproducibility
random.seed(42)  # Always use the same random seed for consistent results
np.random.seed(42)

def process_image(image_path):
    """
    Process an image to detect contours with edge subtraction method using Sobel.
    
    Args:
        image_path (str): Path to the input image
    
    Returns:
        tuple: (image, binary, contours, edges, binary_minus_edges)
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Fixed threshold at value 86 (for improved rectangle detection)
    _, binary = cv2.threshold(gray, 86, 255, cv2.THRESH_BINARY_INV)
    
    # Apply contrast enhancement to emphasize gradients
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Apply Sobel operators with large kernel size
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=5)
    
    # Calculate magnitude
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize to 0-255 range
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply a low threshold to capture subtle edges
    _, edges = cv2.threshold(magnitude, 20, 255, cv2.THRESH_BINARY)
    
    # Apply three iterations of dilation with a 4x4 kernel
    kernel = np.ones((4, 4), np.uint8)
    dilated_edges = edges.copy()
    for _ in range(3):
        dilated_edges = cv2.dilate(dilated_edges, kernel, iterations=1)
    
    # Use the dilated edges for subtraction
    binary_minus_edges = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(dilated_edges))
    
    # Find only external contours using the binary_minus_edges image
    contours, _ = cv2.findContours(binary_minus_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return the dilated edges instead of the original edges
    return image, binary_minus_edges, contours, dilated_edges, binary_minus_edges

def optimize_rectangle_for_contour(image, contour, target_width=36, target_height=24, occupied_mask=None):
    # Store the grayscale image for optimization
    # We will optimize to find the darkest region (lowest pixel values)
    gray_for_opt = image.copy()
    
    # Create a binary mask from the contour
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Shift contour to the local coordinates of the bounding rectangle
    shifted_contour = contour - np.array([x, y])
    
    # Draw the contour on the mask
    cv2.drawContours(mask, [shifted_contour], 0, 255, -1)  # Fill the contour
    
    # Calculate moments from the binary mask
    M = cv2.moments(mask)
    
    if M["m00"] != 0:
        # Get center of mass in local coordinates
        local_cx = int(M["m10"] / M["m00"])
        local_cy = int(M["m01"] / M["m00"])
        
        # Convert to global coordinates
        cx = x + local_cx
        cy = y + local_cy
        
        # Calculate initial angle from moments (more accurate orientation)
        # Using central moments to get principal axes orientation
        mu20 = M["mu20"] / M["m00"]
        mu02 = M["mu02"] / M["m00"]
        mu11 = M["mu11"] / M["m00"]
        
        # Calculate the orientation angle in degrees
        if mu20 != mu02:
            moments_angle = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
            # Normalize to 0-180 range
            moments_angle = moments_angle % 180
        else:
            moments_angle = 0  # Default to horizontal
    else:
        # Fallback to bounding rect center (should rarely happen)
        cx, cy = x + w//2, y + h//2
        moments_angle = 0  # Default to horizontal
        
    # Choose initial angle (prefer moments if available)
    initial_angle = moments_angle
    
    # Normalize angle to 0-180 range for consistency
    initial_angle = initial_angle % 180
    
    # Cache for optimization - prevents recreating the image on every iteration
    h, w = gray_for_opt.shape[:2]
    
    def rect_cost_components(params):
        """
        Calculate the individual cost components for a rectangle.
        
        Args:
            params: [center_x, center_y, angle_radians]
            
        Returns:
            tuple: (brightness_cost, angle_penalty_cost, overlap_penalty)
        """
        cx, cy, angle_rad = params
        angle_deg = np.degrees(angle_rad)
        
        # Constrain to image boundaries
        cx = np.clip(cx, 0, w-1)
        cy = np.clip(cy, 0, h-1)
        
        # Create a rotated rectangle
        rect = ((cx, cy), (target_width, target_height), angle_deg)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        # Check for overlap with existing rectangles using the occupied mask
        overlap_penalty = 0.0
        if occupied_mask is not None:
            # Create a mask for this candidate rectangle
            current_rect_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(current_rect_mask, [box], 1)
            
            # Check if this rectangle overlaps with any existing rectangles
            # by comparing with the occupied mask
            overlap = np.logical_and(current_rect_mask, occupied_mask).sum()
            
            # If there's any overlap, apply a very large penalty
            if overlap > 0:
                overlap_penalty = np.sum(overlap)
        
        # Create a mask for this rectangle for brightness calculation
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [box], 255)
        
        # Extract region using the mask and calculate the sum of pixel values
        # This will measure the total "brightness" inside the rectangle
        # For a grayscale image, lower values mean darker pixels
        masked_region = cv2.bitwise_and(gray_for_opt, gray_for_opt, mask=mask)
        
        # Count non-zero pixels (where mask is applied)
        non_zero_count = cv2.countNonZero(mask)
        
        # Default to high cost if no pixels in mask
        if non_zero_count == 0:
            return float('inf'), 0.0, float('inf')
            
        brightness = np.sum(masked_region) / non_zero_count

        current_angle = angle_deg % 180
        contour_angle = initial_angle % 180
        
        # Calculate angle deviation cost (penalty for rotating too far from initial angle)
        # Handle the discontinuity at 0/180 degrees
        angle_diff = min(
            abs(current_angle - contour_angle),  # Regular difference
            abs(current_angle - (contour_angle + 180) % 180),  # Wrap around one way
            abs((current_angle + 180) % 180 - contour_angle)   # Wrap around other way
        )
        
        angle_penalty = angle_diff * 0.75
        
        return float(brightness), float(angle_penalty), float(overlap_penalty)
    
    def rect_overlap_cost(params):
        """
        Cost function for optimization: sum of pixel values inside the rectangle
        (to be minimized). Lower pixel values (darker) are better.
        
        Args:
            params: [center_x, center_y, angle_radians]
            
        Returns:
            float: Sum of pixel values (lower is better, indicating darker area)
        """
        brightness, angle_penalty, overlap_penalty = rect_cost_components(params)
        
        # Combine the costs - brightness is our main term, angle penalty is secondary
        combined_cost = brightness + angle_penalty + overlap_penalty
        
        return combined_cost
    
    # Initial parameters [x, y, angle_radians]
    initial_params = [cx, cy, np.radians(initial_angle)]
    # Bounds for the parameters:
    # x: can move ±5 pixels from center
    # y: can move ±5 pixels from center
    # angle: can rotate ±90 degrees from initial angle
    bounds = [
        (max(0, cx - 5), min(w-1, cx + 5)),
        (max(0, cy - 5), min(h-1, cy + 5)),
        (np.radians(initial_angle - 90), np.radians(initial_angle + 90))
    ]
    print(bounds)
    
    # Use the minimize function for optimization
    result = minimize(
        rect_overlap_cost,
        initial_params,
        method='L-BFGS-B',  # Works well with bounds
        bounds=bounds,
        options={'maxiter': 100}
    )
    
    # Get the optimized parameters
    opt_cx, opt_cy, opt_angle_rad = result.x
    opt_angle_deg = np.degrees(opt_angle_rad) % 180
    
    # Calculate the final cost for reporting
    final_cost = rect_overlap_cost([opt_cx, opt_cy, opt_angle_rad])
    
    return {
        'center_x': int(opt_cx),
        'center_y': int(opt_cy),
        'angle': opt_angle_deg,
        'width': target_width,
        'height': target_height,
        'score': final_cost,
        'success': result.success
    }

def save_debug_image(image_path, min_area=0, max_area=float('inf'), num_rects=5, save_excel=True):
    """
    Save high-resolution debug images showing:
    A) Original image with optimized 24x36 rectangles
    B) Detected contours with random colors (contours outside size limits are shown in gray)
    C) Edge detection image (Sobel)
    D) Binary minus edges (the processed image used for detection)
    
    Args:
        image_path (str): Path to the input image
        min_area (int): Minimum contour area to highlight in color
        max_area (int): Maximum contour area to highlight in color
        num_rects (int): Number of random contours to optimize and display
        save_excel (bool): Whether to save contour data to Excel file

    Returns:
        str: Path to the saved debug image
    """
    # Generate timestamp for unique filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get input file name without extension
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Create output path with timestamp in current directory
    output_path = f"debug_{base_name}_{timestamp}.tiff"
    
    print(f"Processing debug image for {image_path}...")
    print(f"Output will be saved to {output_path}")
    

    # Process image with edge subtraction and Sobel method
    print(f"Steps 1-5: Processing image using edge subtraction with Sobel...")
    
    # Use edge subtraction method
    image, binary, contours, edges, binary_minus_edges = process_image(image_path)
    
    # Save contour data to Excel if requested
    if save_excel:
        print("Saving contour data to Excel file...")
        contour_data = []
        
        for i, cnt in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(cnt)
            
            # Calculate center of mass using moments
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Fallback to bounding rect center if moments calculation fails
                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w//2, y + h//2
            
            # Calculate bounding rectangle dimensions
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Calculate rectangle fit quality
            # Get the minimum area rectangle that fits the contour
            rect = cv2.minAreaRect(cnt)
            rect_width, rect_height = rect[1]
            
            # Make sure width is the smaller dimension and height is the larger one
            if rect_width > rect_height:
                rect_width, rect_height = rect_height, rect_width
                
            # Calculate area of the min area rectangle
            rect_area = rect_width * rect_height
            
            # Calculate fill ratio (how well the contour fills the rectangle)
            # Higher ratio means better rectangle fit
            fill_ratio = area / rect_area if rect_area > 0 else 0
            
            # Calculate aspect ratio of the rectangle (height/width)
            aspect_ratio = rect_height / rect_width if rect_width > 0 else 0
            
            # Calculate how well the contour fits a 9x20 rectangle
            # Target aspect ratio is 20/9 = 2.22
            target_aspect_ratio = 20.0 / 9.0
            
            # Calculate aspect ratio fit score (lower is better)
            aspect_ratio_fit = abs(aspect_ratio - target_aspect_ratio)
            
            # Calculate size fit (how close is the contour to the target size)
            target_area = 9 * 20
            # Avoid division by zero
            if area == 0:
                area_ratio = 0  # If area is zero, it's a poor fit
            else:
                area_ratio = min(area / target_area, target_area / area)  # Between 0 and 1, higher is better
            
            # Combined 9x20 fit score (higher is better)
            # Weigh aspect ratio fit more heavily
            rect_9x20_fit = (0.7 * (1 - aspect_ratio_fit / 3.0)) + (0.3 * area_ratio)
            
            # Clamp the score between 0 and 1
            rect_9x20_fit = max(0, min(1, rect_9x20_fit))
            
            # Add data to the list
            contour_data.append({
                'Contour_ID': i,
                'Area': area,
                'Center_X': cx,
                'Center_Y': cy,
                'Bounding_Box_X': x,
                'Bounding_Box_Y': y,
                'Bounding_Box_Width': w,
                'Bounding_Box_Height': h,
                'Min_Rect_Width': round(rect_width, 2),
                'Min_Rect_Height': round(rect_height, 2),
                'Min_Rect_Area': round(rect_area, 2),
                'Rectangle_Fill_Ratio': round(fill_ratio, 4),
                'Aspect_Ratio': round(aspect_ratio, 4),
                'Rect_9x20_Fit_Score': round(rect_9x20_fit, 4)
            })
        
        # Create DataFrame and save to Excel
        df = pd.DataFrame(contour_data)
        
        # Sort by 9x20 rectangle fit score (highest first)
        df = df.sort_values(by='Rect_9x20_Fit_Score', ascending=False)
        
        df.to_excel('d.xlsx', index=False)
        print(f"Saved data for {len(contours)} contours to d.xlsx (sorted by 9x20 rectangle fit)")
    
    # Create a copy of the original image for panel A
    panel_a_image = image.copy()
    
    # Create a 2x2 grid
    print("Step 6: Creating composite image grid...")
    # Get dimensions from the input image
    img_height, img_width = image.shape[:2]
    
    # Set fixed output dimensions for debug view
    height, width = binary.shape  # Standard debug size for all panels
    scale_factor = 1  # Scale factor for output resolution
    combined_height = height * 2 * scale_factor
    combined_width = width * 2 * scale_factor
    combined = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)
    
    # Convert grayscale images to BGR for visualization
    print("Step 7: Converting images for visualization...")
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    binary_minus_edges_bgr = cv2.cvtColor(binary_minus_edges, cv2.COLOR_GRAY2BGR)
    
    # Ensure panel_a_image is in BGR format if it's grayscale
    if len(panel_a_image.shape) == 2 or panel_a_image.shape[2] == 1:
        panel_a_image = cv2.cvtColor(panel_a_image, cv2.COLOR_GRAY2BGR)
    
    # No need to resize if scale_factor is 1
    print("Step 8: Preparing images for visualization...")
    
    # Create contour visualization image
    print("Step 9a: Creating contour visualization image with size filtering...")
    contour_image = image.copy()
    
    # Count valid contours (within size limits)
    valid_contours = 0
    valid_contour_list = []
    
    # Generate a random color for each contour
    for i, contour in enumerate(contours):
        # Get contour area
        area = cv2.contourArea(contour)
        
        # Check if contour is within size limits
        if min_area <= area <= max_area:
            # Valid contour - use a random bright color
            color = np.random.randint(0, 255, size=3).tolist()
            
            # Ensure the color has at least one bright channel for visibility
            max_channel = max(color)
            if max_channel < 150:  # If all channels are dim
                bright_channel = np.random.randint(0, 3)  # Choose a random channel to make bright
                color[bright_channel] = 255  # Make that channel bright
                
            valid_contours += 1
            valid_contour_list.append((contour, color))
        else:
            # Invalid contour - use gray
            color = [120, 120, 120]  # Gray color for contours outside size limits
        
        # Draw contour outline slightly thicker for better visibility
        cv2.drawContours(contour_image, [contour], -1, color, 2)
    
    # Optimize and draw rectangles if we have valid contours
    if valid_contour_list:
        # Create a list of contours to optimize, picking contours with highest 9x20 fit score
        valid_contours_with_score = []
        
        for contour, color in valid_contour_list:
            # Calculate how well this contour fits a 9x20 rectangle
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(contour)
            rect_width, rect_height = rect[1]
            
            # Make sure width is the smaller dimension
            if rect_width > rect_height:
                rect_width, rect_height = rect_height, rect_width
                
            # Calculate aspect ratio
            aspect_ratio = rect_height / rect_width if rect_width > 0 else 0
            
            # Target aspect ratio for 9x20
            target_aspect_ratio = 20.0 / 9.0
            
            # Aspect ratio fit score (lower is better)
            aspect_ratio_fit = abs(aspect_ratio - target_aspect_ratio)
            
            # Calculate area
            area = cv2.contourArea(contour)
            target_area = 9 * 20
            # Avoid division by zero
            if area == 0:
                area_ratio = 0  # If area is zero, it's a poor fit
            else:
                area_ratio = min(area / target_area, target_area / area)  # Between 0 and 1, higher is better
            
            # Combined fit score (higher is better)
            fit_score = (0.7 * (1 - aspect_ratio_fit / 3.0)) + (0.3 * area_ratio)
            fit_score = max(0, min(1, fit_score))
            
            valid_contours_with_score.append((contour, color, fit_score))
        
        # Sort by fit score (highest first)
        valid_contours_with_score.sort(key=lambda x: x[2], reverse=True)
        
        # Select top contours (up to num_rects)
        num_to_optimize = min(num_rects, len(valid_contours_with_score))
        contours_to_optimize = [(c, color) for c, color, _ in valid_contours_with_score[:num_to_optimize]]
        
        # Log that optimization is being performed
        print(f"Optimizing {num_to_optimize} rectangles (sorted by 9x20 fit)...")
        
        # Create a transparent overlay for the optimized rectangles
        overlay = panel_a_image.copy()
        
        # Draw the original contours in blue
        for contour, _ in contours_to_optimize:
            cv2.drawContours(overlay, [contour], 0, (255, 0, 0), 1)  # Thin blue line
        
        # Create a mask to keep track of occupied areas
        h, w = image.shape[:2]
        occupied_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Keep track of optimized rectangles (for visualization)
        optimized_rectangles = []
        
        # Now optimize each contour and draw the rectangles
        for i, (contour, _) in enumerate(contours_to_optimize):
            # Show progress in console
            print(f"  Optimizing rectangle {i+1}/{num_to_optimize}...")
            
            # Optimize the rectangle placement, passing the occupied mask to avoid overlap
            opt_rect = optimize_rectangle_for_contour(image, contour, occupied_mask=occupied_mask)
            
            # Only use rectangles where optimization succeeded (not infinite cost)
            if opt_rect['score'] < float('inf'):
                # Add this rectangle to our list of optimized rectangles
                optimized_rectangles.append(opt_rect)
                
                # Update the occupied mask with this rectangle
                rect_center = (opt_rect['center_x'], opt_rect['center_y'])
                rect_size = (opt_rect['width'], opt_rect['height'])
                rect_angle = opt_rect['angle']
                
                # Create box points for the rectangle
                rect_box = cv2.boxPoints((rect_center, rect_size, rect_angle))
                rect_box = rect_box.astype(np.int32)
                
                # Add this rectangle to the occupied mask
                cv2.fillPoly(occupied_mask, [rect_box], 1)
                
                # Draw the optimized rectangle
                center = (opt_rect['center_x'], opt_rect['center_y'])
                size = (opt_rect['width'], opt_rect['height'])
                angle = opt_rect['angle']
                
                # Create the rotated rectangle points
                rect = (center, size, angle)
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                
                # Draw with a thin green line
                cv2.drawContours(overlay, [box], 0, (0, 255, 0), 1)  # Thin green line
                
                # Log the optimization result
                print(f"    Rectangle at ({center[0]}, {center[1]}), angle: {angle:.1f}°, score: {opt_rect['score']:.0f}")
            else:
                print("failure")
        # Apply the overlay with transparency
        alpha = 0.6  # 60% opacity (more visible than before)
        cv2.addWeighted(overlay, alpha, panel_a_image, 1 - alpha, 0, panel_a_image)
        
        print(f"  Drew {num_to_optimize} optimized rectangles with semi-transparency")
        
        # Save optimized rectangles to a .magc file with values multiplied by 10
        if optimized_rectangles:
            print("Saving optimized rectangles to .magc file...")
            # Create a .magc file using the base name of the input image
            magc_output_path = f"{base_name}_{timestamp}.magc"
            
            with open(magc_output_path, 'w') as f:
                # Write the header with the number of sections
                f.write("[sections]\n")
                f.write(f"number = {len(optimized_rectangles)}\n\n")
                
                # Write each section
                for i, rect in enumerate(optimized_rectangles):
                    # Extract rectangle information
                    center_x = rect['center_x'] * 10  # Multiply by 10 as requested
                    center_y = rect['center_y'] * 10  # Multiply by 10 as requested
                    width_here = rect['width'] * 10        # Multiply by 10 as requested
                    height_here = rect['height'] * 10      # Multiply by 10 as requested
                    angle = rect['angle']
                    
                    # Calculate the corner points of the rectangle
                    rect_points = cv2.boxPoints(((rect['center_x'], rect['center_y']), 
                                              (rect['width'], rect['height']), 
                                              rect['angle']))
                    rect_points = rect_points.astype(np.float32)
                    
                    # Multiply the points by 10 for the .magc file
                    rect_points *= 10
                    
                    # Format the polygon string
                    polygon_str = ",".join([f"{p[0]:.1f},{p[1]:.1f}" for p in rect_points])
                    
                    # Calculate area (width * height * 10^2 since both dimensions are multiplied by 10)
                    area = width_here * height_here
                    
                    # Write the section
                    f.write(f"[section.{i:04d}]\n")
                    f.write(f"polygon = {polygon_str}\n")
                    f.write(f"center = {center_x:.2f},{center_y:.2f}\n")
                    f.write(f"area = {area:.1f}\n")
                    f.write(f"angle = {angle}\n\n")
                
            print(f"Saved {len(optimized_rectangles)} rectangles to {magc_output_path} with values multiplied by 10")
    
    # Ensure contour_image is in BGR format if it's grayscale
    if len(contour_image.shape) == 2 or contour_image.shape[2] == 1:
        contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    
    # Place each image in the grid
    print("Step 9b: Arranging images in grid...")

    combined[0:height, 0:width] = panel_a_image              # A) Original with optimized rectangles
    combined[0:height, width:width*2] = contour_image        # B) Detected Contours
    combined[height:height*2, 0:width] = edges_bgr           # C) Edge detection
    combined[height:height*2, width:width*2] = binary_minus_edges_bgr  # D) Binary minus Edges
    
    # Add text labels
    print("Step 10: Adding text labels...")
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5  # Smaller font scale for fixed-size debug view
    font_thickness = 1
    cv2.putText(combined, "A) Optimized 24x36 Rectangles", (10, 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "B) Detected Contours", (width + 10, 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "C) Edge Detection (Sobel)", (10, height + 20), font, font_scale, (0, 255, 255), font_thickness)
    cv2.putText(combined, "D) Binary minus Edges", (width + 10, height + 20), font, font_scale, (0, 255, 255), font_thickness)
    
    # Add contour count as text overlay
    total_contours = len(contours)
    cv2.putText(combined, f"Total Contours: {total_contours} (Valid: {valid_contours})", 
               (10, combined_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # Add size thresholds and method info
    cv2.putText(combined, f"Size range: {min_area}-{max_area}", 
               (10, combined_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    # Add timestamp to the image
    cv2.putText(combined, f"Time: {timestamp}", 
               (combined_width // 2, combined_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    
    cv2.putText(combined, "Method: Edge Subtraction (Sobel)", 
               (combined_width // 2, combined_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, (255, 255, 255), 1)
    

    # Save the debug image as TIFF
    print("Step 12: Writing output TIFF image...")
    cv2.imwrite(output_path, combined)  # TIFF format preserves full quality
    print(f"Debug image saved to {output_path}")
    print("Debug image processing completed successfully!")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate debug image visualization')
    parser.add_argument('--image_path', default="5x_gregc_ORG_10p.tif", help='Path to the input image')
    parser.add_argument('--min-area', type=int, default=1, help='Minimum contour area')
    parser.add_argument('--max-area', type=int, default=400, help='Maximum contour area')
    parser.add_argument('--num-rects', type=int, default=5000,
                        help='Number of random contours to optimize (default: 5)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for consistent contour selection (default: 42)')
    parser.add_argument('--excel', action='store_true', default=True,
                        help='Save contour data to Excel file (default: True)')
    
    args = parser.parse_args()
    random.seed(args.random_seed)
    # Save debug image with automatic timestamped filename
    output_path = save_debug_image(
        args.image_path, 
        args.min_area, 
        args.max_area,
        args.num_rects,
        args.excel)

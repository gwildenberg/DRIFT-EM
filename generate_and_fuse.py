#!/usr/bin/env python3
import random
import argparse
import math
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw
import networkx as nx
import os

def prettify(elem):
    """Return a pretty-printed XML string for the Element without XML declaration."""
    rough_string = etree.tostring(elem, pretty_print=True, encoding='utf-8').decode('utf-8')
    # Remove XML declaration if present
    if rough_string.startswith('<?xml'):
        rough_string = rough_string[rough_string.find('?>')+2:].lstrip()
    return rough_string

def create_transform_element(parent_elem, tag_name, values=None):
    """Create a transform element with M11-M44 and CenterLocal values."""
    transform = etree.SubElement(parent_elem, tag_name)
    
    # Default matrix values (identity matrix)
    matrix_values = {
        'M11': 1, 'M12': 0, 'M13': 0, 'M14': 0,
        'M21': 0, 'M22': 1, 'M23': 0, 'M24': 0,
        'M31': 0, 'M32': 0, 'M33': 1, 'M34': 0,
        'M41': 0, 'M42': 0, 'M43': 0, 'M44': 1,
        'CenterLocalX': 0, 'CenterLocalY': 0
    }
    
    # Override with provided values
    if values:
        for k, v in values.items():
            if k in matrix_values:
                matrix_values[k] = v
    
    # Create each matrix element
    for key, value in matrix_values.items():
        elem = etree.SubElement(transform, key)
        elem.text = str(value)
    
    return transform

def create_vertex_elements(parent_elem, vertices, shift_x, shift_y, rotation, scale, flip_x):
    """Create vertex elements with X,Y coordinates with transformation applied."""
    # Convert rotation to radians
    rotation_rad = math.radians(rotation)
    
    # First apply all transformations to the vertices
    transformed_vertices = []
    for x, y in vertices:
        # Apply flip if enabled
        if flip_x:
            x = -x
        
        # Apply scale
        scaled_x = x * scale
        scaled_y = y * scale
        
        # Apply rotation
        rotated_x = scaled_x * math.cos(rotation_rad) - scaled_y * math.sin(rotation_rad)
        rotated_y = scaled_x * math.sin(rotation_rad) + scaled_y * math.cos(rotation_rad)
        
        # Apply translation
        final_x = rotated_x + shift_x
        final_y = rotated_y + shift_y
        
        transformed_vertices.append((final_x, final_y))
    
    # Calculate the mean of transformed coordinates if centering is requested
    x_coords = [x for x, y in transformed_vertices]
    y_coords = [y for x, y in transformed_vertices]

    mean_x = sum(x_coords) / len(x_coords)
    mean_y = sum(y_coords) / len(y_coords)

    for x, y in transformed_vertices:
        # Center vertices by subtracting the mean
        centered_x = x - mean_x
        centered_y = y - mean_y

        # Create vertex element
        vertex = etree.SubElement(parent_elem, "Vertex")
        x_elem = etree.SubElement(vertex, "X")
        x_elem.text = f"{centered_x:.8f}"
        y_elem = etree.SubElement(vertex, "Y")
        y_elem.text = f"{centered_y:.8f}"
    return (mean_x,mean_y)
def generate_section_xml(mean_x=0, mean_y=0):
    """Generate a Section XML structure similar to short.xml."""
    # Root element
    section = etree.Element("Section")
    
    # Basic section properties
    created_index = etree.SubElement(section, "CreatedIndex")
    created_index.text = "1"
    
    custom_name = etree.SubElement(section, "CustomName")
    custom_name.text = "false"
    
    section_name = etree.SubElement(section, "Name")
    section_name.text = "Section Set 1 - 1"
    
    # Generate random UID
    section_uid = random.randint(100000000, 999999999)
    uid_elem = etree.SubElement(section, "UID")
    uid_elem.text = str(section_uid)
    
    # Parent transform with explicit translation values that account for mean centering
    # Add the mean coordinates to the base values
    parent_transform_values = {
        'M41': -16451.171875 + mean_x,
        'M42': -3049.04931640625 + mean_y
    }
    create_transform_element(section, "ParentTransform", parent_transform_values)
    
    # Visibility
    is_visible = etree.SubElement(section, "IsVisible")
    is_visible.text = "true"
    
    # Default alignment
    default_alignment = etree.SubElement(section, "DefaultAlignment")
    create_transform_element(default_alignment, "ParentTransform")
    
    # Z bounds
    min_z = etree.SubElement(section, "MinZ")
    min_z.text = "0"
    max_z = etree.SubElement(section, "MaxZ")
    max_z.text = "0"
    
    # Geometry section
    geometry = etree.SubElement(section, "Geometry")
    geom_type = etree.SubElement(geometry, "Type")
    geom_type.text = "Polygon"
    
    # Rotation center
    rotation_centre = etree.SubElement(section, "RotationCentre")
    x_elem = etree.SubElement(rotation_centre, "X")
    x_elem.text = "0"
    y_elem = etree.SubElement(rotation_centre, "Y")
    y_elem.text = "0"
    
    # Link UID
    link_uid = random.randint(100000000, 999999999)
    link_uid_elem = etree.SubElement(section, "LinkUID")
    link_uid_elem.text = str(link_uid)
    
    # Geometry acquisition status
    geom_acquired = etree.SubElement(section, "GeometryAcquired")
    geom_acquired.text = "false"
    
    # Section index
    section_index = etree.SubElement(section, "SectionIndex")
    section_index.text = "0"
    
    # Alignment transform
    alignment_transform = etree.SubElement(section, "AlignmentTransform")
    create_transform_element(alignment_transform, "ParentTransform")
    
    # Aligned status
    aligned = etree.SubElement(section, "Aligned")
    aligned.text = "false"
    
    # Acquisition specification
    acquisition_spec = etree.SubElement(section, "AcquisitionSpec")
    
    spec_name = etree.SubElement(acquisition_spec, "Name")
    spec_name.text = "Acquisition Spec 7"
    
    spec_uid = etree.SubElement(acquisition_spec, "UID")
    spec_uid.text = str(link_uid)
    
    placeable_data_uid = etree.SubElement(acquisition_spec, "PlaceableDataUID")
    placeable_data_uid.text = "-1"
    
    acquisition_type = etree.SubElement(acquisition_spec, "AcquisitionTypeEnum")
    acquisition_type.text = "ForAcquisition"
    
    data_type = etree.SubElement(acquisition_spec, "AcquisitionDataTypeEnum")
    data_type.text = "2DData"
    
    state = etree.SubElement(acquisition_spec, "AcquisitionStateEnum")
    state.text = "Fresh"
    
    mode = etree.SubElement(acquisition_spec, "AcquisitionMode")
    mode.text = "Mosaic"
    
    protocol_uid = 1573444068
    protocol_uid_elem = etree.SubElement(acquisition_spec, "WorkingProtocolUID")
    protocol_uid_elem.text = str(protocol_uid)
    
    return section

def parse_magc_file(file_path):
    """Parse the MAGC file to extract polygon coordinates."""
    sections = []
    current_section = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # New section
            if line.startswith('[section.'):
                if current_section:
                    sections.append(current_section)
                current_section = {'vertices': []}
            
            # Parse polygon line
            elif line.startswith('polygon = '):
                coords_str = line.replace('polygon = ', '')
                # Split by commas and convert to float
                coords = [float(c) for c in coords_str.split(',')]
                # Create (x,y) vertex pairs
                vertices = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                current_section = {'vertices': vertices}
    
    # Add the last section
    if current_section:
        sections.append(current_section)
    
    return sections

def calculate_centroid(vertices):
    """Calculate the centroid of a polygon."""
    x_coords = [x for x, y in vertices]
    y_coords = [y for x, y in vertices]
    centroid_x = sum(x_coords) / len(x_coords)
    centroid_y = sum(y_coords) / len(y_coords)
    return (centroid_x, centroid_y)

def optimize_path(sections_data, method="simulated_annealing", time_limit=10):
    """
    Optimize the order of sections using various TSP algorithms with time limits.
    
    Args:
        sections_data: List of section data with vertices
        method: The optimization method to use:
            - "tsp" - Full traveling salesman problem (slow but optimal)
            - "nearest_neighbor" - Greedy nearest neighbor (fast but suboptimal)
            - "simulated_annealing" - Time-limited simulated annealing (balanced)
            - "two_opt" - 2-opt local search (medium speed, good quality)
        time_limit: Maximum runtime in seconds (for applicable methods)
        
    Returns:
        Reordered list of sections_data
    """
    import time
    import random
    
    if len(sections_data) <= 1:
        return sections_data
    
    # Calculate centroids for each section
    centroids = [calculate_centroid(section['vertices']) for section in sections_data]
    
    # Create distance matrix for quick lookups
    n = len(centroids)
    distances = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                x1, y1 = centroids[i]
                x2, y2 = centroids[j]
                # Calculate Euclidean distance
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distances[(i, j)] = distance
    
    if method == "tsp":
        # Create a complete graph for NetworkX
        G = nx.complete_graph(n)
        
        # Add edge weights based on distance matrix
        for i, j in G.edges():
            G[i][j]['weight'] = distances.get((i, j), 0)
        
        # Solve TSP using NetworkX's approximation algorithm
        tsp_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
        
        return [sections_data[i] for i in tsp_path]
    
    elif method == "nearest_neighbor":
        # Simple greedy nearest neighbor algorithm
        unvisited = set(range(n))
        # Start from node 0
        current = 0
        path = [current]
        unvisited.remove(current)
        
        # Keep adding the nearest unvisited node
        while unvisited:
            nearest = min(unvisited, key=lambda x: distances.get((current, x), float('inf')))
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return [sections_data[i] for i in path]
    
    elif method == "simulated_annealing":
        # Implementation of simulated annealing with time limit
        def path_distance(path):
            """Calculate the total path distance"""
            total = 0
            for i in range(len(path)-1):
                total += distances.get((path[i], path[i+1]), 0)
            return total
        
        # Start with nearest neighbor solution instead of random path
        print("Generating initial nearest neighbor path for simulated annealing...")
        unvisited = set(range(n))
        current = 0
        current_path = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distances.get((current, x), float('inf')))
            current_path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        current_distance = path_distance(current_path)
        best_path = current_path.copy()
        best_distance = current_distance
        
        # Set simulated annealing parameters
        temp = 100.0  # Initial temperature
        cooling_rate = 0.995  # Cooling rate
        
        # Time limit enforcement
        start_time = time.time()
        iterations = 0
        
        print(f"Starting simulated annealing optimization from nearest neighbor path (max {time_limit} seconds)...")
        
        while time.time() - start_time < time_limit:
            iterations += 1
            
            # Create a neighboring solution by swapping two cities
            new_path = current_path.copy()
            i, j = sorted(random.sample(range(n), 2))
            new_path[i:j+1] = reversed(new_path[i:j+1])  # Reverse segment
            
            # Calculate new distance
            new_distance = path_distance(new_path)
            
            # Determine if we should accept the new solution
            if new_distance < current_distance:
                # Always accept better solutions
                current_path = new_path
                current_distance = new_distance
                # Update best solution if needed
                if new_distance < best_distance:
                    best_path = new_path.copy()
                    best_distance = new_distance
            else:
                # Accept worse solutions with a probability based on temperature
                delta = new_distance - current_distance
                probability = math.exp(-delta / temp)
                if random.random() < probability:
                    current_path = new_path
                    current_distance = new_distance
            
            # Cool down the temperature
            temp *= cooling_rate
            
            # Restart if temperature gets too low
            if temp < 0.1:
                temp = 100.0
        
        elapsed = time.time() - start_time
        print(f"Completed {iterations} iterations in {elapsed:.2f} seconds")
        print(f"Best path distance: {best_distance:.2f}")
        
        return [sections_data[i] for i in best_path]
    
    elif method == "two_opt":
        # Implementation of 2-opt local search with time limit
        def path_distance(path):
            """Calculate the total path distance"""
            total = 0
            for i in range(len(path)-1):
                total += distances.get((path[i], path[i+1]), 0)
            return total
        
        # Start with nearest neighbor solution for good initial path
        unvisited = set(range(n))
        current = 0
        path = [current]
        unvisited.remove(current)
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: distances.get((current, x), float('inf')))
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        best_distance = path_distance(path)
        
        # Time limit enforcement
        start_time = time.time()
        iterations = 0
        
        print(f"Starting 2-opt optimization (max {time_limit} seconds)...")
        
        improvement = True
        while improvement and time.time() - start_time < time_limit:
            improvement = False
            
            # Try all possible 2-opt swaps
            for i in range(1, n-2):
                for j in range(i+1, n-1):
                    # Calculate the change in distance if we reverse path[i:j+1]
                    before = distances.get((path[i-1], path[i]), 0) + distances.get((path[j], path[j+1]), 0)
                    after = distances.get((path[i-1], path[j]), 0) + distances.get((path[i], path[j+1]), 0)
                    
                    if after < before:
                        # Reverse the segment from i to j
                        path[i:j+1] = reversed(path[i:j+1])
                        improvement = True
                        best_distance -= (before - after)
                        break
                
                if improvement:
                    break
                    
                # Check time limit periodically
                iterations += 1
                if iterations % 100 == 0 and time.time() - start_time >= time_limit:
                    break
        
        elapsed = time.time() - start_time
        print(f"Completed 2-opt optimization in {elapsed:.2f} seconds")
        print(f"Best path distance: {best_distance:.2f}")
        
        return [sections_data[i] for i in path]
    
    else:
        print(f"Unknown method: {method}, falling back to nearest neighbor")
        return optimize_path(sections_data, "nearest_neighbor", time_limit)

def create_tif_overlay(sections_data, tif_path='5x_gregc_ORG_10p.tif', 
                       output_path=None, line_color=(255, 0, 0), line_width=3):
    """
    Create a TIF overlay showing the regions and their visiting sequence.
    
    Args:
        sections_data: List of section data containing vertices
        tif_path: Path to the background TIF image
        output_path: Path to save the overlay image (default: overlay_{timestamp}.tif)
        line_color: Color of the path lines (default: red)
        line_width: Width of the path lines (default: 3 pixels)
        
    Returns:
        Path to the generated overlay image
    """
    import datetime
    
    if not os.path.exists(tif_path):
        print(f"Warning: Background TIF file {tif_path} not found.")
        img = Image.new('RGB', (5000, 5000), (255, 255, 255))
    else:
        # Open the background image
        img = Image.open(tif_path)
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Calculate centroids for each section
    centroids = [calculate_centroid(section['vertices']) for section in sections_data]
    
    # Draw all polygons
    for i, section in enumerate(sections_data):
        # Convert vertices to a flat list for the polygon function, dividing by 10
        poly_points = []
        for x, y in section['vertices']:
            # Simple division by 10 as requested
            poly_points.extend([x/10, y/10])
        
        # Draw the polygon outline
        draw.polygon(poly_points, outline=(0, 255, 0), width=2)
        
        # Draw the centroid with index, dividing by 10
        cx, cy = centroids[i]
        cx, cy = cx/10, cy/10  # Scale down by 10
        
        radius = 10
        # Draw a circle for the centroid
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), 
                    fill=(0, 0, 255), outline=(255, 255, 255))
        
        # Draw the index number
        font_size = 20
        draw.text((cx+radius+5, cy-font_size//2), str(i+1), fill=(255, 255, 255))
    
    # Draw the path connecting centroids in sequence
    for i in range(len(centroids)-1):
        x1, y1 = centroids[i]
        x2, y2 = centroids[i+1]
        
        # Scale down by 10
        x1, y1 = x1/10, y1/10
        x2, y2 = x2/10, y2/10
        
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=line_width)
        
        # Draw direction arrow (midpoint of the line)
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        # Calculate directional vector and normalize it
        dir_x, dir_y = x2 - x1, y2 - y1
        length = math.sqrt(dir_x**2 + dir_y**2)
        if length > 0:
            dir_x, dir_y = dir_x / length * 20, dir_y / length * 20
            # Draw arrowhead
            draw.polygon([
                (mid_x, mid_y),
                (mid_x - dir_y - dir_x/2, mid_y + dir_x - dir_y/2),
                (mid_x + dir_y - dir_x/2, mid_y - dir_x - dir_y/2)
            ], fill=line_color)
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"path_overlay_{timestamp}.png"  # Use PNG for better compatibility
    
    # Save the overlay image
    img.save(output_path)
    print(f"Generated overlay image: {output_path}")
    
    return output_path

def generate_xml(shift_x, shift_y, rotation, scale, flip_x=False, optimize=True, 
               optimize_method="simulated_annealing", time_limit=10,
               create_overlay=True, overlay_output=None, tif_background='5x_gregc_ORG_10p.tif'):
    """Generate a SectionSet XML structure with sections from MAGC file."""
    import datetime
    
    # Parse the MAGC file to get polygon coordinates
    magc_file_path = "sheff01.magc"
    sections_data = parse_magc_file(magc_file_path)
    
    # Optimize the path if requested
    if optimize:
        sections_data = optimize_path(sections_data, method=optimize_method, time_limit=time_limit)
        print(f"Path optimized using {optimize_method} algorithm for {len(sections_data)} sections")
        
    # Create a TIF overlay if requested
    if create_overlay:
        overlay_path = create_tif_overlay(sections_data, tif_path=tif_background, output_path=overlay_output)
        print(f"TIF overlay created at {overlay_path}")
    
    # Use all sections from the MAGC file
    num_sections = len(sections_data)
    
    # Create section set with data from MAGC file
    section_set = etree.Element("SectionSet")
    
    # Add section set metadata
    name_elem = etree.SubElement(section_set, "Name")
    name_elem.text = "Section Set 1"
    
    uid_elem = etree.SubElement(section_set, "UID")
    uid_elem.text = "2053835327"
    
    # Parent transform
    parent_transform = etree.SubElement(section_set, "ParentTransform")
    transform_values = {
        'M11': 1, 'M12': 0, 'M13': 0, 'M14': 0,
        'M21': 0, 'M22': 1, 'M23': 0, 'M24': 0,
        'M31': 0, 'M32': 0, 'M33': 1, 'M34': 0,
        'M41': 0, 'M42': 0, 'M43': 0, 'M44': 1,
        'CenterLocalX': 0, 'CenterLocalY': 0
    }
    for key, value in transform_values.items():
        elem = etree.SubElement(parent_transform, key)
        elem.text = str(value)
    
    # IsVisible
    is_visible = etree.SubElement(section_set, "IsVisible")
    is_visible.text = "true"
    
    # Default alignment
    default_alignment = etree.SubElement(section_set, "DefaultAlignment")
    default_transform = etree.SubElement(default_alignment, "ParentTransform")
    for key, value in transform_values.items():
        elem = etree.SubElement(default_transform, key)
        elem.text = str(value)
    
    # Created region count
    region_count = etree.SubElement(section_set, "CreatedRegionCount")
    region_count.text = str(num_sections)
    
    # Child name root
    child_name_root = etree.SubElement(section_set, "ChildNameRoot")
    child_name_root.text = "Region "
    
    # Add sections with real polygon data
    for i in range(num_sections):
        # Create a basic section structure first
        section = generate_section_xml()
        
        # Update the section with proper indices
        for child in section:
            if child.tag == "Name":
                child.text = f"Section Set 1 - {i + 1}"
            elif child.tag == "CreatedIndex":
                child.text = str(i + 1)
            elif child.tag == "SectionIndex":
                child.text = str(i)
            elif child.tag == "Geometry":
                # Remove any existing Type element (we'll add it back)
                for type_elem in child.findall("Type"):
                    child.remove(type_elem)
                
                # Add the Type element back
                type_elem = etree.SubElement(child, "Type")
                type_elem.text = "Polygon"
                
                # Remove existing vertices
                for vertex in child.findall("Vertex"):
                    child.remove(vertex)
                
                # Add the vertices from MAGC file with transformations
                mean_x, mean_y = create_vertex_elements(
                    child, 
                    sections_data[i]['vertices'], 
                    shift_x, 
                    shift_y, 
                    rotation, 
                    scale,
                    flip_x
                )
                
                # Now update the ParentTransform with the mean values
                for parent_transform in section.findall("ParentTransform"):
                    for elem in parent_transform:
                        if elem.tag == "M41":
                            elem.text = str(float(elem.text) + mean_x)
                        elif elem.tag == "M42":
                            elem.text = str(float(elem.text) + mean_y)
            elif child.tag == "AcquisitionSpec":
                for spec_child in child:
                    if spec_child.tag == "Name":
                        spec_child.text = f"Acquisition Spec {i + 7}"
                        break
        
        # Add the section to the section set
        section_set.append(section)
    
    # Generate XML string
    xml_string = prettify(section_set)
    
    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"generated_sections_{timestamp}.xml"
    
    # Write to file
    with open(output_file, "w") as f:
        f.write(xml_string)
    
    print(f"Generated XML file with {num_sections} sections using real polygon data: {output_file}")
    return output_file

def concat_files(file_list, output_file):
    """
    Bytewise concatenate files into a single output file.
    
    Args:
        file_list: List of file paths to concatenate
        output_file: Path to the output file
    """
    try:
        with open(output_file, 'wb') as outfile:
            for file_path in file_list:
                with open(file_path, 'rb') as infile:
                    outfile.write(infile.read())
        
        print(f"Successfully created fusion file: {output_file}")
    except Exception as e:
        print(f"Error creating fusion file: {str(e)}")

def main():
    import datetime
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate XML with transformed coordinates and create fusion file")
    parser.add_argument('--shift_x', type=float, default=0, help='X-axis shift value')
    parser.add_argument('--shift_y', type=float, default=0, help='Y-axis shift value')
    parser.add_argument('--rotation', type=float, default=0, help='Rotation in degrees')
    parser.add_argument('--scale', type=float, default=0.001, help='Scale factor for coordinates')
    parser.add_argument('--flip_x', action='store_true', help='Flip coordinates horizontally (left-right)')
    parser.add_argument('--no-optimize', action='store_true', help='Disable path optimization')
    parser.add_argument('--optimize-method', type=str, default='simulated_annealing', 
                      choices=['tsp', 'nearest_neighbor', 'simulated_annealing', 'two_opt'],
                      help='Path optimization method to use')
    parser.add_argument('--time-limit', type=int, default=10, 
                      help='Time limit in seconds for optimization algorithms')
    parser.add_argument('--no-overlay', action='store_true', help='Skip creating TIF overlay')
    parser.add_argument('--overlay-output', type=str, help='Custom path for the TIF overlay output')
    parser.add_argument('--tif-background', type=str, default='5x_gregc_ORG_10p.tif', help='Path to background TIF file')
    
    args = parser.parse_args()
    
    # Generate the XML with the provided transformation parameters
    xml_file = generate_xml(
        args.shift_x, 
        args.shift_y, 
        args.rotation, 
        args.scale, 
        args.flip_x,
        not args.no_optimize,      # Optimize unless --no-optimize is specified
        args.optimize_method,      # Optimization method
        args.time_limit,           # Time limit for optimization
        not args.no_overlay,       # Create overlay unless --no-overlay is specified
        args.overlay_output,       # Custom overlay output path
        args.tif_background        # Background TIF file
    )
    
    # Generate timestamp for fusion filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set input and output file paths for fusion
    part1_file = "greggc_5xopticalmap_0321_part1.a5proj"
    part3_file = "greggc_5xopticalmap_0321_part3.a5proj"
    fusion_file = f"fusion_{timestamp}.a5proj"
    
    # Concatenate the files in order
    concat_files([part1_file, xml_file, part3_file], fusion_file)
    
    print("\nCommand line options:")
    print("  To disable path optimization: --no-optimize")
    print("  To choose optimization method: --optimize-method [tsp|nearest_neighbor|simulated_annealing|two_opt]")
    print("  To set optimization time limit: --time-limit SECONDS")
    print("  To skip creating TIF overlay: --no-overlay")
    print("  To specify background TIF: --tif-background path/to/file.tif")
    print("  To specify overlay output path: --overlay-output path/to/output.tif")

if __name__ == "__main__":
    main()
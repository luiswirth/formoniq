import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.tri import Triangulation
from matplotlib.colors import Normalize
from matplotlib.patches import Polygon

def compute_triangle_area(vertices):
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    return 0.5 * abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

def normalize_edge(edge):
    return tuple(sorted(edge))

def generate_triangle_grid(vertices, n_points):
    points = []
    for i in range(n_points+1):
        for j in range(n_points+1-i):
            lambda0 = (n_points - i - j) / n_points
            lambda1 = j / n_points
            lambda2 = i / n_points
            
            x = lambda0 * vertices[0, 0] + lambda1 * vertices[1, 0] + lambda2 * vertices[2, 0]
            y = lambda0 * vertices[0, 1] + lambda1 * vertices[1, 1] + lambda2 * vertices[2, 1]
            points.append([x, y])
    
    return np.array(points)

def compute_barycentric_gradients(vertices):
    area = compute_triangle_area(vertices)
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
        
    grad_lambda0 = np.array([(y1 - y2) / (2 * area), (x2 - x1) / (2 * area)])
    grad_lambda1 = np.array([(y2 - y0) / (2 * area), (x0 - x2) / (2 * area)])
    grad_lambda2 = np.array([(y0 - y1) / (2 * area), (x1 - x0) / (2 * area)])
    
    return np.array([grad_lambda0, grad_lambda1, grad_lambda2])

def compute_barycentric_coordinates(point, vertices):
    """Compute barycentric coordinates for a point in a triangle."""
    x, y = point
    
    area = compute_triangle_area(vertices)
    x0, y0 = vertices[0]
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    
    # Compute sub-triangle areas
    area0 = 0.5 * ((x1 - x) * (y2 - y) - (x2 - x) * (y1 - y))
    area1 = 0.5 * ((x2 - x) * (y0 - y) - (x0 - x) * (y2 - y))
    area2 = 0.5 * ((x0 - x) * (y1 - y) - (x1 - x) * (y0 - y))
    
    lambda0 = area0 / area
    lambda1 = area1 / area
    lambda2 = area2 / area
    
    return np.array([lambda0, lambda1, lambda2])

def whitney_form(point, vertices, i, j, gradients=None):
    """Whitney 1-form λᵢⱼ = λᵢ∇λⱼ - λⱼ∇λᵢ"""
    if gradients is None:
        gradients = compute_barycentric_gradients(vertices)
    grad_i = gradients[i]
    grad_j = gradients[j]
    
    lambdas = compute_barycentric_coordinates(point, vertices)
    
    whitney_vector = lambdas[i] * grad_j - lambdas[j] * grad_i
    return whitney_vector

def evaluate_cochain(point, vertices, triangle, cochain, gradients=None):
    """Evaluate a cochain at a point inside a triangle."""
    tri_vertices = vertices[triangle]
    
    if gradients is None:
        gradients = compute_barycentric_gradients(tri_vertices)
    
    local_edges = [(0, 1), (0, 2), (1, 2)]
    
    result = np.zeros(2)
    for local_i, local_j in local_edges:
        global_i = triangle[local_i]
        global_j = triangle[local_j]
        
        original_edge = (global_i, global_j)
        global_edge = normalize_edge(original_edge)
        edge_sign = 1 if original_edge == global_edge else -1
        
        if global_edge in cochain:
            coeff = cochain[global_edge]
            whitney_value = whitney_form(point, tri_vertices, local_i, local_j, gradients)
            result += edge_sign * coeff * whitney_value
    
    return result

def setup_plot():
    """Create a new figure with common styling."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    return fig, ax

def finalize_plot(fig, ax, vertices, filename, fontsize=20):
    """Apply final styling to the plot and save it."""
    
    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)
    margin = 0.02 * max(max_x - min_x, max_y - min_y)
    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(min_y - margin, max_y + margin)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=1000, bbox_inches='tight')
    print(f"Figure saved as '{filename}'")
    plt.close(fig)

def compute_magnitude(vectors):
    return np.sqrt(vectors[:, 0]**2 + vectors[:, 1]**2)

def normalize_vectors(vectors):
    magnitude = compute_magnitude(vectors)
    with np.errstate(divide='ignore', invalid='ignore'):
        u_norm = np.where(magnitude > 0, vectors[:, 0] / magnitude, 0)
        v_norm = np.where(magnitude > 0, vectors[:, 1] / magnitude, 0)
    return np.column_stack((u_norm, v_norm))

def plot_vector_field(ax, points, vectors, triangle_vertices, triangle_area, length=0.05, width=0.004):
    normalized = normalize_vectors(vectors)
    u, v = normalized[:, 0], normalized[:, 1]
    
    area_ref = 0.5
    area_factor = np.sqrt(triangle_area / area_ref)
    length = length * area_factor
    width = width * area_factor
    
    triangle_polygon = Polygon(triangle_vertices, closed=True, fill=False, visible=False)
    ax.add_patch(triangle_polygon)
    
    ax.quiver(
        points[:, 0], points[:, 1],
        u, v,
        angles='xy', scale_units='xy', units='xy',
        pivot='tail', color='white', alpha=1.0,
        scale=1/length, width=width,
        headwidth=3.5, headlength=3.5, headaxislength=3.0,
        clip_path=triangle_polygon
    )

def plot_mesh_edges(ax, vertices, triangles, cochain_highlight=None):
    linewidth = 2.0

    for triangle in triangles:
        tri_vertices = vertices[triangle]
        tri_vertices = np.vstack([tri_vertices, tri_vertices[0]])
        ax.plot(tri_vertices[:, 0], tri_vertices[:, 1], 'k-', linewidth=linewidth)
    
    if cochain_highlight:
        for edge in cochain_highlight:
            coeff = cochain_highlight[edge]
            if coeff != 0:
                edge = normalize_edge(edge)
                v0, v1 = edge
                edge_vertices = vertices[[v0, v1]]
                ax.plot(edge_vertices[:, 0], edge_vertices[:, 1], 'r-', linewidth=linewidth)

def has_nonzero_dofs(triangle, cochain):
    local_edges = [(0, 1), (0, 2), (1, 2)]
    for local_i, local_j in local_edges:
        global_i = triangle[local_i]
        global_j = triangle[local_j]
        edge = normalize_edge((global_i, global_j))
        if edge in cochain and abs(cochain[edge]) > 1e-10:
            return True
    return False

def compute_vector_field_data(vertices, triangles, cochain, skip_zero_triangles=False, heatmap_resolution=30):
    all_areas = [compute_triangle_area(vertices[triangle]) for triangle in triangles]
    triangle_data = []
    all_magnitudes = []
    
    for t_idx, triangle in enumerate(triangles):
        # Skip triangles with all zero DOFs if requested
        if skip_zero_triangles and not has_nonzero_dofs(triangle, cochain):
            continue
            
        tri_vertices = vertices[triangle]
        area = all_areas[t_idx]
        
        tri_points = generate_triangle_grid(tri_vertices, heatmap_resolution)
        x, y = tri_points[:, 0], tri_points[:, 1]
        triang = Triangulation(x, y)
        
        gradients = compute_barycentric_gradients(tri_vertices)
        
        vectors = np.array([
            evaluate_cochain([xi, yi], vertices, triangle, cochain, gradients) 
            for xi, yi in zip(x, y)
        ])
        magnitude = compute_magnitude(vectors)
        all_magnitudes.extend(magnitude)
        
        triangle_data.append({
            'triangle': triangle,
            'vertices': tri_vertices,
            'points': tri_points,
            'vectors': vectors,
            'magnitude': magnitude,
            'gradients': gradients,
            'triangulation': triang,
            'area': area
        })
    
    return triangle_data, all_magnitudes

def get_magnitude_range(all_magnitudes):
    min_magnitude = min(all_magnitudes)
    max_magnitude = max(all_magnitudes)
    magnitude_range = max_magnitude - min_magnitude

    if magnitude_range < 1e-6:
        mean_magnitude = (max_magnitude + min_magnitude) / 2
        min_magnitude = mean_magnitude * 0.95
        max_magnitude = mean_magnitude * 1.05
        
        if abs(mean_magnitude) < 1e-10:
            min_magnitude = 0
            max_magnitude = 1e-6
    
    return min_magnitude, max_magnitude

def plot_cochain(vertices, triangles, cochain, filename, highlight_edges=False, 
               skip_zero_triangles=False, heatmap_resolution=30, quiver_count=20):
    fig, ax = setup_plot()
    
    # Compute vector field data for all triangles
    triangle_data, all_magnitudes = compute_vector_field_data(
        vertices, triangles, cochain, skip_zero_triangles, heatmap_resolution
    )
    
    if not all_magnitudes:
        print(f"Warning: No non-zero triangles found for {filename}")
        all_magnitudes = [0.0]
    
    # Get magnitude range for consistent coloring
    min_magnitude, max_magnitude = get_magnitude_range(all_magnitudes)
    norm = Normalize(vmin=min_magnitude, vmax=max_magnitude)
    
    # Plot the vector field
    for t_data in triangle_data:
        triangle = t_data['triangle']
        area = t_data['area']
        
        ax.tripcolor(
            t_data['triangulation'], t_data['magnitude'], 
            cmap='viridis', shading='flat', norm=norm
        )
        
        quiver_points = generate_triangle_grid(t_data['vertices'], quiver_count)
        quiver_vectors = np.array([
            evaluate_cochain([p[0], p[1]], vertices, triangle, cochain, t_data['gradients']) 
            for p in quiver_points
        ])
        plot_vector_field(
            ax, quiver_points, quiver_vectors, 
            triangle_vertices=t_data['vertices'], 
            triangle_area=area, 
            length=0.8 / quiver_count, 
            width=0.1 / quiver_count
        )

    #plot_mesh_edges(ax, vertices, triangles, cochain if highlight_edges else None)
    finalize_plot(fig, ax, vertices, filename)

def load_file(filename, dtype=float):
    with open(filename, 'r') as f:
        return [list(map(dtype, line.strip().split())) for line in f if line.strip()]

def load_cochain(filename):
    with open(filename, 'r') as f:
        return np.array([float(line.strip()) for line in f if line.strip()])

def load_mesh(input_path):
    vertices = np.array(load_file(f'{input_path}/vertices.coords'), dtype=float)
    triangles = np.array(load_file(f'{input_path}/cells.skel', dtype=int))
    edges = np.array(load_file(f'{input_path}/edges.skel', dtype=int))
    return vertices, triangles, edges

def build_cochain_map(edges_array, cochain_values):
    cochain_map = {}
    for i, (v0, v1) in enumerate(edges_array):
        edge = normalize_edge((v0, v1))
        cochain_map[edge] = cochain_values[i]
    return cochain_map

def plot_from_files(input_path, skip_zero_triangles=False, heatmap_resolution=30, 
                  quiver_count=20, highlight_edges=True):
    input_path = input_path.rstrip('/')
    folder_name = os.path.basename(input_path)

    vertices, triangles, edges_array = load_mesh(input_path)
    
    for cochain_path in glob.glob(f'{input_path}/*.cochain'):
        cochain_name = os.path.basename(cochain_path)
        cochain_values = load_cochain(cochain_path)
        cochain_map = build_cochain_map(edges_array, cochain_values)

        output_file_path = f'out/{folder_name}_{cochain_name}.png'
        plot_cochain(
            vertices, 
            triangles, 
            cochain_map, 
            output_file_path, 
            highlight_edges=highlight_edges,
            skip_zero_triangles=skip_zero_triangles,
            heatmap_resolution=heatmap_resolution,
            quiver_count=quiver_count
        )

def main():
    os.makedirs("out", exist_ok=True)
    
    import argparse
    parser = argparse.ArgumentParser(description='Whitney Forms Visualizer')
    parser.add_argument('path', help='Path to the input files')
    parser.add_argument('--skip-zero', action='store_true', 
                        help='Skip triangles with all zero DOFs')
    parser.add_argument('--heatmap-res', type=int, default=30,
                        help='Resolution of the heatmap (default: 30)')
    parser.add_argument('--quiver-count', type=int, default=20,
                        help='Number of quiver arrows per triangle dimension (default: 20)')
    parser.add_argument('--highlight', action='store_true',
                        help='Disable highlighting of non-zero DOF edges')
    
    args = parser.parse_args()
    plot_from_files(
        args.path, 
        skip_zero_triangles=args.skip_zero,
        heatmap_resolution=args.heatmap_res,
        quiver_count=args.quiver_count,
        highlight_edges=args.highlight
    )

if __name__ == "__main__":
    main()

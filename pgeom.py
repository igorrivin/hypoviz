import torch
from functools import partial
import numpy as np
import pyvista as pv
from torch.func import vmap

def compute_frame(f, t):
    """Compute the frame at a single point t"""
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t, dtype=torch.float32)
    # Ensure t is a scalar
    t = t.reshape(-1)[-1]
    t.requires_grad_(True)
    
    # Compute position
    pos = f(t)
    
    # Compute tangent vector (first derivative)
    grads = []
    for i in range(pos.shape[0]):
        grad, = torch.autograd.grad(pos[i], t, create_graph=True)
        grads.append(grad.item())  # Convert to scalar
    tangent = torch.tensor(grads, dtype=torch.float32)
    tangent_norm = torch.linalg.norm(tangent)
    tangent = tangent / tangent_norm
    
    # For second derivative, we need to take gradient of a scalar function
    # We'll use the dot product with basis vectors
    unit_vectors = torch.eye(3, dtype=torch.float32)
    second_derivs = []
    
    # Recompute position and first derivatives for second derivative calculation
    t.requires_grad_(True)
    pos = f(t)
    grads = []
    for i in range(pos.shape[0]):
        grad, = torch.autograd.grad(pos[i], t, create_graph=True)
        grads.append(grad)
    tangent_with_grad = torch.stack(grads)
    tangent_with_grad = tangent_with_grad / torch.linalg.norm(tangent_with_grad)
    
    for basis in unit_vectors:
        # Project tangent onto basis vector to get scalar
        scalar_func = torch.sum(tangent_with_grad * basis)  # Use elementwise multiply and sum instead of dot
        grad, = torch.autograd.grad(scalar_func, t, create_graph=True)
        second_derivs.append(grad.item())  # Convert to scalar
    
    second = torch.tensor(second_derivs, dtype=torch.float32)
    
    # Compute normal and binormal
    normal = second - torch.dot(second, tangent) * tangent
    normal_norm = torch.linalg.norm(normal)
    if normal_norm > 1e-10:  # Check for zero norm
        normal = normal / normal_norm
        binormal = torch.linalg.cross(tangent, normal)
    else:
        # If normal is zero, pick an arbitrary perpendicular vector
        normal = get_perpendicular(tangent)
        binormal = torch.linalg.cross(tangent, normal)
    
    return torch.stack([tangent, normal, binormal])

def get_perpendicular(v):
    """Get a unit vector perpendicular to v"""
    # Find smallest component of v
    min_idx = torch.argmin(torch.abs(v))
    
    # Create perpendicular vector by swapping components and negating one
    perp = torch.zeros_like(v)
    if min_idx == 0:
        perp[1] = -v[2]
        perp[2] = v[1]
    elif min_idx == 1:
        perp[0] = -v[2]
        perp[2] = v[0]
    else:
        perp[0] = -v[1]
        perp[1] = v[0]
    
    return perp / torch.linalg.norm(perp)

def get_point(frame, s):
    v1 = frame[1, :]
    v2 = frame[2, :]
    return torch.cos(s) * v1 + torch.sin(s) * v2

def get_reg_grid(f, num1, num2, eps):
    tarray = torch.linspace(0.0, 1.0, num1)
    sarray = torch.linspace(0.0, 2 * torch.pi, num2)
    
    # Create meshgrid
    T, S = torch.meshgrid(tarray, sarray, indexing='ij')
    points = []
    
    for t, s_row in zip(T, S):
        for s in s_row:
            frame = compute_frame(f, t)
            base = f(t)
            point = base + eps * get_point(frame, s)
            points.append(point)
    
    return torch.stack(points)

def get_irreg_grid(f, num, eps):
    parray = torch.rand(num, 2)
    parray[:, 1] *= 2 * torch.pi  # Scale second dimension to [0, 2π]
    
    points = []
    for p in parray:
        t, s = p[0], p[1]
        frame = compute_frame(f, t)
        base = f(t)
        point = base + eps * get_point(frame, s)
        points.append(point)
    
    return torch.stack(points)

def get_irreg_grid_full(f, num, eps):
    parray = torch.rand(num, 3)
    parray[:, 1] *= 2 * torch.pi  # Scale second dimension to [0, 2π]
    
    points = []
    for p in parray:
        t, s, r = p[0], p[1], torch.sqrt(p[2])
        frame = compute_frame(f, t)
        base = f(t)
        point = base + eps * r * get_point(frame, s)
        points.append(point)
    
    return torch.stack(points)

def make_rand_knot(decay, l):
    # Generate random coefficients
    thebatch1x = torch.randn(l)
    thebatch2x = torch.randn(l)
    thebatch1y = torch.randn(l)
    thebatch2y = torch.randn(l)
    thebatch1z = torch.randn(l)
    thebatch2z = torch.randn(l)
    
    # Calculate scaling factor
    scaler = torch.arange(1, l+1, dtype=torch.float32).pow(-decay)
    
    # Scale coefficients
    coefs1x = thebatch1x * scaler
    coefs2x = thebatch2x * scaler
    coefs1y = thebatch1y * scaler
    coefs2y = thebatch2y * scaler
    coefs1z = thebatch1z * scaler
    coefs2z = thebatch2z * scaler
    
    def make_ser(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # Ensure x is a scalar if it isn't already
        x = x.reshape(-1)[-1] if x.dim() > 0 else x
        
        c = torch.arange(1, l+1, dtype=torch.float32)
        cc = c * (2 * torch.pi * x)  # Changed multiplication order and added parentheses
        coses = torch.cos(cc)
        sines = torch.sin(cc)
        
        return torch.stack([
            torch.sum(coefs1x * coses) + torch.sum(coefs2x * sines),
            torch.sum(coefs1y * coses) + torch.sum(coefs2y * sines),
            torch.sum(coefs1z * coses) + torch.sum(coefs2z * sines)
        ])
    
    return make_ser

def get_knot(func, howmany=1000):
    args = torch.linspace(0, 1, howmany)
    return torch.stack([func(x) for x in args])

def grid_to_trimesh(vertices, num1, num2):
    """
    Convert a regular grid of vertices to a trimesh.
    
    Args:
        vertices: torch.Tensor of shape (N, 3) containing the vertex positions
        num1: number of vertices in the first dimension
        num2: number of vertices in the second dimension
    
    Returns:
        trimesh.Trimesh object
    """
    import trimesh
    
    # Convert vertices to numpy if they're torch tensors
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    
    # Create faces by connecting vertices in a grid pattern
    faces = []
    for i in range(num1 - 1):
        for j in range(num2 - 1):
            # Get vertex indices for current quad
            v0 = i * num2 + j
            v1 = v0 + 1
            v2 = (i + 1) * num2 + j
            v3 = v2 + 1
            
            # Create two triangles for the quad
            faces.append([v0, v2, v1])  # First triangle
            faces.append([v1, v2, v3])  # Second triangle
    
    faces = np.array(faces)
    
    # Create the mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    return mesh

def get_reg_grid_mesh(f, num1, num2, eps):
    """
    Generate a regular grid and convert it directly to a trimesh.
    
    Args:
        f: function to generate points
        num1: number of points in first dimension
        num2: number of points in second dimension
        eps: epsilon parameter for grid generation
    
    Returns:
        trimesh.Trimesh object
    """
    vertices = get_reg_grid(f, num1, num2, eps)
    return grid_to_trimesh(vertices, num1, num2)
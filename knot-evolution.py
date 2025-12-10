import torch
import torch.optim as optim

class KnotEvolver:
    def __init__(self, initial_coeffs, decay, spring_constant=1.0, repulsion_strength=0.1):
        """
        Initialize knot evolution.
        
        Args:
            initial_coeffs: tuple of (coefs1x, coefs2x, coefs1y, coefs2y, coefs1z, coefs2z)
            decay: decay parameter for the Fourier modes
            spring_constant: strength of the spring potential
            repulsion_strength: strength of the repulsion between points
        """
        # Create trainable parameters from initial coefficients
        self.coefs1x = torch.nn.Parameter(initial_coeffs[0].clone())
        self.coefs2x = torch.nn.Parameter(initial_coeffs[1].clone())
        self.coefs1y = torch.nn.Parameter(initial_coeffs[2].clone())
        self.coefs2y = torch.nn.Parameter(initial_coeffs[3].clone())
        self.coefs1z = torch.nn.Parameter(initial_coeffs[4].clone())
        self.coefs2z = torch.nn.Parameter(initial_coeffs[5].clone())
        
        self.l = len(initial_coeffs[0])
        self.decay = decay
        self.spring_constant = spring_constant
        self.repulsion_strength = repulsion_strength
        
    def get_points(self, num_points=100):
        """Evaluate the knot at specified number of points"""
        t = torch.linspace(0, 1, num_points)
        c = torch.arange(1, self.l + 1, dtype=torch.float32)
        cc = c.view(-1, 1) * (2 * torch.pi * t.view(1, -1))
        
        coses = torch.cos(cc)
        sines = torch.sin(cc)
        
        x = torch.matmul(self.coefs1x, coses) + torch.matmul(self.coefs2x, sines)
        y = torch.matmul(self.coefs1y, coses) + torch.matmul(self.coefs2y, sines)
        z = torch.matmul(self.coefs1z, coses) + torch.matmul(self.coefs2z, sines)
        
        return torch.stack([x, y, z], dim=0).t()
    
    def spring_potential(self, points):
        """Compute spring potential between adjacent points"""
        # Get differences for adjacent points
        diffs = points[1:] - points[:-1]
        # Add the difference between last and first point
        closing_diff = points[0] - points[-1]
        # Concatenate all differences
        diffs = torch.cat([diffs, closing_diff.unsqueeze(0)], dim=0)
        distances = torch.norm(diffs, dim=1)
        target_distance = distances.mean()  # or could be a fixed parameter
        return self.spring_constant * torch.mean((distances - target_distance)**2)
    
    def repulsion_potential(self, points):
        """Compute repulsion potential between all pairs of points"""
        n = len(points)
        # Compute pairwise distances efficiently using broadcasting
        diffs = points.unsqueeze(0) - points.unsqueeze(1)  # n x n x 3
        distances_squared = torch.sum(diffs * diffs, dim=2)  # n x n
        
        # Create mask for self-interactions
        mask = ~torch.eye(n, dtype=torch.bool, device=points.device)
        
        # Compute repulsion (inverse square law)
        # Add small epsilon to avoid division by zero
        repulsion = mask / (distances_squared + 1e-6)
        
        return self.repulsion_strength * torch.sum(repulsion) / (n * (n-1))
    
    def total_potential(self):
        """Compute total potential energy"""
        points = self.get_points()
        return self.spring_potential(points) + self.repulsion_potential(points)
    
    def evolve(self, num_steps=1000, learning_rate=0.01):
        """Evolve the knot to minimize potential"""
        optimizer = optim.Adam([self.coefs1x, self.coefs2x, 
                              self.coefs1y, self.coefs2y,
                              self.coefs1z, self.coefs2z], lr=learning_rate)
        
        trajectory = []
        energies = []
        
        for step in range(num_steps):
            optimizer.zero_grad()
            potential = self.total_potential()
            potential.backward()
            optimizer.step()
            
            if step % 100 == 0:
                trajectory.append(self.get_points().detach())
                energies.append(potential.item())
                print(f"Step {step}, Potential: {potential.item():.6f}")
        
        return trajectory, energies

# Example usage:
def visualize_evolution(trajectory, interval=50):
    """
    Animate the evolution of the knot.
    
    Args:
        trajectory: list of point tensors from KnotEvolver
        interval: time between frames in milliseconds
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert trajectory tensors to numpy arrays
    trajectory_np = [points.numpy() for points in trajectory]
    
    # Get bounds for consistent axes
    all_points = np.concatenate(trajectory_np)
    min_vals = np.min(all_points, axis=0)
    max_vals = np.max(all_points, axis=0)
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Initialize line
    line, = ax.plot([], [], [], 'b-', lw=2)
    
    # Set consistent axis limits
    margin = 0.1 * (max_vals - min_vals)
    ax.set_xlim(min_vals[0] - margin[0], max_vals[0] + margin[0])
    ax.set_ylim(min_vals[1] - margin[1], max_vals[1] + margin[1])
    ax.set_zlim(min_vals[2] - margin[2], max_vals[2] + margin[2])
    
    # Add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Animation update function
    def update(frame):
        points = trajectory_np[frame]
        # Include the first point again at the end to close the loop
        points_closed = np.vstack([points, points[0]])
        line.set_data(points_closed[:, 0], points_closed[:, 1])
        line.set_3d_properties(points_closed[:, 2])
        ax.view_init(30, frame % 360)  # Rotate view
        ax.set_title(f'Step {frame * 100}')
        return line,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                        interval=interval, blit=True)
    
    plt.show()
    return anim  # Return animation object to prevent garbage collection

def plot_energy_evolution(energies):
    """
    Plot the evolution of the potential energy.
    
    Args:
        energies: list of energy values from KnotEvolver
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(energies) * 100, 100), energies, 'b-')
    plt.xlabel('Step')
    plt.ylabel('Potential Energy')
    plt.title('Evolution of Potential Energy')
    plt.grid(True)
    plt.show()
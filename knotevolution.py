import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_noise_schedule(schedule_type='exponential', initial_noise=0.01, **kwargs):
    """
    Create a noise schedule function.
    
    Args:
        schedule_type: one of 'exponential', 'linear', 'inverse', 'cosine', 'step'
        initial_noise: initial noise level
        **kwargs: schedule-specific parameters
            - exponential: tau (decay time constant)
            - linear: T (total time)
            - inverse: k (decay rate)
            - cosine: T (total time)
            - step: step_size, gamma (reduction factor)
    
    Returns:
        Function that takes step number and returns noise level
    """
    if schedule_type == 'exponential':
        tau = kwargs.get('tau', 1000)
        return lambda t: initial_noise * torch.exp(torch.tensor(-t/tau))
    
    elif schedule_type == 'linear':
        T = kwargs.get('T', 10000)
        return lambda t: initial_noise * max(0, 1 - t/T)
    
    elif schedule_type == 'inverse':
        k = kwargs.get('k', 0.001)
        return lambda t: initial_noise / (1 + k*t)
    
    elif schedule_type == 'cosine':
        T = kwargs.get('T', 20000)
        return lambda t: initial_noise * (1 + torch.cos(torch.tensor(torch.pi * min(t, T)/T)))/2
    
    elif schedule_type == 'step':
        step_size = kwargs.get('step_size', 1000)
        gamma = kwargs.get('gamma', 0.5)
        return lambda t: initial_noise * (gamma ** (t // step_size))
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

class KnotEvolver:
    def __init__(self, initial_coeffs, decay, number_of_points=100, spring_constant=1.0, repulsion_strength=0.1):
        """
        Initialize knot evolution with normalized coefficients.
        
        Args:
            initial_coeffs: tuple of (coefs1x, coefs2x, coefs1y, coefs2y, coefs1z, coefs2z)
            decay: decay parameter for the Fourier modes
            number_of_points: number of points to sample along the knot
            spring_constant: strength of the spring potential
            repulsion_strength: strength of the repulsion between points
        """
        # Store number of points
        self.number_of_points = number_of_points
        
        # First, create initial points to compute normalization
        l = len(initial_coeffs[0])
        coefs1x = torch.tensor(initial_coeffs[0], requires_grad=False)
        coefs2x = torch.tensor(initial_coeffs[1], requires_grad=False)
        coefs1y = torch.tensor(initial_coeffs[2], requires_grad=False)
        coefs2y = torch.tensor(initial_coeffs[3], requires_grad=False)
        coefs1z = torch.tensor(initial_coeffs[4], requires_grad=False)
        coefs2z = torch.tensor(initial_coeffs[5], requires_grad=False)
        
        # Compute initial points using specified number of points
        t = torch.linspace(0, 1, self.number_of_points)
        c = torch.arange(1, l + 1, dtype=torch.float32)
        cc = c.view(-1, 1) * (2 * torch.pi * t.view(1, -1))
        coses = torch.cos(cc)
        sines = torch.sin(cc)
        
        x = torch.matmul(coefs1x, coses) + torch.matmul(coefs2x, sines)
        y = torch.matmul(coefs1y, coses) + torch.matmul(coefs2y, sines)
        z = torch.matmul(coefs1z, coses) + torch.matmul(coefs2z, sines)
        points = torch.stack([x, y, z], dim=0).t()
        
        # Compute normalization factor from average segment length
        diffs = points[1:] - points[:-1]
        avg_distance = torch.norm(diffs, dim=1).mean()
        scale_factor = 1.0 / avg_distance
        
        # Create normalized trainable parameters
        self.coefs1x = torch.nn.Parameter(coefs1x * scale_factor)
        self.coefs2x = torch.nn.Parameter(coefs2x * scale_factor)
        self.coefs1y = torch.nn.Parameter(coefs1y * scale_factor)
        self.coefs2y = torch.nn.Parameter(coefs2y * scale_factor)
        self.coefs1z = torch.nn.Parameter(coefs1z * scale_factor)
        self.coefs2z = torch.nn.Parameter(coefs2z * scale_factor)
        
        self.l = l
        self.decay = decay
        self.spring_constant = spring_constant
        self.repulsion_strength = repulsion_strength
    
    def get_points(self):
        """Evaluate the knot at specified number of points"""
        t = torch.linspace(0, 1, self.number_of_points)
        c = torch.arange(1, self.l + 1, dtype=torch.float32)
        cc = c.view(-1, 1) * (2 * torch.pi * t.view(1, -1))
        
        coses = torch.cos(cc)
        sines = torch.sin(cc)
        
        x = torch.matmul(self.coefs1x, coses) + torch.matmul(self.coefs2x, sines)
        y = torch.matmul(self.coefs1y, coses) + torch.matmul(self.coefs2y, sines)
        z = torch.matmul(self.coefs1z, coses) + torch.matmul(self.coefs2z, sines)
        
        return torch.stack([x, y, z], dim=0).t()
    
    def spring_potential(self, points):
        """Compute spring potential between adjacent points with unit target length"""
        # Get differences for adjacent points
        diffs = points[1:] - points[:-1]
        # Add the difference between last and first point
        closing_diff = points[0] - points[-1]
        # Concatenate all differences
        diffs = torch.cat([diffs, closing_diff.unsqueeze(0)], dim=0)
        distances = torch.norm(diffs, dim=1)
        # Use fixed unit length as target
        return self.spring_constant * torch.mean((distances - 1.0)**2)
    
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
    
    def evolve(self, max_steps=10000, learning_rate=0.01, save_every=100, 
               convergence_tol=1e-6, patience=10, initial_noise=0.01,
               noise_schedule='exponential', schedule_params=None):
        """
        Evolve the knot with annealed noise to avoid local minima.
        
        Args:
            max_steps: maximum number of steps
            learning_rate: learning rate for optimizer
            save_every: how often to save trajectory points
            convergence_tol: tolerance for energy change
            patience: number of checks for early stopping
            initial_noise: initial noise magnitude
            noise_schedule: type of noise schedule
            schedule_params: parameters for the noise schedule
        """
        optimizer = optim.Adam([self.coefs1x, self.coefs2x, 
                              self.coefs1y, self.coefs2y,
                              self.coefs1z, self.coefs2z], lr=learning_rate)
        
        schedule_params = schedule_params or {}
        noise_func = get_noise_schedule(noise_schedule, initial_noise, **schedule_params)
        
        trajectory = []
        energies = []
        best_energy = float('inf')
        patience_counter = 0
        
        step = 0
        while step < max_steps:
            optimizer.zero_grad()
            potential = self.total_potential()
            potential.backward()
            
            # Get current noise level from schedule
            current_noise = noise_func(step)
            
            # Add noise to gradients
            with torch.no_grad():
                for param in [self.coefs1x, self.coefs2x, self.coefs1y, 
                            self.coefs2y, self.coefs1z, self.coefs2z]:
                    if param.grad is not None:
                        param.grad += current_noise * torch.randn_like(param.grad)
            
            optimizer.step()
            
            if step % save_every == 0:
                current_energy = potential.item()
                energies.append(current_energy)
                trajectory.append(self.get_points().detach())
                
                # Check for convergence
                if len(energies) > 1:
                    energy_change = abs(energies[-1] - energies[-2])
                    if energy_change < convergence_tol:
                        patience_counter += 1
                    else:
                        patience_counter = 0
                    
                    if patience_counter >= patience:
                        print(f"Converged at step {step} with energy {current_energy:.6f}")
                        break
                
                print(f"Step {step}, Potential: {current_energy:.6f}, Noise: {current_noise:.6f}")
            
            step += 1
        
        if step >= max_steps:
            print(f"Reached maximum steps ({max_steps}) without convergence")
            
        return trajectory, energies

def evolve_knot(knot_func, max_steps=10000, spring_constant=1.0, repulsion_strength=0.1,
               learning_rate=0.01, save_every=100, convergence_tol=1e-6, patience=10,
               initial_noise=0.01, noise_schedule='exponential', schedule_params=None,
               number_of_points=100):
    """
    Evolve a knot given by make_rand_knot function
    
    Args:
        knot_func: function created by make_rand_knot
        max_steps: maximum number of optimization steps
        spring_constant: strength of the spring potential
        repulsion_strength: strength of the repulsion between points
        learning_rate: learning rate for optimization
        save_every: how often to save trajectory points
        convergence_tol: tolerance for energy change to determine convergence
        patience: number of checks without improvement before stopping
        initial_noise: initial magnitude of noise
        noise_schedule: type of cooling schedule
        schedule_params: parameters for cooling schedule
        number_of_points: number of points to sample along the knot
    
    Returns:
        KnotEvolver object after evolution, trajectory, and energies
    """
    # Extract coefficients from the knot function closure
    l = len(knot_func.__closure__[0].cell_contents)  # coefs1x length
    initial_coeffs = tuple(var.cell_contents for var in knot_func.__closure__[:6])
    
    # Create and evolve the knot
    evolver = KnotEvolver(initial_coeffs, 
                         decay=knot_func.__closure__[6].cell_contents,
                         spring_constant=spring_constant,
                         repulsion_strength=repulsion_strength,
                         number_of_points=number_of_points)
    
    trajectory, energies = evolver.evolve(max_steps=max_steps, 
                                        learning_rate=learning_rate,
                                        save_every=save_every,
                                        convergence_tol=convergence_tol,
                                        patience=patience,
                                        initial_noise=initial_noise,
                                        noise_schedule=noise_schedule,
                                        schedule_params=schedule_params)
    
    return evolver, trajectory, energies

def visualize_evolution(trajectory,save_int = 100, interval=50):
    """
    Animate the evolution of the knot.
    
    Args:
        trajectory: list of point tensors from KnotEvolver
        interval: time between frames in milliseconds
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML
    
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
        ax.set_title(f'Step {frame * save_int}')
        return line,
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(trajectory),
                        interval=interval, blit=True)
    
    # Return HTML object for Jupyter display
    return HTML(anim.to_jshtml())

def plot_energy_evolution(energies):
    """Plot the evolution of the potential energy"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(energies) * 100, 100), energies, 'b-')
    plt.xlabel('Step')
    plt.ylabel('Potential Energy')
    plt.title('Evolution of Potential Energy')
    plt.grid(True)
    plt.show()
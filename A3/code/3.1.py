import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from skimage.transform import resize


# ============================================================
# 1. DATA ACQUISITION AND INITIAL PROCESSING
# ============================================================

def acquire_and_scale_picture(file_path, dimensions=(128, 128)):
    """
    Load a picture and optionally scale it.
    
    Parameters:
        file_path (str): Path to picture file
        dimensions (tuple): Desired dimensions (H, W)
        
    Returns:
        picture (float array): Picture normalized to [0,1]
    """
    picture = io.imread(file_path)
    
    # Transform to floating point in [0,1]
    picture = picture / 255.0
    
    # Scale for improved processing speed
    picture = resize(picture, dimensions, anti_aliasing=True)
    
    return picture


# ============================================================
# 2. GRAPHICAL OUTPUT UTILITIES
# ============================================================

def display_palette(palette_array):
    """
    Display palette as colored rectangles.
    """
    K = palette_array.shape[0]
    plt.figure(figsize=(K, 1))
    plt.imshow([palette_array])
    plt.axis("off")
    plt.title("Extracted Color Palette")
    plt.show()


def display_picture(picture_array, caption="Picture"):
    plt.figure(figsize=(4, 4))
    plt.imshow(picture_array)
    plt.axis("off")
    plt.title(caption)
    plt.show()


# ============================================================
# 3. PALETTE-BASED COMPRESSION
# ============================================================

def compress_with_palette(picture, palette):
    """
    Compress a picture using a predetermined color palette.
    
    Parameters:
        picture   : (H, W, 3) float picture in [0,1]
        palette : (K, 3) array of RGB colors
        
    Returns:
        compressed picture
    """
    H, W, _ = picture.shape
    
    # Flatten picture to (N, 3)
    pixels = picture.reshape(-1, 3)
    
    # Calculate distance between each pixel and each palette color
    # Result shape: (N, K)
    distances = np.linalg.norm(
        pixels[:, None, :] - palette[None, :, :],
        axis=2
    )
    
    # For each pixel, identify nearest palette index
    nearest_idx = np.argmin(distances, axis=1)
    
    # Substitute each pixel with nearest palette color
    compressed_pixels = palette[nearest_idx]
    
    # Reshape back to picture
    compressed_picture = compressed_pixels.reshape(H, W, 3)
    
    return compressed_picture


# ============================================================
# 4. PERFORMANCE METRIC
# ============================================================

def compression_error(picture, palette):
    """
    Calculate total squared compression error.
    
    Fitness function to minimize.
    """
    H, W, _ = picture.shape
    pixels = picture.reshape(-1, 3)
    
    distances = np.linalg.norm(
        pixels[:, None, :] - palette[None, :, :],
        axis=2
    )
    
    minimal_distances = np.min(distances, axis=1)
    
    # Return total squared deviation
    return np.sum(minimal_distances ** 2)


# ============================================================
# 5. SWARM INTELLIGENCE OPTIMIZER
# ============================================================

class SwarmOptimizer:
    """
    Swarm intelligence implementation for palette optimization.
    Each agent represents a color palette of size K.
    """

    def __init__(self, picture, K=8, agent_count=20, inertia=0.7, cognitive=1.5, social=1.5, max_iterations=30):
        
        self.picture = picture
        self.K = K
        self.dim = 3 * K  # Each color has 3 values (RGB)
        self.agent_count = agent_count
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iterations = max_iterations
        
        # Flattened pixel array (for faster evaluation)
        self.pixels = picture.reshape(-1, 3)
        
        # Initialize agents
        self.setup_agents()
        
        # History storage
        self.history_global_best = []
        self.history_fitness = []

    def setup_agents(self):
        """
        Initialize agent positions and velocities.
        """
        # Random positions in [0,1]
        self.positions = np.random.rand(self.agent_count, self.dim)
        
        # Initialize velocities to zero
        self.velocities = np.zeros((self.agent_count, self.dim))
        
        # Personal best positions
        self.personal_best_positions = self.positions.copy()
        
        # Evaluate initial fitness
        self.personal_best_scores = np.array([
            self.evaluate_agent(p)
            for p in self.positions
        ])
        
        # Global best
        best_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[best_idx].copy()
        self.global_best_score = self.personal_best_scores[best_idx]

    def evaluate_agent(self, agent):
        """
        Evaluate fitness of an agent.
        """
        palette = agent.reshape(self.K, 3)
        return compression_error(self.picture, palette)

    def optimize(self):
        """
        Execute swarm optimization iterations.
        """
        for iteration in range(self.max_iterations):
            
            for i in range(self.agent_count):
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Velocity update
                self.velocities[i] = (
                    self.inertia * self.velocities[i]
                    + self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i])
                    + self.social * r2 * (self.global_best_position - self.positions[i])
                )
                
                # Position update
                self.positions[i] += self.velocities[i]
                
                # Clip to valid RGB range [0,1]
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                
                # Evaluate new fitness
                fitness = self.evaluate_agent(self.positions[i])
                
                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i].copy()
                
                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()
            
            # Store history
            self.history_global_best.append(self.global_best_position.copy())
            self.history_fitness.append(self.global_best_score)
            
            print(f"Iteration {iteration+1}/{self.max_iterations}, "
                  f"Best Fitness: {self.global_best_score:.4f}")

        return self.global_best_position.reshape(self.K, 3)


# ============================================================
# 6. TRADITIONAL CLUSTERING APPROACH
# ============================================================

def kmeans_compression(picture, K=8):
    """
    Perform K-means clustering for palette extraction.
    """
    H, W, _ = picture.shape
    pixels = picture.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=K, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    palette = kmeans.cluster_centers_
    
    compressed_pixels = palette[labels]
    
    return compressed_pixels.reshape(H, W, 3), palette


# ============================================================
# 7. PRIMARY EXECUTION BLOCK
# ============================================================

if __name__ == "__main__":
    
    # Acquire picture
    picture = acquire_and_scale_picture("A3/si-exercises/exercise_pso/image.png", dimensions=(128, 128))
    
    display_picture(picture, "Original Picture")
    
    # -------------------------
    # Swarm-Based Compression
    # -------------------------
    swarm = SwarmOptimizer(picture, K=8, agent_count=20, max_iterations=30)
    optimal_palette = swarm.optimize()
    
    compressed_swarm = compress_with_palette(picture, optimal_palette)
    
    display_palette(optimal_palette)
    display_picture(compressed_swarm, "Swarm-Optimized Picture")
    
    # -------------------------
    # K-Means Compression
    # -------------------------
    compressed_kmeans, kmeans_palette = kmeans_compression(picture, K=8)
    
    display_palette(kmeans_palette)
    display_picture(compressed_kmeans, "K-Means Compressed Picture")
    
    # Compare errors
    swarm_error = compression_error(picture, optimal_palette)
    kmeans_error = compression_error(picture, kmeans_palette)
    
    print("Swarm Method Error:", swarm_error)
    print("K-Means Method Error:", kmeans_error)
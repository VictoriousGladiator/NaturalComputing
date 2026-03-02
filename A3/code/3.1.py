import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
from skimage.transform import resize


# ============================================================
# 1. IMAGE LOADING + PREPROCESSING
# ============================================================

def load_and_resize_image(path, size=(128, 128)):
    """
    Load an image and optionally resize it.
    
    Parameters:
        path (str): Path to image file
        size (tuple): Desired size (H, W)
        
    Returns:
        image (float array): Image normalized to [0,1]
    """
    image = io.imread(path)
    
    # Convert to float in [0,1]
    image = image / 255.0
    
    # Resize for faster PSO if needed
    image = resize(image, size, anti_aliasing=True)
    
    return image


# ============================================================
# 2. QUANTIZATION WITH GIVEN PALETTE
# ============================================================

def quantize_image(image, palette):
    """
    Quantize an image using a fixed color palette.
    
    Parameters:
        image   : (H, W, 3) float image in [0,1]
        palette : (K, 3) array of RGB colors
        
    Returns:
        quantized image
    """
    H, W, _ = image.shape
    
    # Flatten image to (N, 3)
    pixels = image.reshape(-1, 3)
    
    # Compute distance between each pixel and each palette color
    # Result shape: (N, K)
    distances = np.linalg.norm(
        pixels[:, None, :] - palette[None, :, :],
        axis=2
    )
    
    # For each pixel, find closest palette index
    closest_color_idx = np.argmin(distances, axis=1)
    
    # Replace each pixel with closest palette color
    quantized_pixels = palette[closest_color_idx]
    
    # Reshape back to image
    quantized_image = quantized_pixels.reshape(H, W, 3)
    
    return quantized_image


# ============================================================
# 3. FITNESS FUNCTION FOR PSO
# ============================================================

def quantization_error(image, palette):
    """
    Compute total squared quantization error.
    
    Fitness function to minimize.
    """
    H, W, _ = image.shape
    pixels = image.reshape(-1, 3)
    
    distances = np.linalg.norm(
        pixels[:, None, :] - palette[None, :, :],
        axis=2
    )
    
    min_distances = np.min(distances, axis=1)
    
    # Return total squared error
    return np.sum(min_distances ** 2)


# ============================================================
# 4. PARTICLE SWARM OPTIMIZATION FOR CLUSTERING
# ============================================================

class PSOColorQuantizer:
    """
    PSO implementation for color quantization.
    Each particle represents a color palette of size K.
    """

    def __init__(self, image, K=8, n_particles=20,
                 omega=0.7, alpha1=1.5, alpha2=1.5,
                 n_iterations=30):
        
        self.image = image
        self.K = K
        self.dim = 3 * K  # Each color has 3 values (RGB)
        self.n_particles = n_particles
        self.omega = omega
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.n_iterations = n_iterations
        
        # Flattened pixel array (for faster evaluation)
        self.pixels = image.reshape(-1, 3)
        
        # Initialize particles
        self.initialize_particles()
        
        # History storage
        self.history_global_best = []
        self.history_fitness = []

    def initialize_particles(self):
        """
        Initialize particle positions and velocities.
        """
        # Random positions in [0,1]
        self.positions = np.random.rand(self.n_particles, self.dim)
        
        # Initialize velocities to zero
        self.velocities = np.zeros((self.n_particles, self.dim))
        
        # Local best positions
        self.local_best_positions = self.positions.copy()
        
        # Evaluate initial fitness
        self.local_best_scores = np.array([
            self.evaluate_particle(p)
            for p in self.positions
        ])
        
        # Global best
        best_idx = np.argmin(self.local_best_scores)
        self.global_best_position = self.local_best_positions[best_idx].copy()
        self.global_best_score = self.local_best_scores[best_idx]

    def evaluate_particle(self, particle):
        """
        Evaluate fitness of a particle.
        """
        palette = particle.reshape(self.K, 3)
        return quantization_error(self.image, palette)

    def optimize(self):
        """
        Run PSO iterations.
        """
        for iteration in range(self.n_iterations):
            
            for i in range(self.n_particles):
                
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Velocity update
                self.velocities[i] = (
                    self.omega * self.velocities[i]
                    + self.alpha1 * r1 * (self.local_best_positions[i] - self.positions[i])
                    + self.alpha2 * r2 * (self.global_best_position - self.positions[i])
                )
                
                # Position update
                self.positions[i] += self.velocities[i]
                
                # Clip to valid RGB range [0,1]
                self.positions[i] = np.clip(self.positions[i], 0, 1)
                
                # Evaluate new fitness
                fitness = self.evaluate_particle(self.positions[i])
                
                # Update local best
                if fitness < self.local_best_scores[i]:
                    self.local_best_scores[i] = fitness
                    self.local_best_positions[i] = self.positions[i].copy()
                
                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i].copy()
            
            # Store history
            self.history_global_best.append(self.global_best_position.copy())
            self.history_fitness.append(self.global_best_score)
            
            print(f"Iteration {iteration+1}/{self.n_iterations}, "
                  f"Best Fitness: {self.global_best_score:.4f}")

        return self.global_best_position.reshape(self.K, 3)


# ============================================================
# 5. K-MEANS COMPARISON
# ============================================================

def kmeans_quantization(image, K=8):
    """
    Perform K-means clustering for color quantization.
    """
    H, W, _ = image.shape
    pixels = image.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=K, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    palette = kmeans.cluster_centers_
    
    quantized_pixels = palette[labels]
    
    return quantized_pixels.reshape(H, W, 3), palette


# ============================================================
# 6. VISUALIZATION UTILITIES
# ============================================================

def show_palette(palette):
    """
    Display palette as color squares.
    """
    K = palette.shape[0]
    plt.figure(figsize=(K, 1))
    plt.imshow([palette])
    plt.axis("off")
    plt.title("Color Palette")
    plt.show()


def show_image(image, title="Image"):
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


# ============================================================
# 7. MAIN EXECUTION EXAMPLE
# ============================================================

if __name__ == "__main__":
    
    # Load image
    image = load_and_resize_image("A3/si-exercises/exercise_pso/image.png", size=(128, 128))
    
    show_image(image, "Original Image")
    
    # -------------------------
    # PSO Quantization
    # -------------------------
    pso = PSOColorQuantizer(image, K=8, n_particles=20, n_iterations=30)
    best_palette = pso.optimize()
    
    quantized_pso = quantize_image(image, best_palette)
    
    show_palette(best_palette)
    show_image(quantized_pso, "PSO Quantized Image")
    
    # -------------------------
    # K-Means Quantization
    # -------------------------
    quantized_kmeans, kmeans_palette = kmeans_quantization(image, K=8)
    
    show_palette(kmeans_palette)
    show_image(quantized_kmeans, "K-Means Quantized Image")
    
    # Compare fitness
    pso_error = quantization_error(image, best_palette)
    kmeans_error = quantization_error(image, kmeans_palette)
    
    print("PSO Error:", pso_error)
    print("K-Means Error:", kmeans_error)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import imageio

def run_gray_scott_animated(size=256, steps=100, snapshot_interval=2):
    """
    Simulates the Gray-Scott model and yields frames for animation.
    """
    # Parameters
    F, k = 0.0545, 0.062
    Du, Dv = 0.2097, 0.105

    # Grids
    U = np.ones((size, size))
    V = np.zeros((size, size))

    # Initial state
    r = int(size * 0.1)
    center = size // 2
    U[center-r:center+r, center-r:center+r] = 0.50
    V[center-r:center+r, center-r:center+r] = 0.25

    # Laplacian kernel
    laplacian_kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])
    
    # Simulation loop
    for i in range(steps * snapshot_interval):
        lap_U = convolve2d(U, laplacian_kernel, mode='same', boundary='wrap')
        lap_V = convolve2d(V, laplacian_kernel, mode='same', boundary='wrap')
        uvv = U * V * V
        U += Du * lap_U - uvv + F * (1 - U)
        V += Dv * lap_V + uvv - (F + k) * V
        
        if i % snapshot_interval == 0:
            yield V

# --- Plotting & Animation ---
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 8))
ax.axis('off')

# Generator for the frames
simulation_generator = run_gray_scott_animated(size=128, steps=150, snapshot_interval=50)

# Create frames
images = []
for i, frame_data in enumerate(simulation_generator):
    print(f"Generating frame {i+1}...")
    ax.clear()
    ax.imshow(frame_data, cmap='magma', interpolation='bicubic')
    ax.axis('off')
    fig.canvas.draw()
    
    # A more robust way to get the image data
    image = np.array(fig.canvas.buffer_rgba())
    # The buffer gives RGBA, we need RGB for the GIF.
    images.append(image[:, :, :3])

# Save as GIF
print("Saving GIF...")
imageio.mimsave('reaction_diffusion_animated.gif', images, fps=15)
print("Animation saved as reaction_diffusion_animated.gif")

# Clean up plot
plt.close(fig)

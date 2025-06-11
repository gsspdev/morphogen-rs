import streamlit as st
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

@st.cache_data
def run_gray_scott(F, k, Du, Dv, size=128, steps=7000):
    """
    Run the Gray-Scott simulation with given parameters.
    The st.cache_data decorator prevents re-running the simulation
    if the parameters haven't changed.
    """
    # Initialize grids
    U = np.ones((size, size))
    V = np.zeros((size, size))

    # Initial perturbation
    r = int(size * 0.1)
    center = size // 2
    U[center - r:center + r, center - r:center + r] = 0.50
    V[center - r:center + r, center - r:center + r] = 0.25

    # Laplacian kernel
    laplacian_kernel = np.array([[0.05, 0.2, 0.05], [0.2, -1, 0.2], [0.05, 0.2, 0.05]])

    for i in range(steps):
        lap_U = convolve2d(U, laplacian_kernel, mode='same', boundary='wrap')
        lap_V = convolve2d(V, laplacian_kernel, mode='same', boundary='wrap')
        uvv = U * V * V
        U += Du * lap_U - uvv + F * (1 - U)
        V += Dv * lap_V + uvv - (F + k) * V
    
    return V

# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Reaction-Diffusion Explorer")

st.title("🎨 Reaction-Diffusion Pattern Generator")
st.markdown("""
Welcome to the interactive pattern explorer! This app simulates the **Gray-Scott model**, 
a mathematical model of two 'virtual chemicals' reacting and diffusing. By changing the 
parameters below, you can discover a huge variety of life-like, emergent patterns. 

This is a form of **Generative Art**, where the artist sets the initial conditions and rules, 
but allows the system to create the final piece itself.
""")

st.sidebar.header("🔬 Simulation Parameters")
st.sidebar.markdown("Adjust these sliders to explore different patterns.")

# Parameter presets
presets = {
    "Mitosis (U-Skate World)": {"F": 0.0545, "k": 0.062},
    "Coral Growth": {"F": 0.058, "k": 0.065},
    "Fingerprints": {"F": 0.026, "k": 0.051},
    "Worms and Loops": {"F": 0.078, "k": 0.061},
    "Custom": {"F": 0.03, "k": 0.06},
}

preset_choice = st.sidebar.selectbox("Load a Preset", list(presets.keys()))

# Sliders for parameters
F_val = st.sidebar.slider("Feed Rate (F)", 0.01, 0.1, presets[preset_choice]['F'], 0.001, format="%.4f")
k_val = st.sidebar.slider("Kill Rate (k)", 0.04, 0.07, presets[preset_choice]['k'], 0.001, format="%.4f")

st.sidebar.markdown("---")
st.sidebar.markdown("### Advanced Parameters")
Du_val = st.sidebar.slider("Diffusion Rate (U)", 0.1, 0.3, 0.2097, 0.001, format="%.4f")
Dv_val = st.sidebar.slider("Diffusion Rate (V)", 0.05, 0.15, 0.105, 0.001, format="%.4f")


# Main panel
if st.button("🚀 Generate Artwork"):
    with st.spinner("Simulating... This may take a moment."):
        result_pattern = run_gray_scott(F_val, k_val, Du_val, Dv_val)

        fig, ax = plt.subplots(figsize=(10, 10))
        plt.style.use('dark_background')
        ax.imshow(result_pattern, cmap='magma', interpolation='bicubic')
        ax.axis('off')
        
        st.pyplot(fig)
        st.success("Simulation complete!")
        
        st.markdown("#### What do these parameters mean?")
        st.markdown(f"""
        *   **Feed Rate (F):** Controls the rate at which chemical `U` is added to the system. Think of it as the 'food' source.
        *   **Kill Rate (k):** Controls the rate at which chemical `V` is removed.
        
        The final pattern is a delicate balance between these two rates, plus the diffusion rates which control how fast the chemicals spread.
        Small changes can lead to vastly different results!
        """)
else:
    st.info("Click the 'Generate Artwork' button to begin.") 
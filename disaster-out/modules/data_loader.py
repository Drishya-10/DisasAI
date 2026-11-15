import numpy as np
from PIL import Image
import io

class FloodDataIngestor:
    """Generates and prepares spatial raster matrices (Synthetic or Uploaded Satellite Imagery)."""
    
    def __init__(self, height: int = 64, width: int = 64):
        self.h = height
        self.w = width

    def process_uploaded_image(self, file_bytes: bytes) -> tuple:
        """
        Converts uploaded satellite image (PNG/JPG/TIFF) into 4-channel tensor for U-Net:
        Channel 0: DEM (Elevation derived from brightness)
        Channel 1: Slope (Gradient)
        Channel 2: NDWI (Normalized Difference Water Index derived from Green channel)
        Channel 3: Soil Moisture (derived from Blue channel)
        """
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_resized = img.resize((self.w, self.h))
        arr = np.array(img_resized).astype(np.float32) / 255.0  # Normalize (H, W, 3)
        
        # Channel derivation for PyTorch U-Net input
        ch0_dem = arr[:, :, 0] * 300.0  # Red channel -> DEM proxy
        dy, dx = np.gradient(ch0_dem)
        ch1_slope = np.sqrt(dx**2 + dy**2)  # Slope gradient
        ch2_ndwi = (arr[:, :, 1] - arr[:, :, 0]) / (arr[:, :, 1] + arr[:, :, 0] + 1e-5)  # NDWI proxy
        ch3_moisture = arr[:, :, 2]  # Blue channel -> Moisture proxy
        
        raster = np.stack([ch0_dem, ch1_slope, ch2_ndwi, ch3_moisture], axis=0).astype(np.float32)
        return raster, img

    def generate_synthetic_raster(self, seed: int = 42) -> np.ndarray:
        """Fallback synthetic satellite raster generator."""
        np.random.seed(seed)
        x = np.linspace(0, 5, self.w)
        y = np.linspace(0, 5, self.h)
        xx, yy = np.meshgrid(x, y)
        dem = 100 * (np.sin(xx) + np.cos(yy)) + 200
        dy, dx = np.gradient(dem)
        slope = np.sqrt(dx**2 + dy**2)
        ndwi = np.random.uniform(-0.5, 0.8, (self.h, self.w))
        moisture = np.random.uniform(0.2, 0.95, (self.h, self.w))
        return np.stack([dem, slope, ndwi, moisture], axis=0).astype(np.float32)

    def generate_rainfall_sequence(self, timesteps: int = 6, seed: int = 42) -> np.ndarray:
        """Generates spatio-temporal rainfall forecast sequence (mm/hr)."""
        np.random.seed(seed)
        sequence = []
        base_intensity = np.random.uniform(5, 45, (self.h, self.w))
        for t in range(timesteps):
            shift = t * 2
            storm_cell = np.roll(base_intensity, shift=shift, axis=1) * (1.0 + 0.15 * t)
            sequence.append(storm_cell[np.newaxis, :, :])
        return np.stack(sequence, axis=0).astype(np.float32)
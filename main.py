import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AnomalyDetector:
    """
    Detects anomalies using Exponential Moving Average (EMA) and Z-score.
    Capable of adapting to concept drift and seasonal variations.
    """
    def __init__(self, window_size=50, threshold=3, ema_alpha=0.1):
        """
        Initialize anomaly detector.

        :param window_size: Number of data points considered in sliding window.
        :param threshold: Z-score threshold for detecting anomalies.
        :param ema_alpha: Alpha value for Exponential Moving Average (EMA).
        """
        self.window_size = window_size
        self.threshold = threshold
        self.ema_alpha = ema_alpha
        self.data_window = []
        self.ema = None  # Exponential Moving Average initialization
    
    def update(self, new_value):
        """Updates the window and calculates the Exponential Moving Average (EMA)."""
        # Maintain window size
        if len(self.data_window) >= self.window_size:
            self.data_window.pop(0)
        self.data_window.append(new_value)

        # Update EMA
        if self.ema is None:
            self.ema = new_value  # First value initializes EMA
        else:
            self.ema = (self.ema_alpha * new_value) + (1 - self.ema_alpha) * self.ema

    def detect_anomaly(self, new_value):
        """Checks if a new value is an anomaly based on Z-score."""
        self.update(new_value)

        # Only detect anomalies when window is full
        if len(self.data_window) < self.window_size:
            return False, None

        # Calculate statistics
        mean = np.mean(self.data_window)
        std = np.std(self.data_window)

        # Handle division by zero in case of very low variance
        if std == 0:
            return False, None
        
        # Z-score using EMA
        z_score = (new_value - self.ema) / std

        if abs(z_score) > self.threshold:
            return True, z_score
        return False, z_score

def generate_data_stream():
    """
    Simulates a continuous data stream with seasonal variation, random noise, and occasional anomalies.
    """
    time = 0
    while True:
        # Seasonal pattern (sine wave)
        seasonal = np.sin(2 * np.pi * time / 50)  
        # Random noise
        noise = random.uniform(-0.1, 0.1)
        # Simulate occasional anomalies
        anomaly = random.choice([0, 0, 0, 0, 5 * random.uniform(0.8, 1.2)])
        value = seasonal + noise + anomaly
        time += 1
        yield value

def visualize_real_time_anomalies():
    """
    Visualizes the data stream in real-time with Matplotlib, highlighting detected anomalies.
    """
    detector = AnomalyDetector(window_size=50, threshold=3, ema_alpha=0.1)
    data_stream = generate_data_stream()

    # Data containers for plotting
    x_data, y_data = [], []
    anomaly_x, anomaly_y = [], []

    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Data Stream")
    anomaly_scatter = ax.scatter([], [], color='red', label="Anomalies")
    ax.legend()
    
    def init():
        ax.set_xlim(0, 100)
        ax.set_ylim(-2, 7)
        return line, anomaly_scatter

    def update(frame):
        value = next(data_stream)
        
        # Detect anomaly
        is_anomaly, z_score = detector.detect_anomaly(value)

        # Update data stream
        x_data.append(frame)
        y_data.append(value)
        if len(x_data) > 100:
            x_data.pop(0)
            y_data.pop(0)
        line.set_data(x_data, y_data)

        # Update anomaly points
        if is_anomaly:
            anomaly_x.append(frame)
            anomaly_y.append(value)
            anomaly_scatter.set_offsets(np.c_[anomaly_x, anomaly_y])

        # Adjust plot x-limits dynamically
        ax.set_xlim(max(0, frame - 100), frame + 1)

        return line, anomaly_scatter

    ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), init_func=init, blit=True, interval=100)
    plt.show()

if __name__ == "__main__":
    visualize_real_time_anomalies()

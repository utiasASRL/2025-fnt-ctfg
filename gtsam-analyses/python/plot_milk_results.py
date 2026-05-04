import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "results"


def plotGiantGlassofMilkResults(filename, start_time=None, end_time=None):
    
    # Ensure filename is a Path
    filename = Path(filename)

    # Load data
    data = np.genfromtxt(str(filename), delimiter=',', dtype=float, skip_header=0)
    if start_time is None:
        start_time = data[:, 0].min()
    if end_time is None:
        end_time = data[:, 0].max()

    # First 6 columns of data are results
    results = data[:, :6]
    # Last 3 columns of data are measurements
    measurements = data[:, 6:]
    # Remove rows from measurements, where the first column is zero, unless it's the first row
    measurements = measurements[~((measurements[:, 0] == 0) & (np.arange(measurements.shape[0]) != 0))]
    # Filter results based on time
    mask = (results[:, 0] >= start_time) & (results[:, 0] <= end_time)
    results = results[mask]
    # Filter measurements based on time
    mask = (measurements[:, 0] >= start_time) & (measurements[:, 0] <= end_time)
    measurements = measurements[mask]

    # split data
    times = results[:, 0]
    x_est = results[:, 1]
    x_std = results[:, 2]
    v_est = results[:, 3]
    v_std = results[:, 4]
    x_real = results[:, 5]
    times_meas = measurements[:, 0]
    x_meas = measurements[:, 1]
    v_meas = measurements[:, 2]

    # use x_real and times to compute velocity
    v_real = np.gradient(x_real, times)

    # Plot for position
    fig1, ax1 = plt.subplots()
    ax1.plot(times, x_est, label="Estimated Position", color="blue")
    ax1.fill_between(times, x_est - 3*x_std, x_est + 3*x_std, alpha=0.2, color="blue")
    ax1.scatter(times_meas, x_meas, label="Measured Position", color="red", marker='x')
    ax1.plot(times, x_real, label="True Position", linestyle='--', color="green")

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Position [m]")
    ax1.set_title("Position Estimate")
    ax1.legend()

    # Plot for velocity
    fig2, ax2 = plt.subplots()
    ax2.plot(times, v_est, label="Estimated Velocity", color="blue")
    ax2.fill_between(times, v_est - 3*v_std, v_est + 3*v_std, alpha=0.2, color="blue")
    ax2.scatter(times_meas, v_meas, label="Measured Velocity", color="red", marker='x')
    ax2.plot(times, v_real, label="True Velocity", linestyle='--', color="green")

    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Velocity [m/s]")
    ax2.set_title("Velocity Estimate")
    ax2.legend()

    # Save figures to PROJECT_DIR/plots instead of showing
    plots_dir = PROJECT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def _fmt(x):
        s = str(x)
        return s.replace('.', 'p')

    base = filename.stem
    pos_name = plots_dir / f"{base}_position_{_fmt(start_time)}-{_fmt(end_time)}.png"
    vel_name = plots_dir / f"{base}_velocity_{_fmt(start_time)}-{_fmt(end_time)}.png"

    fig1.savefig(pos_name, dpi=200, bbox_inches='tight')
    fig2.savefig(vel_name, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    filename = RESULTS_DIR / "milk.csv"
    plotGiantGlassofMilkResults(filename=filename, start_time=50, end_time=100)

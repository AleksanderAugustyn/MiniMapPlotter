import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


def read_data(filename):
    data = np.loadtxt(filename)
    return data


def create_custom_cmap():
    colors = ['#648FFF', '#785EF0', '#DC267F', '#FE6100', '#FFB000']
    return LinearSegmentedColormap.from_list("custom_cmap", colors)


def filter_data(data):
    mask = (data[:, 1] <= 500.0) & (data[:, 2] <= 400.0) & (data[:, 3] <= 400.0)
    return data[mask]


def create_3d_plot(ax: Axes3D, data, z_col, title, cmap):
    x = data[:, 5]  # B20 column
    y = data[:, 6]  # B30 column
    z = data[:, z_col]

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='cubic')

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='black', alpha=1.0, linewidth=0.2)

    # Use public methods to configure grid properties
    ax.grid(linestyle=':', color='black', linewidth=0.5, alpha=0.75)

    ax.set_xlabel('$B_{20}$', fontsize=12)
    ax.set_ylabel('$B_{30}$', fontsize=12)
    ax.set_zlabel(f'{title} (MeV)', fontsize=12)
    ax.set_title(title, fontsize=18)
    return surf


def create_2d_plot(number_of_protons, number_of_neutrons, ax: plt.Axes, data, z_col, title, cmap):
    x = data[:, 5]  # B20 column
    y = data[:, 6]  # B30 column
    z = data[:, z_col]

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    X, Y = np.meshgrid(xi, yi)

    Z = griddata((x, y), z, (X, Y), method='cubic')

    im = ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
                   origin='lower', aspect='auto', cmap=cmap)

    levels = np.arange(np.floor(z.min()), np.ceil(z.max()) + 1, 1)
    contour = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.75)
    ax.clabel(contour, inline=True, fontsize=12, fmt='%1.0f', colors='black', inline_spacing=1)

    # Add markers based on proton and neutron numbers
    markers = {
        (90, 140): [(0.19, 0.0, 'ko'), (0.75, 0.15, 'ks')],
        (92, 144): [(0.21, 0.0, 'ko'), (0.75, 0.15, 'ks')],
        (94, 146): [(0.23, 0.0, 'ko'), (0.85, 0.20, 'ks')],
        (98, 152): [(0.25, 0.0, 'ko'), (1.00, 0.10, 'ks')]
    }

    if (number_of_protons, number_of_neutrons) in markers:
        for x, y, marker in markers[(number_of_protons, number_of_neutrons)]:
            ax.plot(x, y, marker, markersize=10)

    ax.set_xlabel('$B_{20}$', fontsize=12)
    ax.set_ylabel('$B_{30}$', fontsize=12)
    ax.set_title(title, fontsize=18)
    return im


def create_plots(data, Z, N, dimensions, plot_type):
    fig = plt.figure(figsize=(18, 12))

    fig.suptitle(f"$B_{{20}}B_{{30}}$ {plot_type} Map P={Z} N={N} ({dimensions}D)", fontsize=28)

    titles = ['E', 'ELD', 'ESH']
    z_columns = [1, 2, 3]  # Columns for E, ELD, ESH

    cmap = create_custom_cmap()

    for i, (title, z_col) in enumerate(zip(titles, z_columns)):
        # 3D plot
        ax1 = fig.add_subplot(2, 3, i + 1, projection='3d')
        surf = create_3d_plot(ax1, data, z_col, title, cmap)
        cbar = fig.colorbar(surf, ax=ax1, label=f'{title} (MeV)', pad=0.1)
        cbar.ax.yaxis.label.set_fontsize(12)

        # 2D plot
        ax2 = fig.add_subplot(2, 3, i + 4)
        im = create_2d_plot(Z, N, ax2, data, z_col, title, cmap)
        cbar = fig.colorbar(im, ax=ax2, label=f'{title} (MeV)')
        cbar.ax.yaxis.label.set_fontsize(12)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    return fig


def process_file(filename, Z, N, dimensions, plot_type):
    try:
        data = read_data(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if data.size == 0:
        print(f"No data found in the file {filename}")
        return

    filtered_data = filter_data(data)

    if filtered_data.size == 0:
        print(f"No data left after filtering for file {filename}")
        return

    fig = create_plots(filtered_data, Z, N, dimensions, plot_type)

    plt.show()

    output_filename = f"{Z}_{N}_{dimensions}D_{plot_type}_MiniMap.png"
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")

    plt.close(fig)

    # Clear memory
    del data
    del filtered_data
    gc.collect()


def main(Z, N):
    # Process all file types
    file_types = [
        ('Minimized', ''),
        ('Starting', '_Starting'),
        ('Fusion', '_Fusion')
    ]

    for plot_type, suffix in file_types:
        for dim in ['6', '4']:
            filename = os.path.join("MiniMaps", f"{Z}_{N}_{dim}D_B20B30{suffix}_MiniMap.txt")
            process_file(filename, Z, N, dim, plot_type)

    # Process B10 files
    for B10 in np.arange(-1.6, 1.61, 0.05):
        filename = os.path.join("MiniMaps", f"{Z}_{N}_6D_B10const_MiniMap_B10_{B10:.3f}.txt")
        process_file(filename, Z, N, "6", f"B10_{B10:.3f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <Protons> <Neutrons>")
    else:
        Protons = int(sys.argv[1])
        Neutrons = int(sys.argv[2])
        main(Protons, Neutrons)

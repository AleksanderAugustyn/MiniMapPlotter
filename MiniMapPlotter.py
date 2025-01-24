import sys
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting


def create_output_directories(number_of_protons, number_of_neutrons):
    """Create the directory structure for output files."""
    base_dir = f"{number_of_protons}_{number_of_neutrons}"
    b10_dir = os.path.join(base_dir, "B10")
    b40_dir = os.path.join(base_dir, "B40")
    b10b40_dir = os.path.join(base_dir, "B10B40")

    # Create directories if they don't exist
    for directory in [base_dir, b10_dir, b40_dir, b10b40_dir]:
        os.makedirs(directory, exist_ok=True)

    return base_dir, b10_dir, b40_dir, b10b40_dir


def read_data(filename):
    data = np.loadtxt(filename)

    # If the file has less than 600 rows, skip it
    if data.shape[0] < 600:
        return np.array([])

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

    Z_data = griddata((x, y), z, (X, Y), method='cubic')

    im = ax.imshow(Z_data, extent=(x.min(), x.max(), y.min(), y.max()),
                   origin='lower', aspect='auto', cmap=cmap)

    levels = np.arange(np.floor(z.min()), np.ceil(z.max()) + 1, 1)
    contour = ax.contour(X, Y, Z_data, levels=levels, colors='black', alpha=0.75)
    ax.clabel(contour, inline=True, fontsize=12, fmt='%1.0f', colors='black', inline_spacing=1)

    markers = {
        (90, 140): [(0.19, 0.0, 'ko'), (0.75, 0.15, 'ks')],
        (92, 144): [(0.21, 0.0, 'ko'), (0.75, 0.15, 'ks')],
        (94, 146): [(0.23, 0.0, 'ko'), (0.85, 0.20, 'ks')],
        (96, 150): [(0.24, 0.0, 'ko'), (0.85, 0.20, 'ks')],
        (98, 152): [(0.25, 0.0, 'ko'), (1.00, 0.10, 'ks')]
    }

    if (number_of_protons, number_of_neutrons) in markers:
        for x, y, marker in markers[(number_of_protons, number_of_neutrons)]:
            ax.plot(x, y, marker, markersize=10)

    ax.set_xlabel('$B_{20}$', fontsize=12)
    ax.set_ylabel('$B_{30}$', fontsize=12)
    ax.set_title(title, fontsize=18)
    return im


def create_plots(data, number_of_protons, number_of_neutrons, dimensions, plot_type):
    fig = plt.figure(figsize=(18, 12))

    fig.suptitle(f"$B_{{20}}B_{{30}}$ {plot_type} Map P={number_of_protons} N={number_of_neutrons} ({dimensions}D)", fontsize=28)

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
        im = create_2d_plot(number_of_protons, number_of_neutrons, ax2, data, z_col, title, cmap)
        cbar = fig.colorbar(im, ax=ax2, label=f'{title} (MeV)')
        cbar.ax.yaxis.label.set_fontsize(12)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    return fig


def create_33grid_plot(number_of_protons, number_of_neutrons, deformation_type, values, file_pattern, output_dir):
    """Create a 3x3 grid of 2D E plots for different deformation values (B10 or B40)."""
    fig = plt.figure(figsize=(20, 20))
    fig.suptitle(f"Energy Surface for Different $B_{{{deformation_type}}}$ Values (P={number_of_protons}, N={number_of_neutrons})",
                 fontsize=24)

    cmap = create_custom_cmap()
    all_z_values = []

    # First pass: collect all z values
    for value in values:
        filename = os.path.join("MiniMaps", file_pattern.format(
            protons=number_of_protons,
            neutrons=number_of_neutrons,
            value=value))
        try:
            data = read_data(filename)
            if data.size > 0:
                filtered_data = filter_data(data)
                if filtered_data.size > 0:
                    all_z_values.extend(filtered_data[:, 1])
        except Exception:
            continue

    if not all_z_values:
        print(f"No valid data found for B{deformation_type} grid plot")
        return None

    vmin, vmax = np.min(all_z_values), np.max(all_z_values)

    # Second pass: create plots
    for idx, value in enumerate(values):
        filename = os.path.join("MiniMaps", file_pattern.format(
            protons=number_of_protons,
            neutrons=number_of_neutrons,
            value=value))

        try:
            data = read_data(filename)
            if data.size == 0:
                continue

            filtered_data = filter_data(data)
            if filtered_data.size == 0:
                continue

            ax = fig.add_subplot(3, 3, idx + 1)
            x = filtered_data[:, 5]
            y = filtered_data[:, 6]
            z = filtered_data[:, 1]

            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            X, Y = np.meshgrid(xi, yi)

            Z = griddata((x, y), z, (X, Y), method='cubic')

            im = ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
                           origin='lower', aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax)

            levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)
            contour = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.75)
            ax.clabel(contour, inline=True, fontsize=12, fmt='%1.0f', colors='black', inline_spacing=1)

            markers = {
                (90, 140): [(0.19, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                (92, 144): [(0.21, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                (94, 146): [(0.23, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                (96, 150): [(0.24, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                (98, 152): [(0.25, 0.0, 'ko'), (1.00, 0.10, 'ks')]
            }

            if (number_of_protons, number_of_neutrons) in markers:
                for x, y, marker in markers[(number_of_protons, number_of_neutrons)]:
                    ax.plot(x, y, marker, markersize=10)

            ax.set_xlabel('$B_{20}$', fontsize=12)
            ax.set_ylabel('$B_{30}$', fontsize=12)
            ax.set_title(f'$B_{{{deformation_type}}} = {value:.3f}$', fontsize=14)

        except Exception as e:
            print(f"Error processing B{deformation_type}={value}: {str(e)}")
            continue

    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Energy (MeV)', fontsize=12)

    plt.tight_layout(rect=(0.0, 0.03, 0.9, 0.95))

    if show_plots:
        plt.show()

    output_filename = os.path.join(output_dir, f"{number_of_protons}_{number_of_neutrons}_B{deformation_type}_33grid_plot.png")
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"B{deformation_type} grid plot saved as {output_filename}")

    plt.close(fig)
    return fig


def create_32grid_plot(number_of_protons, number_of_neutrons, deformation_type, values, file_pattern, output_dir):
    """Create a 3x2 grid of 2D E plots for different deformation values (B10 or B40)."""
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle(f"Energy Surface for Different $B_{{{deformation_type}}}$ Values (P={number_of_protons}, N={number_of_neutrons})",
                 fontsize=24)

    cmap = create_custom_cmap()
    all_z_values = []

    # First pass: collect all z values
    for value in values:
        filename = os.path.join("MiniMaps", file_pattern.format(
            protons=number_of_protons,
            neutrons=number_of_neutrons,
            value=value))
        try:
            data = read_data(filename)
            if data.size > 0:
                filtered_data = filter_data(data)
                if filtered_data.size > 0:
                    all_z_values.extend(filtered_data[:, 1])
        except Exception:
            continue

    if not all_z_values:
        print(f"No valid data found for B{deformation_type} grid plot")
        return None

    vmin, vmax = np.min(all_z_values), np.max(all_z_values)

    # Second pass: create plots
    for idx, value in enumerate(values):
        filename = os.path.join("MiniMaps", file_pattern.format(
            protons=number_of_protons,
            neutrons=number_of_neutrons,
            value=value))

        try:
            data = read_data(filename)
            if data.size == 0:
                continue

            filtered_data = filter_data(data)
            if filtered_data.size == 0:
                continue

            ax = fig.add_subplot(3, 3, idx + 1)
            x = filtered_data[:, 5]
            y = filtered_data[:, 6]
            z = filtered_data[:, 1]

            xi = np.linspace(x.min(), x.max(), 100)
            yi = np.linspace(y.min(), y.max(), 100)
            X, Y = np.meshgrid(xi, yi)

            Z = griddata((x, y), z, (X, Y), method='cubic')

            im = ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
                           origin='upper', aspect='auto', cmap=cmap,
                           vmin=vmin, vmax=vmax)

            levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)
            contour = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.75)
            ax.clabel(contour, inline=True, fontsize=12, fmt='%1.0f', colors='black', inline_spacing=1)

            markers = {
                (90, 140): [(0.19, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                (92, 144): [(0.21, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                (94, 146): [(0.23, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                (96, 150): [(0.24, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                (98, 152): [(0.25, 0.0, 'ko'), (1.00, 0.10, 'ks')]
            }

            if (number_of_protons, number_of_neutrons) in markers:
                for x, y, marker in markers[(number_of_protons, number_of_neutrons)]:
                    ax.plot(x, y, marker, markersize=10)

            ax.set_xlabel('$B_{20}$', fontsize=12)
            ax.set_ylabel('$B_{30}$', fontsize=12)
            ax.set_title(f'$B_{{{deformation_type}}} = {value:.3f}$', fontsize=14)

        except Exception as e:
            print(f"Error processing B{deformation_type}={value}: {str(e)}")
            continue

    cbar_ax = fig.add_axes((0.92, 0.35, 0.02, 0.6))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Energy (MeV)', fontsize=12)

    plt.tight_layout(rect=(0.0, 0.03, 0.9, 0.95))

    if show_plots:
        plt.show()

    output_filename = os.path.join(output_dir, f"{number_of_protons}_{number_of_neutrons}_B{deformation_type}_32grid_plot.png")
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"B{deformation_type} grid plot saved as {output_filename}")

    plt.close(fig)
    return fig


def process_file(filename, number_of_protons, number_of_neutrons, dimensions, plot_type, output_dir):
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

    fig = create_plots(filtered_data, number_of_protons, number_of_neutrons, dimensions, plot_type)

    if show_plots:
        plt.show()

    output_filename = os.path.join(output_dir, f"{number_of_protons}_{number_of_neutrons}_{dimensions}D_{plot_type}_MiniMap.png")
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {output_filename}")

    plt.close(fig)

    del data
    del filtered_data
    gc.collect()


def create_double_grid_plot(number_of_protons, number_of_neutrons, b10_values, b40_values, output_dir):
    """Create a 5x5 grid of 2D E plots for different combinations of B10 and B40 values."""
    fig = plt.figure(figsize=(25, 25))
    fig.suptitle(f"Energy Surface for Different $B_{{10}}$ and $B_{{40}}$ Values\n(P={number_of_protons}, N={number_of_neutrons})",
                 fontsize=24)

    cmap = create_custom_cmap()
    all_z_values = []

    # First pass: collect all z values for consistent colorbar
    for b10_idx, b10_value in enumerate(b10_values):
        for b40_idx, b40_value in enumerate(b40_values):
            filename = os.path.join("MiniMaps", f"{number_of_protons}_{number_of_neutrons}_6D_B20B30_MiniMap_B10_{b10_value:.3f}_B40_{b40_value:.3f}.txt")
            try:
                data = read_data(filename)
                if data.size > 0:
                    filtered_data = filter_data(data)
                    if filtered_data.size > 0:
                        all_z_values.extend(filtered_data[:, 1])
            except Exception:
                continue

    if not all_z_values:
        print("No valid data found for B10-B40 grid plot")
        return None

    vmin, vmax = np.min(all_z_values), np.max(all_z_values)

    # Second pass: create plots
    for b10_idx, b10_value in enumerate(b10_values):
        for b40_idx, b40_value in enumerate(b40_values):
            filename = os.path.join("MiniMaps", f"{number_of_protons}_{number_of_neutrons}_6D_B20B30_MiniMap_B10_{b10_value:.3f}_B40_{b40_value:.3f}.txt")

            # Calculate plot position (1-based indexing for subplot)
            plot_idx = b10_idx * len(b40_values) + b40_idx + 1

            try:
                data = read_data(filename)
                if data.size == 0:
                    continue

                filtered_data = filter_data(data)
                if filtered_data.size == 0:
                    continue

                ax = fig.add_subplot(len(b10_values), len(b40_values), plot_idx)
                x = filtered_data[:, 5]  # B20 column
                y = filtered_data[:, 6]  # B30 column
                z = filtered_data[:, 1]  # Energy column

                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                X, Y = np.meshgrid(xi, yi)

                Z = griddata((x, y), z, (X, Y), method='cubic')

                im = ax.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()),
                               origin='lower', aspect='auto', cmap=cmap,
                               vmin=vmin, vmax=vmax)

                levels = np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1)
                contour = ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.75)
                ax.clabel(contour, inline=True, fontsize=10, fmt='%1.0f', colors='black', inline_spacing=1)

                # Add markers if applicable
                markers = {
                    (90, 140): [(0.19, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                    (92, 144): [(0.21, 0.0, 'ko'), (0.75, 0.15, 'ks')],
                    (94, 146): [(0.23, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                    (96, 150): [(0.24, 0.0, 'ko'), (0.85, 0.20, 'ks')],
                    (98, 152): [(0.25, 0.0, 'ko'), (1.00, 0.10, 'ks')]
                }

                if (number_of_protons, number_of_neutrons) in markers:
                    for x, y, marker in markers[(number_of_protons, number_of_neutrons)]:
                        ax.plot(x, y, marker, markersize=8)

                # Only show x-label for bottom row
                if b10_idx == len(b10_values) - 1:
                    ax.set_xlabel('$B_{20}$', fontsize=10)
                else:
                    ax.set_xticklabels([])

                # Only show y-label for leftmost column
                if b40_idx == 0:
                    ax.set_ylabel('$B_{30}$', fontsize=10)
                else:
                    ax.set_yticklabels([])

                ax.set_title(f'$B_{{10}}={b10_value:.3f}, B_{{40}}={b40_value:.3f}$', fontsize=12)

            except Exception as e:
                print(f"Error processing B10={b10_value}, B40={b40_value}: {str(e)}")
                continue

    # Add colorbar
    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Energy (MeV)', fontsize=12)

    plt.tight_layout(rect=(0.0, 0.03, 0.9, 0.95))

    if show_plots:
        plt.show()

    output_filename = os.path.join(output_dir, f"{number_of_protons}_{number_of_neutrons}_B10B40_grid_plot.png")
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"B10-B40 grid plot saved as {output_filename}")

    plt.close(fig)
    return fig


def main(number_of_protons, number_of_neutrons):
    # Create directory structure
    base_dir, b10_dir, b40_dir, b10b40_dir = create_output_directories(number_of_protons, number_of_neutrons)

    # Process standard file types (save to base directory)
    file_types = [
        ('Minimized', ''),
        ('Starting', '_Starting'),
        ('Fusion', '_Fusion')
    ]

    for plot_type, suffix in file_types:
        for dim in ['6', '4']:
            filename = os.path.join("MiniMaps", f"{number_of_protons}_{number_of_neutrons}_{dim}D_B20B30{suffix}_MiniMap.txt")
            process_file(filename, number_of_protons, number_of_neutrons, dim, plot_type, base_dir)

    # Create and save B10 grid plot
    b10_33grid_values = [-0.400, -0.300, -0.200, -0.100, 0.000, 0.100, 0.200, 0.300, 0.400]
    b10_32grid_values = [0.000, 0.100, 0.200, 0.300, 0.400, 0.500]
    b10_file_pattern = "{protons}_{neutrons}_6D_B10const_MiniMap_B10_{value:.3f}.txt"
    create_33grid_plot(number_of_protons, number_of_neutrons, "10", b10_33grid_values, b10_file_pattern, base_dir)
    create_32grid_plot(number_of_protons, number_of_neutrons, "10", b10_32grid_values, b10_file_pattern, base_dir)

    # Create and save B40 grid plot
    b40_33grid_values = [-0.400, -0.300, -0.200, -0.100, 0.000, 0.100, 0.200, 0.300, 0.400]
    b40_file_pattern = "{protons}_{neutrons}_6D_B40const_MiniMap_B40_{value:.3f}.txt"
    create_33grid_plot(number_of_protons, number_of_neutrons, "40", b40_33grid_values, b40_file_pattern, base_dir)

    # Process individual B10 files
    for B10 in np.arange(-1.6, 1.61, 0.05):
        filename = os.path.join("MiniMaps", f"{number_of_protons}_{number_of_neutrons}_6D_B10const_MiniMap_B10_{B10:.3f}.txt")
        process_file(filename, number_of_protons, number_of_neutrons, "6", f"B10_{B10:.3f}", b10_dir)

    # Process individual B40 files for specified values
    for B40 in np.arange(-0.5, 0.51, 0.05):
        if abs(B40) <= 1e-10:
            B40 = 0.0

        filename = os.path.join("MiniMaps", f"{number_of_protons}_{number_of_neutrons}_6D_B40const_MiniMap_B40_{B40:.3f}.txt")
        process_file(filename, number_of_protons, number_of_neutrons, "6", f"B40_{B40:.3f}", b40_dir)

    # Create and save B10-B40 combined grid plot
    b10b40_grid_values = [-0.400, -0.200, 0.000, 0.200, 0.400]
    create_double_grid_plot(number_of_protons, number_of_neutrons, b10b40_grid_values, b10b40_grid_values, b10b40_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <Protons> <Neutrons>")
    else:
        Protons = int(sys.argv[1])
        Neutrons = int(sys.argv[2])

        show_plots = False

        main(Protons, Neutrons)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import product, combinations

def plot_3d_norm_balls(center, norm_types=('infinity', 'l1', 'l2'), radius_dict=None, same_figure=True):
    x_center, y_center, z_center = center
    
    if radius_dict is None:
        radius_dict = {norm: 1 for norm in norm_types}  # Default radius of 1 for all if not specified
    
    if same_figure:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        axs = [ax] * len(norm_types)  # Use the same ax for all plots
    else:
        figs = [plt.figure(figsize=(6, 6)) for _ in norm_types]
        axs = [fig.add_subplot(111, projection='3d') for fig in figs]

    for ax, norm_type in zip(axs, norm_types):
        radius = radius_dict.get(norm_type, 1)  # Get the radius for each norm type

        if norm_type == 'infinity':
            # Plot for L-infinity norm (cube)
            r = [-radius, radius]
            for s, e in combinations(np.array(list(product(r, r, r))), 2):
                if np.sum(np.abs(s-e)) == r[1]-r[0]:
                    ax.plot3D(*zip(s+center, e+center), color="b")

        elif norm_type == 'l1':
            # Plot for L1 norm (octahedron)
            vertices = np.array([
                [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
            ]) * radius + center
            faces = [[vertices[i] for i in [0, 2, 4]], [vertices[i] for i in [0, 3, 4]], 
                     [vertices[i] for i in [0, 2, 5]], [vertices[i] for i in [0, 3, 5]],
                     [vertices[i] for i in [1, 2, 4]], [vertices[i] for i in [1, 3, 4]],
                     [vertices[i] for i in [1, 2, 5]], [vertices[i] for i in [1, 3, 5]]]
            face_collection = Poly3DCollection(faces, facecolors='g', linewidths=1, edgecolors='r', alpha=0.25)
            ax.add_collection3d(face_collection)

        elif norm_type == 'l2':
            # Plot for L2 norm (sphere)
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u)*np.sin(v)*radius + x_center
            y = np.sin(u)*np.sin(v)*radius + y_center
            z = np.cos(v)*radius + z_center
            ax.plot_surface(x, y, z, color='r', alpha=0.6)

        # Plot center point
        ax.scatter(*center, color='k', s=100, marker='o', depthshade=True)  # Black point at the center

        # Set plot limits and labels
        max_radius = max(radius_dict.values()) * 1.5
        ax.set_xlim(x_center - max_radius, x_center + max_radius)
        ax.set_ylim(y_center - max_radius, y_center + max_radius)
        ax.set_zlim(z_center - max_radius, z_center + max_radius)
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1

        if not same_figure:
            ax.set_title(f"{norm_type.upper()} Norm Ball (3D), r={radius}")

    if same_figure:
        ax.set_title("Combined Norm Balls in 3D")
        plt.tight_layout()
        plt.show(block=True)
    else:
        for fig in figs:
            fig.tight_layout()
            fig.show(block=True)

import numpy as np
import h5py
import matplotlib.pyplot as plt
import os

Np = np.int64(256) ** 3  # number of particles
mp = np.float32(3.64453e10)  # particle mass in Msun; all 32-bit to save memory
G = np.float32(4.3009e-9)  # gravitational constant in Mpc*(km/s)^2/Msun
h = np.float32(
    0.3755
)  # Hubble parameter (this is a Einstein-de Sitter universe with Omega_m=1)
L = np.float32(250.0)  # side length of periodic cubic simulation volume
scale_factor = np.float32(0.1)  # scale factor a
redshift = 1.0 / scale_factor - 1
rho_mean = (
    Np * mp / L**3
)  # mean density in Msun/Mpc^3 (comoving, matches 3*H_0^2/(8*pi*G))


class Octree_Node:
    """
    A node in a 3D octree.

    Parameters
    ----------
    center_position : ndarray
        Geometrical center of the node, shape (3,)
    depth : int
        Depth of the node. The root has depth 0
    index : tuple
        Integer index of the node at its depth
    CoM : ndarray
        Centre-of-mass position, shape (3,)
    particles : ndarray
        Particle indices inside the node
        This is only needed for leaf nodes / bonus question
    children : ndarray or None
        Child nodes, shape (2, 2, 2), or None for leaf nodes
    """

    def __init__(
        self,
        center_position,
        size,
        depth,
        index,
        particles,
        all_particles,
        children=None,
    ):  # note: the fewer of these you keep, the better! these are just examples
        self.center_position = center_position
        self.size = size
        self.depth = depth
        self.index = index
        self.particles = particles
        self.all_particles = all_particles
        self.children = children

        if self.particles is None or len(self.particles) == 0:
            self.mass = np.float32(0.0)
            self.CoM = center_position
        else:
            self.mass = np.float32(len(self.particles)) * mp
            self.CoM = np.mean(self.all_particles[self.particles], axis=0)

    def build_octree(
        self,
        max_depth=7,
    ):
        """
        Build an octree recursively.

        Parameters
        ----------
        particle_positions : ndarray
            Particle positions inside the current node, shape (N_node, 3)
        center_position : ndarray
            Geometrical center of the current node, shape (3,)
        index : tuple
            Integer index of this node at the current depth, (ix, iy, iz)
        box_size : float
            Side length of the current node
        depth : int
            Current octree depth
        max_depth : int
            Maximum tree depth

        Returns
        -------
        node : Octree_Node
            Current octree node
        """
        # Stop when the node is empty or a max depth is reached
        if (
            self.particles is None
            or len(self.particles) == 0
            or self.depth >= max_depth
        ):
            return
        
        # Extract the positions of the particles in the cell
        idx = self.particles
        pos = self.all_particles[idx]

        # Make masks for the location of the partcicles relative to the center
        cx, cy, cz = self.center_position
        right = pos[:, 0] > cx
        top = pos[:, 1] > cy
        front = pos[:, 2] > cz

        # Make placeholder children for the node
        self.children = np.empty((2, 2, 2), dtype=object)

        # Split the node into 8 (2 per axis)
        ix, iy, iz = self.index
        for i in (0, 1):
            x_mask = right == i
            for j in (0, 1):
                y_mask = top == j
                for k in (0, 1):
                    z_mask = front == k

                    # Masking the particles in this octant
                    mask = x_mask & y_mask & z_mask
                    child_particles = idx[mask]

                    # Skip the empty octant
                    if len(child_particles) == 0:
                        self.children[i, j, k] = None
                        continue
                    
                    # Find the center of the octant
                    half = self.size / 2
                    child_center = self.center_position + np.array(
                        [
                            (i - 0.5) * half,
                            (j - 0.5) * half,
                            (k - 0.5) * half,
                        ]
                    )

                    # Create a child node
                    self.children[i, j, k] = Octree_Node(
                        child_center,
                        self.size / 2,
                        self.depth + 1,
                        (2 * ix + i, 2 * iy + j, 2 * iz + k),
                        child_particles,
                        self.all_particles,
                    )
        
        # Loop over the child nodes to build the octree
        for child in self.children.flatten():
            if child is not None:
                child.build_octree(max_depth)

    def get_node_at_level(self, target_level, target_index):
        node = self
        
        # Loop over the levels
        for d in range(target_level):
            if node is None or node.children is None:
                return None

            # Bitshift target index and extract the last bit
            shift = target_level - d - 1
            i = (target_index[0] >> shift) & 1
            j = (target_index[1] >> shift) & 1
            k = (target_index[2] >> shift) & 1

            # Go down into the corresponding node
            node = node.children[i, j, k]

        return node

    def fill_massmap_from_octree(
        self,
        level,
        massmap,
    ):
        """
        Fill the four requested x-slices from the octree

        Parameters
        ----------
        root : Octree_Node
            Root of the octree
        level : int
            Octree level to plot
        massmap : ndarray
            Mass map, shape (4, pixels, pixels)

        Notes
        -----
        massmap[0,:,:] corresponds to x-index 0
        massmap[1,:,:] corresponds to x-index 1
        massmap[2,:,:] corresponds to x-index 2
        massmap[3,:,:] corresponds to x-index 3
        """

        pixels = 2**level

        for offset, ix in enumerate(range(len(massmap))):
            for iy in range(pixels):
                for iz in range(pixels):
                    # Finding the node corresponding to this pixel
                    node = self.get_node_at_level(
                        target_level=level,
                        target_index=(ix, iy, iz),
                    )

                    # If the node exist set this pixel to the mass
                    if node is not None:
                        massmap[offset, iy, iz] = node.mass


def fft(array):
    # Casting the array to complex
    array = array.astype(np.complex64)

    # Length of the array and the corresponding number of bit
    N = len(array)
    N_bits = int(np.log2(N))

    # Reversing the bits to reindex the array
    idx = np.array([int(f"{i:0{N_bits}b}"[::-1], 2) for i in np.arange(N)])
    array = array[idx]

    # Performing the FFT
    N_j = 2
    two_pi_i = 2j * np.pi
    while N_j <= N:
        half = N_j // 2
        n = np.arange(0, N, N_j)
        k = np.arange(0, half)
        m = n[:, None] + k
        w = np.exp(two_pi_i * k / N_j)
        t = array[m].copy()
        v = w * array[(m + half)]
        array[m] += v
        array[(m + half)] = t - v
        N_j *= 2

    return array


def ifft(array):
    # Using the FFT to IFFT
    return fft(array.conj()).conj() / len(array)


def fft_nd(A):
    # Casting the array to complex
    A = A.astype(np.complex64)
    for axis in range(A.ndim):
        A = np.apply_along_axis(fft, axis, A)
    return A


def ifft_nd(A):
    # Casting the array to complex
    A = A.astype(np.complex64)
    for axis in range(A.ndim):
        A = np.apply_along_axis(ifft, axis, A)
    return A


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Question 2: Calculating potentials

    with h5py.File("/disks/cosmodm/DMO_a0.1_256.hdf5", "r") as handle:
        pos = handle["Position"][...]  # particle positions, shape (Np,3), comoving
        # vel=handle["Velocity"][...] #particle velocities, shape (Np,3), comoving <-- not used, but if you're interested
    # Question 2a: using Barnes-Hut [note: not actually calculating a potential, unless you do the bonus question]

    tree = Octree_Node(
        np.array([L / 2, L / 2, L / 2], dtype=np.float32),
        size=L,
        depth=0,
        index=(0, 0, 0),
        particles=np.arange(len(pos)),
        all_particles=pos,
    )
    tree.build_octree(
        max_depth=7,
    )
    # Plotting the mass distribution for a slice

    for level in [3, 5, 7]:  # feel free to change any of this code
        pixels = 2**level
        massmap = np.zeros((4, pixels, pixels), dtype=np.float32)

        tree.fill_massmap_from_octree(
            level=level,
            massmap=massmap,
        )

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        pcm = ax[0, 0].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[0, :, :]
        )
        ax[0, 0].set(ylabel="z index", title="x index = 0")
        fig.colorbar(pcm, ax=ax[0, 0], label="Total mass inside node")

        pcm = ax[0, 1].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[1, :, :]
        )
        ax[0, 1].set(title="x index = 1")
        fig.colorbar(pcm, ax=ax[0, 1], label="Total mass inside node")

        pcm = ax[1, 0].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[2, :, :]
        )
        ax[1, 0].set(ylabel="z index", xlabel="y index", title="x index = 2")
        fig.colorbar(pcm, ax=ax[1, 0], label="Total mass inside node")

        pcm = ax[1, 1].pcolormesh(
            np.arange(pixels), np.arange(pixels), massmap[3, :, :]
        )
        ax[1, 1].set(xlabel="y index", title="x index = 3")
        fig.colorbar(pcm, ax=ax[1, 1], label="Total mass inside node")

        ax[0, 0].set_aspect("equal", "box")
        ax[0, 1].set_aspect("equal", "box")
        ax[1, 0].set_aspect("equal", "box")
        ax[1, 1].set_aspect("equal", "box")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"fig2a_level{level}.png"), dpi=300)
        plt.close()

    # Question 2b: using the FFT

    # Making the density grid from the leaf nodes of the octree
    Ngrid = np.int64(128)
    densgrid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    tree.fill_massmap_from_octree(
        level=7,
        massmap=densgrid,
    )

    # FFT of the density grid
    rho_k = fft_nd(densgrid)

    # Calculate the corresponding frequencies
    freqs = 2 * np.pi * np.concatenate([np.arange(0, Ngrid // 2), 
                                        np.arange(-Ngrid // 2, 0)]) / L
    kx, ky, kz = np.meshgrid(freqs, freqs, freqs, indexing="ij")

    # Using the frequencies to rescale the FT of the density grid
    k2 = kx**2 + ky**2 + kz**2
    phi_k = rho_k / k2

    # Setting the FT of the potential to 0 for the mode where k2 == 0 as the rescaling results in this mode going to infiniti
    # Setting it to 0 can be done as the potential is defined up to a constant
    phi_k[k2 == 0] = 0

    # Finding the potential
    phi = ifft_nd(phi_k)
    potential = -G * np.abs(phi) * np.sign(phi.real) / np.pi

    # Plotting four slices of a grid
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    grid_points = np.arange(Ngrid) / Ngrid * L
    pcm = ax[0, 0].pcolormesh(grid_points, grid_points, potential[0, :, :])
    ax[0, 0].set(ylabel="z [cMpc]", title=r"Slice x$_0$")
    fig.colorbar(pcm, ax=ax[0, 0], label="Potential")
    pcm = ax[0, 1].pcolormesh(grid_points, grid_points, potential[16, :, :])
    ax[0, 1].set(title=r"Slice x$_{16}$")
    fig.colorbar(pcm, ax=ax[0, 1], label="Potential")
    pcm = ax[1, 0].pcolormesh(grid_points, grid_points, potential[32, :, :])
    ax[1, 0].set(ylabel="z [cMpc]", xlabel="y [cMpc]", title=r"Slice x$_{32}$")
    fig.colorbar(pcm, ax=ax[1, 0], label="Potential")
    pcm = ax[1, 1].pcolormesh(grid_points, grid_points, potential[64, :, :])
    ax[1, 1].set(xlabel="y [cMpc]", title=r"Slice x$_{64}$")
    fig.colorbar(pcm, ax=ax[1, 1], label="Potential")
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 1].set_aspect("equal", "box")
    ax[1, 0].set_aspect("equal", "box")
    ax[1, 1].set_aspect("equal", "box")
    plt.savefig(os.path.join(output_dir, "fig2b.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

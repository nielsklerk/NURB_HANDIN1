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

        if particles is None:
            self.mass = np.float32(0.0)
            self.CoM = center_position
        else:
            self.mass = np.float32(len(particles)) * mp
            self.CoM = np.mean(all_particles[particles], axis=0)

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
        if (
            self.particles is None
            or len(self.particles) == 0
            or self.depth >= max_depth
        ):
            return
        if len(self.particles) <= 1:
            return
        cx, cy, cz = self.center_position
        pos = self.all_particles[self.particles]
        right = pos[:, 0] >= cx
        top = pos[:, 1] >= cy
        front = pos[:, 2] >= cz
        self.children = np.empty((2, 2, 2), dtype=object)
        ix, iy, iz = self.index
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mask = (right == i) & (top == j) & (front == k)
                    child_particles = self.particles[mask]
                    child_center = self.center_position + (
                        np.array([i, j, k], dtype=np.float32) - 0.5
                    ) * (self.size / 2)
                    self.children[i, j, k] = Octree_Node(
                        child_center,
                        size=self.size / 2,
                        depth=self.depth + 1,
                        index=(2 * ix + i, 2 * iy + j, 2 * iz + k),
                        particles=child_particles,
                        all_particles=self.all_particles,
                    )
        for child in self.children.flatten():
            child.build_octree(max_depth)
        # If node contains no particles
        # mass should be zero, CoM can be set equal to center_position
        # and children should be None

        # if this is a leaf node, return the node without children

        # compute depth, index, mass, CoM, and children
        # For the bonus question, store the particle indices

        # split the node into 8 child nodes
        # Each child has half the side length of the parent
        #  compute the child index
        # If the parent index is (ix, iy, iz), then the child index is
        # (2*ix + offest_x, 2*iy + offest_y, 2*iz + offest_z)

        #  select particles that lie inside this child node

        # recursively call build on this child

    def get_node_at_level(
        self,
        target_level,
        target_index,
    ):
        """
        Traverse the octree and return a node at a given level and index

        Parameters
        ----------
        node : Octree_Node
            Current node
        target_level : int
            Level to reach
        target_index : tuple
            Index of the target node at target_level, (ix, iy, iz)

        Returns
        -------
        node : Octree_Node or None
            Node at the requested level and index
        """
        node = self

        while node.depth < target_level:
            ox = target_index[0] & 1
            oy = target_index[1] & 1
            oz = target_index[2] & 1

            if node.children is None:
                return None

            node = node.children[ox, oy, oz]

            target_index = (
                target_index[0] >> 1,
                target_index[1] >> 1,
                target_index[2] >> 1,
            )

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

        # loop over the first four x index slices
        # and fill the corresponding y,z mass maps

        for ix in range(4):
            for iy in range(pixels):
                for iz in range(pixels):

                    node = self.get_node_at_level(
                        target_level=level,
                        target_index=(ix, iy, iz),
                    )

                    if node is not None:
                        massmap[ix, iy, iz] = node.mass

        return


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    # Question 2: Calculating potentials

    with h5py.File("/disks/cosmodm/DMO_a0.1_256.hdf5", "r") as handle:
        pos = handle["Position"][...]  # particle positions, shape (Np,3), comoving
        # vel=handle["Velocity"][...] #particle velocities, shape (Np,3), comoving <-- not used, but if you're interested
    # Question 2a: using Barnes-Hut [note: not actually calculating a potential, unless you do the bonus question]

    # TO DO: build an octree
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

    for level in [1, 2, 3, 4, 5, 6, 7]:  # feel free to change any of this code
        pixels = 2**level
        massmap = np.zeros((4, pixels, pixels), dtype=np.float32)
        # TO DO: traverse the octree, fill map massmap[0,:,:] with the masses of nodes at depth 3 and x_index=x_0,
        #        massmap[1,:,:] with the masses of nodes at depth 3 and x_index=x_1, etc; then plot these slices;
        #        then do the same for levels 5 and 7

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

    Ngrid = np.int64(128)
    densgrid, edges = np.histogramdd(pos, bins=Ngrid)
    # densgrid = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    # potential = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float32)
    rho_k = np.fft.fftn(densgrid)

    kfreq = np.fft.fftfreq(Ngrid, d=L / Ngrid) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kfreq, kfreq, kfreq, indexing="ij")

    k2 = kx**2 + ky**2 + kz**2

    phi_k = rho_k / k2
    phi_k[k2 == 0] = 0

    potential = -G * np.fft.ifftn(phi_k).real / np.pi
    # TO DO: assign particle masses to densgrid, convert to density, and calculate potentials from it

    # Plotting four slices of a grid

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))
    pcm = ax[0, 0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[0, :, :])
    # ax[0,0].set(ylabel='...', title='...')
    fig.colorbar(pcm, ax=ax[0, 0], label="Potential")
    pcm = ax[0, 1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[16, :, :])
    # ax[0,1].set(title='...')
    fig.colorbar(pcm, ax=ax[0, 1], label="Potential")
    pcm = ax[1, 0].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[32, :, :])
    # ax[1,0].set(ylabel='...', xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1, 0], label="Potential")
    pcm = ax[1, 1].pcolormesh(np.arange(Ngrid), np.arange(Ngrid), potential[64, :, :])
    # ax[1,1].set(xlabel='...', title='...')
    fig.colorbar(pcm, ax=ax[1, 1], label="Potential")
    ax[0, 0].set_aspect("equal", "box")
    ax[0, 1].set_aspect("equal", "box")
    ax[1, 0].set_aspect("equal", "box")
    ax[1, 1].set_aspect("equal", "box")
    plt.savefig(os.path.join(output_dir, "fig2b.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    main()

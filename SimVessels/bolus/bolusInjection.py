import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Gauss:
    """Gauss with values (not area!) normalized to 1"""
    def __init__(self, mu=0., sigma=1.):
        self.mu = mu
        self.sigma = sigma

    def sample(self, x):
        return np.exp(-0.5*((x-self.mu)/self.sigma)**2)


class VesselNode:
    """ The Class for a vessel node. Each vessel will have x, y, and z position,
    as well as a diameter and a link to the next vessel node(s)."""
    def __init__(self, x=0., y=0., z=0., d=0., next=[]):
        self.x = x
        self.y = y
        self.z = z
        self.d = d
        self.next = next


def coordinates2vessel(coords, interpolate=0):
    """This function takes an array of coordinates in the format
            [[x1, y1, z1, d1],
             [x2, y2, z2, d2],
             [x3, y3, z3, d3],
             [nan, nan, nan, nan],
             [x4, y4, z4, d4],
             [x5, y5, z5, d5]]
    as input. Each strand is separated by a row of nans. We first get the separate
    the individual strands. Then strand2vessel() is used to connect the coordinates
    of one strand to a linked list using the VesselNode class. For the last point
    we look if there are strands beginning with the same coordinate as the current
    strand ends. If so, we found a branch point and need to connect the last point
    of this strand with the first point of the next strand.
    """
    def strand2vessel(strand):
        # Start with first node and connect each node
        head = VesselNode()
        node = head
        i = 0
        while i < strand.shape[1]:
            nextN = VesselNode(x=strand[0, i], y=strand[1, i], z=strand[2, i], d=strand[3, i])
            node.next = [nextN]
            node = node.next[0]
            i += 1
        # We reached the end of this strand. See if there are any strands beginning
        # at this coordinate. If so, call them recursively to form the complete
        # vessel tree.
        child_strands = [s for j, s in enumerate(strands) if np.isclose(s[:, 0], strand[:, -1]).all()]
        node.next = [strand2vessel(s) for s in child_strands]
        return head.next[0]

    # there are several gaps in generated vessels if use interpolate, something is wrong,debug needed
    def interpolate_strands(strands, factor=2):
        # This function gets list of strands and interpolate with given factor
        # (factor must be even).
        strands = strands.copy()
        for s in range(len(strands)):
            strand = strands[s]
            for _ in range(0, factor, 2):
                strand_interp = np.zeros((strand.shape[0], strand.shape[1]*2-1))
                for i in range(strand.shape[1]):
                    strand_interp[:, i*2] = strand[:, i]
                    if i+1 < strand.shape[1]:
                        interp = (strand[:, i] + strand[:, i+1]) / 2
                        # interp[-1] = strand[-1, i + 1]  # diameter
                        strand_interp[:, i*2+1] = interp

                strand = strand_interp
            strands[s] = strand
        return strands

    # Get individual strands. Generates a list of coordinate chunks separated by
    # nan-rows in the original coordinate array.
    strands = []
    i = j = 0
    while i < coords.shape[1]:
        if j >= coords.shape[1] or np.isnan(coords[0, j]):
            strands.append(coords[:, i:j])
            i=j+1
        j += 1

    # Interpolate with factor interpolate=0,2,4,...
    if interpolate:
        strands = interpolate_strands(strands, factor=interpolate)

    # Generate vessel tree
    return strand2vessel(strands[0])


def compute_distances(head, add_gaussian=False, v=0.5, sigma=1.):
    """Take head as input and compute distance to the origin using iterative
    Depth-First-Search. If desired, we can also add a gaussian curve using the
    parameters provided """
    max_dist = 0
    stack = [(0, head)]
    while stack:
        dist, node = stack.pop()
        if node:
            node.dist = dist
            if add_gaussian:
                # The gaussian has mean dist/v. This means we reach the center of
                # the distribution after t = dx/v time steps.
                node.gauss = Gauss(mu=dist/v, sigma=sigma)
            max_dist = max(max_dist, dist)
            for child in node.next:
                # Complete euclidian distance between this node and child node and
                # add it to the total distance
                add_dist = ((node.x - child.x)**2 + (node.y - child.y)**2 + (node.z - child.z)**2)**0.5
                stack.append((dist+add_dist, child))

    # Return max_dist to normalize distances to one for plotting
    return head, max_dist


def plot_vessel(head, max_dist=None, time_step=None, title=''):
    """Plotting function. Can be used to plot distances using
        plot_vessel(head, max_dist=max_dist, title='Distances')
    or bolus injection using
        plot_vessel(head, time_step=10, title='Bolus')
    """
    # Setup plot and colorbar
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)

    cax = fig.add_axes([0.15, 0.25, 0.02, 0.5])
    ax.view_init(azim=-139, elev=-145)
    sm = cm.ScalarMappable(cmap=cm.Blues)
    fig.colorbar(sm, cax=cax)
    cax.yaxis.set_ticks_position('left')

    # Plot the vessel
    stack = [head]
    while stack:
        head = stack.pop()
        if head:
            for child in head.next:
                if time_step != None:
                    # Plot bolus injection given a time step
                    c = head.gauss.sample(time_step)
                    ax.plot([head.x, child.x], [head.y, child.y], [head.z, child.z],
                            linewidth=0.5 * head.d, c=cm.Blues(c))
                elif max_dist:
                    # Plot distances and normalize using max_dist
                    ax.plot([head.x, child.x], [head.y, child.y], [head.z, child.z],
                            linewidth=0.5 * head.d, c=cm.Blues(head.dist/max_dist))
                else:
                    # Plot Vessel tree in uniform color
                    ax.plot([head.x, child.x], [head.y, child.y], [head.z, child.z],
                            linewidth=0.5 * head.d, color='Blue')
                stack.append(child)
    return fig


def print_vessel(head):
    """Convenience function that does iterative DFS on the vessel and prints out
    the individual coordinates"""
    stack = [(0, head)]
    while stack:
        strand, node = stack.pop()
        if node:
            print(f'{strand} | x: {node.x}, y:{node.y}, z:{node.z}, d:{node.d}, distance:{node.dist}')
            stack.extend([(strand+i, c) for i, c in enumerate(node.next)])

def coordinates_back(head, t):
    """Convenience function that does iterative DFS on the vessel, compute the intensity and then returns
    the individual coordinates
            [[x1, y1, z1, d1, I1],
             [x2, y2, z2, d2, I2],
             [x3, y3, z3, d3, I3],
             [nan, nan, nan, nan, nan],
             [x4, y4, z4, d4, I4],
             [x5, y5, z5, d5, I5]]
    """
    stack = [(0, head)]
    strand_last = 0
    updaten = []
    while stack:
        strand, node = stack.pop()
        if node:
            # [nan, nan, nan, nan, nan] as the end for every strand
            if strand != strand_last:
                # updaten = np.vstack((updaten, [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]))
                updaten = np.c_[updaten, [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]]
            if len(updaten) == 0:
                updaten = [node.x, node.y, node.z, node.d, head.gauss.sample(t)]
            else:
                # updaten = np.vstack((updaten, np.array([node.x, node.y, node.z, node.d, node.gauss.sample(t)]).T))
                updaten = np.c_[updaten, [node.x, node.y, node.z, node.d, node.gauss.sample(t)]]
            strand_last = strand
            stack.extend([(strand+i, c) for i, c in enumerate(node.next)])
    return updaten

def bolus_injection(coords, v,t,sigma = 2.5, interp_coords_factor = 0):
    """Input
            [[x1, y1, z1, d1],
             [x2, y2, z2, d2],
             [x3, y3, z3, d3],
             [nan, nan, nan, nan],
             [x4, y4, z4, d4],
             [x5, y5, z5, d5]]
        Return
              [[x1, y1, z1, d1, I1],
             [x2, y2, z2, d2, I2],
             [x3, y3, z3, d3, I3],
             [nan, nan, nan, nan, nan],
             [x4, y4, z4, d4, I4],
             [x5, y5, z5, d5, I5]]
        x,y,z the coordinates, d is the diameter, I is the intensity for the point (x,y,z).
    """
    # Get vessel tree and distances and add gaussian curves to each node.
    head = coordinates2vessel(coords, interpolate=interp_coords_factor)
    head, max_dist = compute_distances(head, add_gaussian=True, v=v, sigma=sigma)
    coords_new = coordinates_back(head, t)

    return coords_new, max_dist

def main():
    # Parameters
    sigma = 2.5
    v = 10
    t_start = 0
    t_end = 30
    interp_coords_factor = 2

    # Load vessel coordinates
    coords = np.loadtxt("D:\\vessels\\update_save\\update_d10_dr12_epsilon4_iter6.txt")

    # Get vessel tree and distances and add gaussian curves to each node.
    head = coordinates2vessel(coords, interpolate=interp_coords_factor)
    head, max_dist = compute_distances(head, add_gaussian=True, v=v, sigma=sigma)

    # Plot over time
    for t in range(t_start, t_end):
        print(f'Time step {t} of {t_end-t_start}')
        fig = plot_vessel(head, time_step=t, title=f'$v={{{v}}}$, $\sigma={{{sigma}}}$, $t_0={{{t_start}}}$')
        plt.pause(1e-5)
        plt.close()

    # test Bolus_injection
    # sigma = 2.5
    # v = 10
    # interp_coords_factor = 2
    # t = 0
    # # Load vessel coordinates
    # coords = np.loadtxt("D:\\vessels\\update_save\\update_d10_dr12_epsilon4_iter6.txt")
    # updaten = bolus_injection(coords, v, t, sigma=sigma, interp_coords_factor=interp_coords_factor)


if __name__ == "__main__":
    main()



import pyvista as pv
import trimesh
from hyputils import klein_to_poincare
from meshhacks import mungmesh
from argparse import ArgumentParser
import numpy as np
import sys

tetrahedron_vertices = np.array([
    [ 0.57735027,  0.57735027,  0.57735027],
    [ 0.57735027, -0.57735027, -0.57735027],
    [-0.57735027,  0.57735027, -0.57735027],
    [-0.57735027, -0.57735027,  0.57735027]
])

cube_vertices = np.array([
    [ 0.57735027,  0.57735027,  0.57735027],
    [-0.57735027,  0.57735027,  0.57735027],
    [ 0.57735027, -0.57735027,  0.57735027],
    [-0.57735027, -0.57735027,  0.57735027],
    [ 0.57735027,  0.57735027, -0.57735027],
    [-0.57735027,  0.57735027, -0.57735027],
    [ 0.57735027, -0.57735027, -0.57735027],
    [-0.57735027, -0.57735027, -0.57735027]
])

octahedron_vertices = np.array([
    [ 1.,  0.,  0.],
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.]
])

dodecahedron_vertices = np.array([
    [ 0.        ,  0.35682209,  0.93417236],
    [ 0.        , -0.35682209,  0.93417236],
    [ 0.        ,  0.35682209, -0.93417236],
    [ 0.        , -0.35682209, -0.93417236],
    [ 0.35682209,  0.93417236,  0.        ],
    [-0.35682209,  0.93417236,  0.        ],
    [ 0.35682209, -0.93417236,  0.        ],
    [-0.35682209, -0.93417236,  0.        ],
    [ 0.93417236,  0.        ,  0.35682209],
    [-0.93417236,  0.        ,  0.35682209],
    [ 0.93417236,  0.        , -0.35682209],
    [-0.93417236,  0.        , -0.35682209]
])

icosahedron_vertices = np.array([
    [ 0.        ,  0.35682209,  0.93417236],
    [ 0.        , -0.35682209,  0.93417236],
    [ 0.        ,  0.35682209, -0.93417236],
    [ 0.        , -0.35682209, -0.93417236],
    [ 0.35682209,  0.93417236,  0.        ],
    [-0.35682209,  0.93417236,  0.        ],
    [ 0.35682209, -0.93417236,  0.        ],
    [-0.35682209, -0.93417236,  0.        ],
    [ 0.93417236,  0.        ,  0.35682209],
    [-0.93417236,  0.        ,  0.35682209],
    [ 0.93417236,  0.        , -0.35682209],
    [-0.93417236,  0.        , -0.35682209]
])



polydict = {'tetrahedron': trimesh.convex.convex_hull(tetrahedron_vertices),
            'cube': trimesh.convex.convex_hull(cube_vertices),
            'octahedron': trimesh.convex.convex_hull(octahedron_vertices),
            'dodecahedron': trimesh.convex.convex_hull(dodecahedron_vertices),
            'icosahedron': trimesh.convex.convex_hull(icosahedron_vertices),
            # 'cube': trimesh.creation.box(),
            # 'icosahedron': trimesh.creation.icosahedron(),
            # 'octahedron': trimesh.creation.octahedron(),
            #'dodecahedron': mesh.generate_dodecahedron(radius=1.0, center=[0,0,0]),
}


argument_parser = ArgumentParser()
argument_parser.add_argument('polyhedron', type=str, choices=polydict.keys())
argument_parser.add_argument('--outfile', type=str)
argument_parser.add_argument('--show_edges', action='store_true')
argument_parser.add_argument('--scale', type=float, default=1)
argument_parser.add_argument('--center', type=float, nargs=3, default=None)
argument_parser.add_argument('--rotate', type=float, nargs=3, default=None)
argument_parser.add_argument('--translate', type=float, nargs=3, default=None)

args = argument_parser.parse_args()

poly = polydict[args.polyhedron]

def main():
    mesh = mungmesh(poly, klein_to_poincare)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, opacity=0.5, specular = 0.5, specular_power=10, show_edges=args.show_edges)
    #plotter.open_movie(args.outfile)

    plotter.open_gif(args.outfile, fps = 3)
    # Rotate the object and capture frames
    for angle in range(0,360,8):
        plotter.camera.azimuth = angle
        plotter.write_frame()

    # Close the plotter
    plotter.close()



if __name__ == '__main__':
    main()

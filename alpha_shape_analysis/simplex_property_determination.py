import numpy as np
from scipy.spatial import distance

def cayley_menger_analysis(vertices, d = 2):
    """
    Determines volume and circumradius for a tetrahedron given the vertices
    https://westy31.home.xs4all.nl/Circumsphere/ncircumsphere.htm#Coxeter
    """
    if d == 3:
        cm_matrix = np.array([[0, 1, 1, 1, 1], [1, 0, distance.sqeuclidean(vertices[0], vertices[1]),
                            distance.sqeuclidean(vertices[0], vertices[2]),
                            distance.sqeuclidean(vertices[0], vertices[3])],
                            [1, distance.sqeuclidean(vertices[1], vertices[0]), 0,
                            distance.sqeuclidean(vertices[1], vertices[2]),
                            distance.sqeuclidean(vertices[1], vertices[3])],
                            [1, distance.sqeuclidean(vertices[2], vertices[0]),
                            distance.sqeuclidean(vertices[2], vertices[1]), 0,
                            distance.sqeuclidean(vertices[2], vertices[3])],
                            [1, distance.sqeuclidean(vertices[3], vertices[0]),
                            distance.sqeuclidean(vertices[3], vertices[1]),
                            distance.sqeuclidean(vertices[3], vertices[2]), 0]])

        cm_inv = np.linalg.inv(cm_matrix)
        centroid = [(vertices[0, 0] + vertices[1, 0] + vertices[2, 0] + vertices[3, 0])/4, \
                    (vertices[0, 1] + vertices[1, 1] + vertices[2, 1] + vertices[3, 1])/4, \
                    (vertices[0, 2] + vertices[1, 2] + vertices[2, 2] + vertices[3, 2])/4]
        v_simplex = np.sqrt(np.divide(np.linalg.det(cm_matrix), 288))
        r_simplex = np.sqrt(cm_inv[0,0]/-2)

    if d == 2:
        cm_matrix = np.array([[0, 1, 1, 1], [1, 0, distance.sqeuclidean(vertices[0], vertices[1]),
                            distance.sqeuclidean(vertices[0], vertices[2])],
                            [1, distance.sqeuclidean(vertices[1], vertices[0]), 0,
                            distance.sqeuclidean(vertices[1], vertices[2])],
                            [1, distance.sqeuclidean(vertices[2], vertices[0]),
                            distance.sqeuclidean(vertices[2], vertices[1]), 0]])

        cm_inv = np.linalg.inv(cm_matrix)
        centroid = [(vertices[0, 0] + vertices[1, 0] + vertices[2, 0])/3, (vertices[0, 1] + vertices[1, 1] + vertices[2, 1])/3]
        v_simplex = np.sqrt(np.divide(np.linalg.det(cm_matrix), -16))
        r_simplex = np.sqrt(cm_inv[0,0]/-2)

    return v_simplex, r_simplex, cm_inv[1::], centroid

def get_simplex_properties(tri, d = 2):
    
    if d == 2:
        area = []
        circumradius = []
        circumcenter = []
        centroid = []
        for i in tri.simplices:
            a, r, center, centr = cayley_menger_analysis(tri.points[i], d = d)
            area.append(a)
            circumradius.append(r)
            circumcenter.append(center)
            centroid.append(centr)
        
        return np.array(area), np.array(circumradius), np.array(circumcenter), np.array(centroid)

    if d == 3:
        volume = []
        circumradius = []
        circumcenter = []
        centroid = []
        for i in tri.simplices:
            a, r, center, centr = cayley_menger_analysis(tri.points[i], d = d)
            volume.append(a)
            circumradius.append(r)
            circumcenter.append(center)
            centroid.append(centr)
        
        return np.array(volume), np.array(circumradius), np.array(circumcenter), np.array(centroid)

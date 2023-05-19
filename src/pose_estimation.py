import numpy as np


class pinhole_cam:
    def __init__(self, resolution, intrinsic_matrix):
        self.resolution = np.array(resolution)

        self.f = intrinsic_matrix[0, 0] / self.resolution[0]

        # test_point = np.transpose(np.array([[1, 1, self.f, 1]]))
        # coords = intrinsic_matrix @ test_point
        # coords = self.normalize_coords(coords)
        # self.size = coords[0:2, 0] / self.resolution

        self.center = intrinsic_matrix[0:2, 2]

    def normalize_coords(self, coords):
        return coords / coords[-1, -1]

    # def __init__(self, f, resolution, size, center=None):
    #     self.f = f
    #     self.resolution = np.array(resolution)

    #     if size is tuple: # If given directly as tuple, store it
    #         self.size = size
    #     else: # if given as a scaling factor, just multiply by resolution
    #         self.size = size * self.resolution

    #     if center is None:
    #         self.center = np.multiply([0.5,0.5], self.resolution)

    def model(self, u, v, Z):
        du = u - self.center[0]
        dv = v - self.center[1]

        X = (Z / self.f) * du
        Y = (Z / self.f) * dv

        return (X, Y)


if __name__ == "__main__":
    res = [1280, 720]
    f = 500
    center = np.array(res) / 2

    intrinsics = np.array(
        [[f * res[0], 0, center[0], 0], [0, f * res[1], center[1], 0], [0, 0, 1, 0]]
    )

    test_point = [0,0,5]
    test_point.append(1)
    test_point_mat = np.transpose(np.array([test_point]))
    out = intrinsics @ test_point_mat
    out = out / out[-1,-1]

    print("test_point:\n",out)

    a = pinhole_cam(res, intrinsics)

    print("X,Y:", a.model(1280, 0, 2))

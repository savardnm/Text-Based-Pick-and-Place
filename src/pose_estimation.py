import numpy as np
from scipy.spatial.transform import Rotation


class PinholeCam:
    def __init__(self, resolution, camera_matrix, Z=108):
        self.resolution = np.array(resolution)

        self.f = camera_matrix[0, 0] / self.resolution[0]

        # test_point = np.transpose(np.array([[1, 1, self.f, 1]]))
        # coords = intrinsic_matrix @ test_point
        # coords = self.normalize_coords(coords)
        # self.size = coords[0:2, 0] / self.resolution

        self.center = 0.5 * self.resolution

        self.Z = Z

    def normalize_coords(self, coords):
        return coords / coords[-1, -1]

    def model(self, u, v):
        du = u - self.center[0]
        dv = v - self.center[1]

        X = (self.Z / self.f) * du
        Y = (self.Z / self.f) * dv

        return (X, Y)

    def construct_transform(self, pose, rotation, degrees=True):
        T = np.zeros((4, 4))
        T[3, 3] = 1

        T[0:3, 3] = pose

        R = Rotation.from_euler("z", rotation, degrees=degrees)

        T[0:3, 0:3] = R.as_matrix()

        return T


if __name__ == "__main__":
    res = [1280, 720]
    f = 500
    center = np.array(res) / 2

    intrinsics = np.array(
        [[f * res[0], 0, center[0], 0], [0, f * res[1], center[1], 0], [0, 0, 1, 0]]
    )

    test_point = [1, 2, 5]
    test_point.append(1)
    test_point_mat = np.transpose(np.array([test_point]))
    out = intrinsics @ test_point_mat
    out = out / out[-1, -1]

    print("test_point:\n", out)

    a = pinhole_cam(res, intrinsics)

    X, Y = a.model(1280, 0, 2)
    Z = 108 # TODO Fix units to the ones in ROBWORK (This is in cm right now)

    print("X,Y:", X, Y)

    print(a.construct_transform(np.array([X, Y, Z]), 45))

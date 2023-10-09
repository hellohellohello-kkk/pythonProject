import numpy as np


def normalized_to_image_coordinates(normalized_image_points, focal_length,
                                    pixel_pitch, image_center):
    x = normalized_image_points[0]
    y = normalized_image_points[1]
    xc = image_center[0]
    yc = image_center[1]

    X = x * focal_length / pixel_pitch + xc
    Y = y * focal_length / pixel_pitch + yc
    return np.array([X, Y])


def compose_rotation_matrix(angle_x, angle_y, angle_z):
    """
    Rx * Ry * Rz の回転行列を計算します。

    Args:
        angle_x (float): X軸周りの回転角度（deg）
        angle_y (float): Y軸周りの回転角度（deg）
        angle_z (float): Z軸周りの回転角度（deg）

    Returns:
        np.ndarray: 合成された回転行列
    """
    cos_x = np.cos(np.deg2rad(angle_x))
    sin_x = np.sin(np.deg2rad(angle_x))
    Rx = np.array([
        [1, 0, 0, 0],
        [0, cos_x, -sin_x, 0],
        [0, sin_x, cos_x, 0],
        [0, 0, 0, 1]
    ])
    cos_y = np.cos(np.deg2rad(angle_y))
    sin_y = np.sin(np.deg2rad(angle_y))
    Ry = np.array([
        [cos_y, 0, sin_y, 0],
        [0, 1, 0, 0],
        [-sin_y, 0, cos_y, 0],
        [0, 0, 0, 1]
    ])
    cos_z = np.cos(np.deg2rad(angle_z))
    sin_z = np.sin(np.deg2rad(angle_z))
    Rz = np.array([
        [cos_z, -sin_z, 0, 0],
        [sin_z, cos_z, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    # 合成して回転行列を計算
    composed_matrix = np.dot(Rz, np.dot(Ry, Rx))
    return composed_matrix


point_3d_homogeneous = np.array([100, 0, 0, 1])

rotation_matrix = compose_rotation_matrix(0, 1, 0)

# 平行移動ベクトルを定義します。
translation_vector = np.array([0, 0, 2000, 0])

# 同次変換行列を作成します。
transformation_matrix = rotation_matrix
transformation_matrix[:3, 3] = translation_vector[:3]  # 平行移動ベクトルを設定

print(transformation_matrix)
transformed_point_homogeneous = np.dot(transformation_matrix,
                                       point_3d_homogeneous)

X = transformed_point_homogeneous[0]
Y = transformed_point_homogeneous[1]
Z = transformed_point_homogeneous[2]
normalized_image_points = np.array([X / Z, Y / Z])
image_points = normalized_to_image_coordinates(normalized_image_points, 49,
                                               0.00345, [1024, 1024])

# 結果を出力します。
print("3D座標 (同次座標):", point_3d_homogeneous)
print("回転と平行移動後の3D座標 (同次座標):", transformed_point_homogeneous)
print("回転と平行移動後の正規化画像座標:", normalized_image_points)
print("回転と平行移動後の画像座標:", image_points)

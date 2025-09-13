import os
import colmap_utils


def txt_to_bin(input_path, output_path):
    # Convert cameras.txt to cameras.bin
    # cameras_txt = os.path.join(input_path, "cameras.txt")
    # cameras_bin = os.path.join(output_path, "cameras.bin")
    # colmap_utils.write_cameras_bin(cameras_txt, cameras_bin)

    # Convert images.txt to images.bin
    images_txt = os.path.join(input_path, "images.txt")
    images_bin = os.path.join(output_path, "images.bin")
    colmap_utils.write_images_bin(images_txt, images_bin)

    # Convert points3D.txt to points3D.bin if the file exists
    # points3D_txt = os.path.join(input_path, "points3D.txt")
    # points3D_bin = os.path.join(output_path, "points3D.bin")
    # if os.path.exists(points3D_txt):
    #     colmap_utils.write_points3d_bin(points3D_txt, points3D_bin)


if __name__ == "__main__":
    input_path = "/data/vdb/zhongyuhe/workshop/SuGaR/data_all/data_4/sparse/0"
    output_path = "/data/vdb/zhongyuhe/workshop/SuGaR/data_all/data_4/sparse/0"
    txt_to_bin(input_path, output_path)

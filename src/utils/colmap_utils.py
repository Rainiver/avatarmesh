import struct


def write_cameras_bin(cameras_txt, cameras_bin):
    MODEL_PARAMS_LENGTH = {
        "SIMPLE_PINHOLE": 3,
        "PINHOLE": 4,
        "SIMPLE_RADIAL": 3,
        "RADIAL": 4,
        "OPENCV": 8,
        "OPENCV_FISHEYE": 8,
        "FULL_OPENCV": 12,
        "FOV": 5,
        "THIN_PRISM_FOV": 12,
        "RADIAL_FOV": 8
    }

    with open(cameras_txt, "r") as txt_file, open(cameras_bin, "wb") as bin_file:
        num_cameras = 0
        cameras = []
        for line in txt_file:
            if not line.strip() or line.startswith("#"):
                continue
            elements = line.strip().split()
            cam_id = int(elements[0])
            model = elements[1]
            width = int(elements[2])
            height = int(elements[3])
            param_count = MODEL_PARAMS_LENGTH.get(model, 0)
            if param_count == 0:
                raise ValueError(f"Unknown or unsupported camera model: {model}")

            params = list(map(float, elements[4:4 + param_count]))
            if len(params) != param_count:
                raise ValueError(f"Expected {param_count} parameters for model {model}, got {len(params)}")

            cameras.append((cam_id, model, width, height, params))
            num_cameras += 1

        bin_file.write(struct.pack("<I", num_cameras))
        for cam_id, model, width, height, params in cameras:
            bin_file.write(struct.pack("<I", cam_id))
            bin_file.write(struct.pack("<I", len(model)))
            bin_file.write(model.encode('utf-8'))
            bin_file.write(struct.pack("<II", width, height))
            binary_params_format = "<" + "d" * len(params)
            bin_file.write(struct.pack(binary_params_format, *params))


def write_images_bin(images_txt, images_bin):
    with open(images_txt, "r") as txt_file, open(images_bin, "wb") as bin_file:
        num_images = 0
        images = []
        for line in txt_file:
            if not line.strip() or line.startswith("#"):
                continue
            elements = line.strip().split()
            img_id = int(elements[0])
            try:
                qw, qx, qy, qz = map(float, elements[1:5])
                tx, ty, tz = map(float, elements[5:8])
                cam_id = int(elements[8])
                name = elements[9]
            except ValueError as ve:
                print(f"Error parsing line: {line}")
                raise ve
            images.append((img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name))
            num_images += 1
        bin_file.write(struct.pack("<I", num_images))
        for img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, name in images:
            try:
                bin_file.write(struct.pack("<I", img_id))
                bin_file.write(struct.pack("<4d", qw, qx, qy, qz))  # 4 doubles
                bin_file.write(struct.pack("<3d", tx, ty, tz))      # 3 doubles
                bin_file.write(struct.pack("<I", cam_id))
                bin_file.write(struct.pack("<I", len(name)))
                bin_file.write(name.encode('utf-8'))
            except struct.error as e:
                print(f"Error packing parameters for image_id {img_id}: qw, qx, qy, qz = {qw}, {qx}, {qy}, {qz}, tx, ty, tz = {tx}, {ty}, {tz}")
                raise e


def write_points3d_bin(points3D_txt, points3D_bin):
    with open(points3D_txt, "r") as txt_file, open(points3D_bin, "wb") as bin_file:
        num_points = 0
        points = []
        for line in txt_file:
            if not line.strip() or line.startswith('#'):
                continue
            elements = line.strip().split()
            pt_id = int(elements[0])
            x, y, z = map(float, elements[1:4])
            r, g, b = map(int, elements[4:7])
            error = float(elements[7])
            track_info = list(map(int, elements[8:]))
            points.append((pt_id, x, y, z, r, g, b, error, track_info))
            num_points += 1

        bin_file.write(struct.pack("<I", num_points))
        for pt_id, x, y, z, r, g, b, error, track_info in points:
            bin_file.write(struct.pack("<Q", pt_id))
            bin_file.write(struct.pack("<3d", x, y, z))  # 3 doubles
            bin_file.write(struct.pack("<3B", r, g, b))  # 3 bytes
            bin_file.write(struct.pack("<d", error))
            bin_file.write(struct.pack("<Q", len(track_info) // 2))
            bin_file.write(struct.pack("<I" * len(track_info), *track_info))

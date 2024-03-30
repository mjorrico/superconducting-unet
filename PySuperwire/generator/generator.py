import warnings, pkg_resources, imageio, hashlib, pickle, cv2, os
from copy import deepcopy
from .poly_generator import generate as pg
from tqdm import tqdm

import concurrent.futures
import numpy as np


class WireGenerator:
    def __init__(self):
        self.void_dict = self._read_void()

        self.img_size = 1000
        self.variance = 0.4
        self.largevoid_count = 200
        self.smallvoid_count = 300
        self.smallvoid_sizes = [15, 10, 7]
        self.smallvoid_weight = [1, 2, 2]
        self.zmat = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        x = np.arange(self.img_size, step=1)
        y = np.arange(self.img_size, step=1)
        c = (self.img_size - 1) / 2
        self.gridx, self.gridy = np.meshgrid(x, y)
        self.dist_array = np.sqrt((self.gridx - c) ** 2 + (self.gridy - c) ** 2)

    def _read_void(self):
        sizes = [7, 10, 15, 20]
        files = [f"void_{str(s).zfill(2)}.pickle" for s in sizes]

        outputdict = {}
        for file in files:
            with pkg_resources.resource_stream(__name__, f"void/{file}") as f:
                arr = pickle.load(f)
                size = np.shape(arr)[1]
                outputdict[size] = arr

        return outputdict

    def _random__hash(self, len):
        random_data = os.urandom(16)
        hash_object = hashlib.sha256()
        hash_object.update(random_data)
        random_hash = hash_object.hexdigest()
        return random_hash[:len]

    def get_animation(self, save_path, file_name):
        img, msk = self.superwire_gen()
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
        animation_name = os.path.join(save_path, f"{file_name}.gif")
        imageio.mimsave(animation_name, [img, msk], fps=0.5, format="GIF", loop=0)

    def _image_saver(self, save_path, file_prefix, img_idx):
        img_idx = str(img_idx).zfill(5)
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
        img_path = os.path.join(save_path, "img", f"{file_prefix}-{img_idx}.jpg")
        msk_path = os.path.join(save_path, "mask", f"{file_prefix}-{img_idx}.png")
        img, msk = self.superwire_gen()
        status_img = cv2.imwrite(img_path, img)
        status_msk = cv2.imwrite(msk_path, msk)

        if img_idx % 1000 == 999:
            print(f"Image {img_idx} saved!")

    def _check_copper_size(self, copper_mask):
        fillrate = np.sum(copper_mask > 0)
        if fillrate < 0.01:
            raise ValueError("Cable is too small. Increase wire radius.")
        elif fillrate < 0.05:
            warnings.warn("Cable fillrate is lower than 5%.", UserWarning)

    def _spread_void(self, void_group, n_void, copper_mask, void_mask, n_trial=300):
        void_count, void_size, _ = np.shape(void_group)

        for _ in range(n_void):
            is_succeed = False
            for _ in range(n_trial):  # multiple attempts to insert void
                xpos, ypos = np.random.randint(0, self.img_size - void_size, 2)
                xcheck = xpos + int(void_size / 2)
                ycheck = ypos + int(void_size / 2)

                if copper_mask[xcheck, ycheck] > 0:
                    is_succeed = True
                    break

            if not is_succeed:
                break

            chosen_void = deepcopy(void_group[np.random.randint(0, void_count)])
            x1, y1 = xpos, ypos
            x2, y2 = xpos + void_size, ypos + void_size
            cv2.bitwise_and(chosen_void, copper_mask[x1:x2, y1:y2], chosen_void)
            cv2.bitwise_or(chosen_void, void_mask[x1:x2, y1:y2], chosen_void)
            void_mask[x1:x2, y1:y2] = chosen_void

    def _get_cowskin(self, color, color_range):
        blur = np.random.uniform(10, 50)
        skin = np.random.uniform(0, 255, (1000, 1000))
        cv2.GaussianBlur(skin, (0, 0), blur, skin, blur)
        minval, maxval = np.min(skin), np.max(skin)
        rangeval = maxval - minval
        skin = (skin - minval) * (color_range / rangeval)
        skin = skin - np.mean(skin) + color
        skin = np.clip(skin, 0, 255)

        return skin.astype(np.uint8)

    def apply_cowmask(self, image, mask, color, color_range):
        skin = self._get_cowskin(color, color_range)
        image = np.where(mask > 0, skin, image)
        return image

    def _interference_base(self):
        u, v = np.random.uniform(0.2, 0.8, 2)
        a = np.random.uniform(0.01, 0.012)
        c = np.random.uniform(100, 250)
        f = lambda x: np.sin(u * x) * np.cos(v * x)
        s = lambda x: np.clip(np.sin(a * (x - c)), 0, None)
        return f(self.dist_array) * s(self.dist_array)

    def _interference_mask(self):
        c = (self.img_size - 1) / 2
        angles = np.arctan2(
            self.gridx.astype(np.float32) - c, self.gridy.astype(np.float32) - c
        )
        u, v = np.random.uniform(4, 11, 2)
        f = lambda x: (np.sin(u * x) * np.cos(v * x) + 1) / 2
        return f(angles)

    def interference(self, img):
        amplitude = np.random.uniform(10, 30)
        intmask = self._interference_base() * self._interference_mask() * amplitude
        img = np.clip(img + intmask, 0, 255)
        return img.astype(np.uint8)

    def superwire_gen(self):
        # MASKS GENERATION
        radius = np.random.uniform(400, 450)
        element_size = (radius - 400) * 6 / 50 + 27 + np.random.uniform(0, 6)
        coat_in, coat_out, hex_in, hex_out = pg(
            radius, element_size, self.variance, self.img_size
        )
        coat_mask = cv2.fillPoly(deepcopy(self.zmat), coat_out + coat_in, 255)
        elem_mask = cv2.fillPoly(deepcopy(self.zmat), hex_out + hex_in, 255)
        copp_mask = cv2.fillPoly(deepcopy(self.zmat), coat_in + hex_out + hex_in, 255)
        copp_mask_in = cv2.fillPoly(deepcopy(self.zmat), hex_in, 255)
        void_mask = deepcopy(self.zmat)

        # ERROR CHECKING
        self._check_copper_size(copp_mask)

        # GENERATING VOIDS
        s = np.sum(self.smallvoid_weight)
        counts = np.array(self.smallvoid_weight) / s * self.smallvoid_count
        counts = counts.astype(np.int32)
        for size, vcount in zip(self.smallvoid_sizes, counts):
            void_group = self.void_dict[size]
            self._spread_void(void_group, vcount, copp_mask, void_mask)
        self._spread_void(
            self.void_dict[20], self.largevoid_count, copp_mask_in, void_mask
        )

        # PREPARING MASKS
        rotation_angle = np.random.uniform(0, 360)
        c = int(self.img_size / 2)
        t_mat = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
        t_flag = cv2.INTER_NEAREST

        cv2.warpAffine(coat_mask, t_mat, np.shape(self.zmat), coat_mask, t_flag)
        cv2.warpAffine(elem_mask, t_mat, np.shape(self.zmat), elem_mask, t_flag)
        cv2.warpAffine(copp_mask, t_mat, np.shape(self.zmat), copp_mask, t_flag)
        cv2.warpAffine(copp_mask_in, t_mat, np.shape(self.zmat), copp_mask_in, t_flag)
        cv2.warpAffine(void_mask, t_mat, np.shape(self.zmat), void_mask, t_flag)

        kerneld = np.ones((4, 4), np.uint8)
        pseudoelem_mask = cv2.dilate(void_mask, kerneld, cv2.BORDER_REFLECT)
        pseudoelem_mask = np.subtract(pseudoelem_mask, void_mask)
        pseudoelem_mask = cv2.bitwise_and(pseudoelem_mask, copp_mask_in)

        kernel1 = np.ones((3, 3), np.uint8)
        void_mask_in1 = cv2.erode(void_mask, kernel1, cv2.BORDER_REFLECT)

        kernel2 = np.ones((6, 6), np.uint8)
        void_mask_in2 = cv2.erode(void_mask, kernel2, cv2.BORDER_REFLECT)

        # COLORING
        finalimg = deepcopy(self.zmat)
        finalimg = self.apply_cowmask(finalimg, copp_mask, 150, 60)
        finalimg[elem_mask == 255] = 200
        finalimg[pseudoelem_mask == 255] = 200

        cv2.GaussianBlur(finalimg, (9, 9), sigmaX=50, dst=finalimg)

        finalimg[void_mask == 255] = 70
        finalimg[void_mask_in1 == 255] = 80
        finalimg[void_mask_in2 == 255] = 90

        finalimg[coat_mask == 255] = 50
        finalimg[finalimg == 0] = 63

        cv2.GaussianBlur(finalimg, (3, 3), sigmaX=50, dst=finalimg)

        # NOISE
        noise = np.random.normal(0, 0.04 * finalimg + 1, (self.img_size, self.img_size))
        finalimg = np.subtract(finalimg, noise)
        finalimg = np.clip(finalimg, 0, 255).astype(np.uint8)

        # INTERFERENCE
        if np.random.rand() < 0.5:
            finalimg = self.interference(finalimg)

        # FINALIZING MASKS
        elem_mask[void_mask > 0] = 0
        copp_mask[void_mask > 0] = 0
        void_mask[void_mask > 0] = 255
        mask = np.stack((copp_mask, elem_mask, void_mask), axis=-1)

        return finalimg, mask

    def generate(self, save_path, N):
        img_dir = os.path.join(save_path, "img")
        msk_dir = os.path.join(save_path, "mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(msk_dir, exist_ok=True)
        file_prefix = self._random__hash(10)

        with tqdm(total=N, desc="Drawing") as pbar:
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as e:
                futures = [
                    e.submit(self._image_saver, save_path, file_prefix, i) for i in range(N)
                ]

                for future in concurrent.futures.as_completed(futures):
                    pbar.update(1)

        self.get_animation(save_path, file_prefix)
import h5py
import cv2
import numpy as np
import pandas as pd
import cv
import os.path

from h5py._hl import group
from numpy import ndarray


def get_descriptors(image, multiple_descriptors=False, stride=(32, 32)):
    if not multiple_descriptors:
        return cv2.HOGDescriptor().compute(img=cv2.resize(image, (64, 128)))
        # change ratio and resize to match 64x128 size
    else:
        # задание шага окошка детектора, с которым он будет скользить по изображению
        width,height = stride
        # я вычисляю только дескрипторы, которые целиком лежат на картинке
        i = 0
        # здесь получаю локации точек, где нужно считать hog
        loc = []
        while i + 128 <= image.shape[0]:
            j = 0
            while j + 64 <= image.shape[1]:
                loc.append((j, i))
                j += width
            i += height
        return cv2.HOGDescriptor().compute(image,(0,0),(0,0),locations=loc)

def generate_hog_hdf(image_dir, pos_df, hdf_path, multipledescriptors=True, append = False):
    mode = "w"
    if append:
        mode = "a"
    hdf = h5py.File(hdf_path, mode)
    if not append: hog_group = hdf.create_group("HOG")
    else:
        hog_group = hdf.get("/HOG")
    for i, position in pos_df.iterrows():
        img_name = position['image_name']
        x = position['x']
        y = position['y']
        theta = position['theta']
        img = cv2.imread(os.path.join(image_dir, img_name))  # type: ndarray
        dsc = get_descriptors(img,multipledescriptors)
        img_group = hog_group.create_group(img_name, True)
        img_group.create_dataset("description", data=dsc,
                                 chunks=True,
                                 compression="gzip", compression_opts=9)
        img_group.attrs["x"] = float(x)
        img_group.attrs["y"] = float(y)
        img_group.attrs["theta"] = float(theta)

def main():
    pos_df = pd.read_csv("hog_test/img_n_pos.csv")
    generate_hog_hdf("./hog_test/imgs", pos_df, "./hog_test/hogs.hdf")


if __name__ == '__main__':
    main()

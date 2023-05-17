import numpy as np
import cv2 as cv
import os


class Logger():
    def __init__(self, write_path) -> None:
        self.write_path = write_path

        os.makedirs(write_path, exist_ok=True)

        self.current_subdir = None

    def write_img(self, img: np.ndarray, name: str, subdir: str=None) -> None:
        if subdir is None:
            if self.current_subdir is None:
                path = f'{self.write_path}/{name}.png'
            else:
                path = f'{self.write_path}/{self.current_subdir}/{name}.png'
        else:
            path = f'{self.write_path}/{subdir}/{name}.png'

        cv.imwrite(path, img)

    def create_subdir(self, name: str):
        os.makedirs(f'{self.write_path}/{name}', exist_ok=True)

    def set_subdir(self, subdir: str) -> None:
        self.current_subdir = subdir
import os
import cv2
import numpy as np
import onnxruntime as ort


def draw_landmarks(image, kpts, color=(0, 0, 255), draw_connections=True):
    """Draw mark points on image"""
    end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1

    image = image.copy()
    kpts = kpts.copy()

    for j in range(kpts.shape[0]):
        # i = j + 17
        st = kpts[j, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, color, -1, cv2.LINE_AA)

        if not draw_connections or j in end_list:
            continue

        ed = kpts[j + 1, :2]
        image = cv2.line(
            image,
            (int(st[0]), int(st[1])),
            (int(ed[0]), int(ed[1])),
            color,
            1,
        )

    return image


def transform_pixel(pt, trans, inverse=False):
    if inverse is False:
        pt = pt @ (trans[:, 0:2].T) + trans[:, 2]
    else:
        pt = (pt - trans[:, 2]) @ np.linalg.inv(trans[:, 0:2].T)
    return pt


def bbox_to_warp_mat(bbox: np.ndarray, bbox_scale=1.0, dsize=256):
    x1, y1, x2, y2 = (bbox[:4] + 0.5).astype(np.int32)

    w = x2 - x1 + 1
    h = y2 - y1 + 1
    cx = x1 + w // 2
    cy = y1 + h // 2

    size = np.maximum(x2 - x1, y2 - y1)

    # ensure an even number
    size = (size * bbox_scale / 2.0 + 0.5).astype(np.int32) * 2

    x1 = cx - size / 2.0
    y1 = cy - size / 2.0

    scale = dsize / size
    warp_mat = np.array(
        [[scale, 0, -x1 * scale], [0, scale, -y1 * scale]], dtype=np.float32
    )

    return warp_mat


class DSLPT(object):
    """
    Facial landmark detector by DSLPT (https://github.com/Jiahao-UTS/DSLPT)
    """

    # fmt: off
    WFLW_98_TO_DLIB_68_IDX_MAPPING = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, \
                                      36, 37, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, \
                                      64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, \
                                      86, 87, 88, 89, 90, 91, 92, 93, 94, 95]
    # fmt: on

    def __init__(self, model_path: str):
        """Initialize a mark detector.

        Args:
            model_file (str): ONNX model path.
        """
        self.model_path = model_path

        assert os.path.exists(self.model_path), f"File not found: {self.model_path}"
        # sess_opts = ort.SessionOptions()
        # sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        # sess_opts.optimized_model_filepath = "DSLPT_opt.onnx"
        sess_options = ort.SessionOptions()

        self.ort_sess = ort.InferenceSession(
            self.model_path, sess_options, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self._input_size = self.ort_sess.get_inputs()[0].shape[-1]

    def __call__(self, image_rgb: np.ndarray, bbox: np.ndarray, return_input=False):
        """Detect facial marks from an face image.

        Args:
            image_rgb

        Returns:
            marks: the facial marks as a numpy array of shape [Batch, 98*2].
        """
        warp_mat = bbox_to_warp_mat(np.asarray(bbox), 1.15, self._input_size)
        warped_img = cv2.warpAffine(image_rgb, warp_mat, (self._input_size, self._input_size))
        img = warped_img.astype(np.float32).transpose(2, 0, 1) / 255.0

        marks = self.ort_sess.run(None, {"image": img[None]})[0]
        marks = transform_pixel(marks[0] * self._input_size, warp_mat, inverse=True)

        if return_input:
            return marks, warped_img, warp_mat
        else:
            return marks
    
    def forward(self, img):

        marks = self.ort_sess.run(None, {"image": img})[0][0] * self._input_size
 
        return marks

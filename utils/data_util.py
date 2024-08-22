import cv2
import torch, math
import numpy as np
from torchvision.utils import make_grid

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 1:
            if img.dtype == 'float64':
                img = img.astype('float32')
        img = torch.from_numpy(img.transpose(2, 0, 1)) / 255.
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)
    
    
def tensor2img(tensors, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    def _toimg(tensor, rgb2bgr, min_max):
        if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
            raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

        result_img = []
        for _tensor in tensor:
            _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
            _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

            n_dim = _tensor.dim()
            if n_dim == 4:
                img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
                img_np = img_np.transpose(1, 2, 0)
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif n_dim == 3:
                img_np = _tensor.numpy()
                img_np = img_np.transpose(1, 2, 0)
                if img_np.shape[2] == 1:  # gray image
                    img_np = np.squeeze(img_np, axis=2)
                else:
                    if rgb2bgr:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            elif n_dim == 2:
                img_np = _tensor.numpy()
            else:
                raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
            if out_type == np.uint8:
                # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
                img_np = (img_np * 255.0).round()
            img_np = img_np.astype(out_type)
            result_img.append(img_np)
        if len(result_img) == 1:
            result_img = result_img[0]
        if len(result_img) != 1:
            result_img = np.stack(result_img, axis=0)
        return result_img
    
    if isinstance(tensors, list):
        return [_toimg(tensor, rgb2bgr, min_max) for tensor in tensors]
    else:
        return _toimg(tensors, rgb2bgr, min_max)


def eliminate_batch(numpys):
    """ 
        Transfor multi-dimention to vis.
        b, h, w, c --> h, b*w, c
    """
    def elimi_batch(numpy_img):
        b, h, w, c = numpy_img.shape
        numpy_img = numpy_img.reshape(b*h, w, c)
        return numpy_img
    
    if isinstance(numpys, list):
        return [elimi_batch(numpy_img) for numpy_img in numpys]
    else:
        return elimi_batch(numpys)
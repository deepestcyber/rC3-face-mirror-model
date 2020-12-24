import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from functools import partial

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import skimage
import torch
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
from scipy.spatial import ConvexHull

import websocket
try:
    import thread
except ImportError:
    import _thread as thread
import time


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    return generator, kp_detector

import contextlib
import time

@contextlib.contextmanager
def timing(name, **kwargs):
    a = time.time()
    try:
        yield
    finally:
        pass
    b = time.time()
    print(f'{name} took {b-a} seconds')


def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            with timing('kp_detector'):
                kp_driving = kp_detector(driving_frame)
            with timing('normalize_kp'):
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            with timing('generator'):
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions


import base64
import PIL.Image
from io import BytesIO

class Animator:
    def __init__(self, source_image, driving_video, cpu=False):
        """`driving_video` is a misnomer, it is just a list with the first
        driving frame (as an image)
        """
        with torch.no_grad():
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()

            self.source = source
            self.kp_source = kp_detector(source)

            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            self.kp_driving_initial = kp_detector(driving[:, :, 0])

    def make_animation(
        self, source_image, driving_video, generator,
        kp_detector, relative=True, adapt_movement_scale=True, cpu=False,
    ):
        with torch.no_grad():
            source = self.source
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            if not cpu:
                driving = driving.cuda()
            kp_source = self.kp_source
            kp_driving_initial = self.kp_driving_initial

            driving_frame = driving[:, :, 0]
            with timing('kp_detector'):
                kp_driving = kp_detector(driving_frame)
            with timing('normalize_kp'):
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            with timing('generator'):
                out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            return out['prediction'].permute(0, 2, 3, 1).data.cpu().numpy()


animator = None

def on_message_real(ws, message):
    global animator

    # when a client is done, reset the worker
    # to its initial state to receive a new
    # stream of images; this is only necessary as long
    # as we assume that one worker == one user.
    if message == "reset":
        animator = None
        return

    # we expect an image encoded as base64 data url,
    # i.e. b"data:image/png;base64,<base64data>"
    cidx = message.find(b',')
    print(message[:30], cidx)
    frame_in_data = message
    frame_in = base64.b64decode(frame_in_data[cidx:])
    frame_img = PIL.Image.open(BytesIO(frame_in)).convert('RGBA')
    frame_in = np.array(frame_img)

    #frame_img.save('in.png', 'png')

    # normally we would resize but we make sure that we only get
    # 256x256 images so we are fine doing just the normalization
    # and channel selection.
    #driving_frame = resize(frame_in, (256, 256))[..., :3]
    driving_frame = frame_in[..., :3] / 255.

    # FIXME: this is problematic as it introduces a dependency on the
    # sender. for some reason the model wants the keypoints of the
    # first driving frame and keeps it for further processing.
    # maybe this is ok and we assign a worker to each user and
    # we only have a limited amount of possible users. we could
    # also work around this by introducing a session id which is
    # then used to get the animator instance for that session.
    if animator is None:
        animator = Animator(
            source_image,
            [driving_frame],
            cpu=opt.cpu,
        )

    predictions = animator.make_animation(
        source_image,
        [driving_frame],
        generator,
        kp_detector,
        relative=opt.relative,
        adapt_movement_scale=opt.adapt_scale,
        cpu=opt.cpu,
    )

    img = PIL.Image.fromarray(img_as_ubyte(predictions[0])).convert('RGB')

    #img.save('out.jpg', 'jpeg')

    b = BytesIO()
    img.save(b, 'jpeg')
    im_bytes = b.getvalue()
    frame_out_data = base64.b64encode(im_bytes)

    ws.send(frame_out_data)

def on_message(ws, message):
    with timing('on_message'):
        on_message_real(ws, message)

def on_error(ws, error):
    print(error)

def on_close(ws):
    print("### closed ###")

def on_open(opt, ws):
    global animator
    # in case of reconnect, reset the animator
    # instance that may still be initialized
    animator = None



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default='vox-cpk.pth.tar', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--server", default='ws://localhost:8080', help="address of the frontend")

    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    source_image = resize(source_image, (256, 256))[..., :3]

    generator, kp_detector = load_checkpoints(
        config_path=opt.config,
        checkpoint_path=opt.checkpoint,
        cpu=opt.cpu
    )

    # pip install websocket-client

    while True:
        # debug output when set to true
        websocket.enableTrace(False)
        ws = websocket.WebSocketApp(
            f"{opt.server}/registerCompute/test/supersecretsauce",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        ws.on_open = partial(on_open, opt)
        ws.run_forever()

        print("Connection was closed. Retrying in 5 seconds...")
        time.sleep(5)

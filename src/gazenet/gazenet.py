import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np

from . import vendgaze


class Gazenet:

    def __init__(self, model_path):
        self.model = Gazenet._load_model(model_path)
        self.idx_tensor = torch.FloatTensor([idx for idx in range(66)])

    def image_to_euler_angles(self, image, bbox):
        """
        image: width x height x 3
        bbox: (x_min, y_min, x_max, y_max)
        returns np.array of the 3 angles
        """
        image = self._preprocess(image, bbox)
        image = self._transform(image)

        image_shape = image.size()
        image = image.view(1, image_shape[0], image_shape[1], image_shape[2])

        yaw, pitch, roll = self.model(Variable(image))

        yaw_pred = F.softmax(yaw)
        pitch_pred = F.softmax(pitch)
        roll_pred = F.softmax(roll)

        return self._map_angles_to_continuous(yaw_pred, pitch_pred, roll_pred)

    def _preprocess(self, image, bbox):
        x_min = max(bbox[0] - 50, 0)
        y_min = max(bbox[1] - 50, 0)
        x_max = min(image.shape[0], bbox[2] + 50)
        y_max = min(image.shape[1], bbox[3] + 50)

        image = image[x_min:x_max, y_min:y_max]
        image = Image.fromarray(image)

        return image

    def _transform(self, image):
        transformations = transforms.Compose(
            [
                transforms.Scale(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        return transformations(image)

    def _map_angles_to_continuous(self, yaw, pitch, roll):
        def postproc(angle):
            return (
                torch.sum(
                    angle[0] *
                    self.idx_tensor) *
                3 -
                99).data.numpy()
        return np.array([postproc(yaw), postproc(pitch), postproc(roll)])

    @staticmethod
    def _load_model(model_path):
        model = vendgaze.GazeNet(
            block=torchvision.models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_bins=66
        )
        saved_state_dict = torch.load(model_path, map_location='cpu')
        model.eval()

        return model

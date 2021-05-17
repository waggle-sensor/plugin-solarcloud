import torch
import torch.nn as nn
from torchvision import transforms

import network
import utils

import os
from PIL import Image
import numpy as np
import datetime

import argparse
import glob

import waggle.plugin as plugin
from waggle.data import open_data_source

TOPIC_CLOUDCOVER = "env.coverage.cloud"
TOPIC_SOLARCLOUD = "env.irradiance.cloud"

plugin.init()


class ASPP_Main:
    def __init__(self, args):

        self.denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.args = args

        self.model = network.deeplabv3_resnet101(num_classes=2, output_stride=16)
        utils.set_bn_momentum(self.model.backbone, momentum=0.01)

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
        else:
            print("CUDA is not avilable; use CPU")
            self.device = torch.device('cpu')
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cpu'))

        self.model.load_state_dict(checkpoint['model_state'])
        self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()


    def run(self, image):
        """Do validation and return specified samples"""
        input_image = Image.open(image)
        input_image = input_image.resize((self.args.resize, self.args.resize))

        preprocess = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])
        input_tensor = preprocess(input_image)
        input_tensor = torch.unsqueeze(input_tensor, 0)

        input_image = input_tensor.to(self.device, dtype=torch.float32)
        score = self.model(input_image)

        output = score[0]
        output_predictions = output.argmax(0)

        scores = output_predictions.cpu().numpy().reshape(-1)
        cloud = 0
        for i in scores:
            if i == 1:
                cloud += 1
        ratio = cloud/len(scores)

        return ratio

class cal_max_irr:
    def __init__(self, maxirr_path):
        self.maxirrs = {}
        with open(maxirr_path, 'r') as f:
            for line in f:
                a = line.strip().split(' ')
                self.maxirrs[datetime.datetime.strptime(a[0], '%H:%M:%S')] = float(a[1])

    def cal(self, timestamp, args):
        timestamp_low = datetime.datetime.fromtimestamp(timestamp).time()
        timestamp_high = (datetime.datetime.fromtimestamp(timestamp)+datetime.timedelta(seconds=args.timeinterval)).time()
        for k, v in self.maxirrs.items():
            if k.time() > timestamp_low and k.time() < timestamp_high:
                return v

def run(args):
    aspp = ASPP_Main(args)
    if args.include_solar:
        maxirr = cal_max_irr(args.maxirr_path)

    with open_data_source(id=args.stream) as cap:
        timestamp, image = cap.get()

        ratio = aspp.run(image)

        plugin.publish(TOPIC_CLOUDCOVER, ratio, timestamp=timestamp)
        if args.debug:
            print(f"Measures published: Cloud coverage = {ratio}")

        if args.include_solar:
            current_max_irr = maxirr.cal(timestamp, args)
            irr = (1-ratio) * current_max_irr
            plugin.publish(TOPIC_SOLARCLOUD, irr, timestamp=timestamp)
            if args.debug:
                print(f"Measures published: Solar irradiance = {irr}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-stream', dest='stream', action='store', default="camera", help='ID or name of a stream, e.g. sample')
    parser.add_argument('--maxirr-path', type=str, help='maxirr path, required to calculate solar irradiance')
    parser.add_argument('--checkpoint', type=str, required=True, help='checkpoint path')
    parser.add_argument('--resize', type=int, default=300, help='resize image size')

    parser.add_argument('--include-solar', action='store_true', default=False, help="Publish solar irradiance")
    parser.add_argument('--timeinterval', type=int, default=15, help='time interval between each estimation in seconds')

    parser.add_argument('--debug', dest='debug', action='store_true', default=False, help='Debug flag')
    args = parser.parse_args()

    if args.include_solar and args.maxirr_path == None:
        parser.print_help()
        exit(1)

    if args.debug:
        print(f"Cloud cover estimation model loaded")

    run(args)

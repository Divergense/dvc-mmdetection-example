from argparse import ArgumentParser

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector



def parse_args():
	parser = ArgumentParser()
	parser.add_argument('config', help='Path to the config file')
	args = parser.parse_args()
	return args


def main(args):
    cfg = Config.fromfile(args.config)

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_detector(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    train_detector(model, datasets, cfg, distributed=False, validate=True)



if __name__ == '__main__':
    args = parse_args()
    main(args)

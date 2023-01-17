import copy

from mmcv import Config
from argparse import ArgumentParser


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('config', help='Path to the config file')
	parser.add_argument('output', help='Name of the resulted config')
	parser.add_argument('--modify', action='store_true', help='Whether \
		perform modification of the source config or not')
	parser.add_argument('--no-modify', dest='modify', action='store_false')
	parser.set_defaults(modify=True)
	args = parser.parse_args()
	return args


def main(args):
	base_cfg = Config.fromfile(args.config)
	cfg = copy.deepcopy(base_cfg)
	if args.modify:
		import os
		from yaml import safe_load
		from dotenv import load_dotenv
		from cfg_modification import (
			modify_cfg, 
			set_mlflow_logger,
			)

		load_dotenv()
		params_data = os.environ.get('PARAMS_DATA')
		with open(params_data, 'r') as data:
			params = safe_load(data)
		
		cfg = modify_cfg(cfg, params)
		try:
			import mlflow
		except ImportError as exc:
			print(exc)
		cfg = set_mlflow_logger(cfg, params)
	
	cfg.dump(args.output)  # save config into file



if __name__ == '__main__':
	args = parse_args()
	main(args)

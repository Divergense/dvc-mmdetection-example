import requests

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from urllib.parse import urlparse, unquote



TIMEOUT = 10
CHUNK_SIZE = 1024


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('url', help='URL of the downloaded file')
	parser.add_argument('-o', '--output', default=None, help='File name of the download result')
	args = parser.parse_args()	
	return args


def download(url, output):
	"""
	Download file located at specified URL and write it
	to output file.
	"""
	with requests.get(url, stream=True, timeout=TIMEOUT) as response:
		n = int(response.headers.get('content-length', 0))
		with tqdm.wrapattr(open(output, 'wb'), 'write', total=n) as out_file:
			for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
				out_file.write(chunk)


def extract_url_filename(url: str) -> str:
	url_parsed = urlparse(url)
	path = Path(unquote(url_parsed.path))
	filename = path.name
	return filename


def main(args):
	url = args.url
	output = args.output
	if output is None:
		output = Path('./')
	else:
		output = Path(output)
	if output.is_dir():
		filename = extract_url_filename(url)
		output = output / Path(filename)
	elif not output.parent.is_dir():
		raise RuntimeError(f'{output}. No such directory.')
	print("output file:", output)
	download(url, output)



if __name__ == '__main__':
	args = parse_args()
	main(args)


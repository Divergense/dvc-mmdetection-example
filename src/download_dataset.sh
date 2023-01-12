#!/bin/bash

link=https://download.openmmlab.com/mmdetection/data/kitti_tiny.zip
dest="$(pwd)/data"
filename="kitti_tiny.zip"
full_path="$dest/$filename"

echo "File will be safe at: $full_path"
wget -nc $link -O "$full_path"
echo "Extracting archive..."
unzip -n $full_path -d $dest > /dev/null
echo "Done!"


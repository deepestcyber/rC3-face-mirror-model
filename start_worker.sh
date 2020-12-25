#!/bin/sh

set -e

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 <id>"
	exit 1
fi

id=$1
server=wss://rc3-m.x3ro.de

run() {
	echo $@
	$@
}

opsdir=$(dirname $0)

run nvidia-docker run -t \
	-v "${opsdir}/volume:/app/volume" \
	-e SERVER="${server}" \
	-e SOURCE_IMAGE=source.png \
	-e CUDA_VISIBLE_DEVICES="$id" \
	anima

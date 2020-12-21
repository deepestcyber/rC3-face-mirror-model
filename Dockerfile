#FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
FROM nvcr.io/nvidia/cuda:11.0-cudnn8-runtime-ubuntu18.04
#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN DEBIAN_FRONTEND=noninteractive apt-get -qq update \
 && DEBIAN_FRONTEND=noninteractive apt-get -qqy install python3-pip ffmpeg git less nano libsm6 libxext6 libxrender-dev \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m app
USER app

COPY --chown=app . /app/
WORKDIR /app

RUN pip3 install --upgrade pip
#RUN pip3 install \
#  https://download.pytorch.org/whl/cu100/torch-1.0.0-cp36-cp36m-linux_x86_64.whl \
#  git+https://github.com/1adrianb/face-alignment \
#  -r requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

CMD python3 serve.py \
           --config config/vox-256.yaml \
           --source_image source.png \
           --checkpoint vox-adv-cpk.pth.tar \
           --relative \
		   --adapt_scale \
		   --server ${SERVER}

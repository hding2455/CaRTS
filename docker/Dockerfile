From pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel 

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y vim ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev

RUN pip install opencv-python scipy matplotlib einops imageio scikit-image positional_encodings
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git" 

RUN git clone https://github.com/hding2455/CaRTS /workspace/code
WORKDIR /workspace/code
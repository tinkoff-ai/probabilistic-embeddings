FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bck
RUN cat /etc/apt/sources.list.bck | sed 's#us\.archive#old-releases#g' > /etc/apt/sources.list
RUN apt-get update --fix-missing
RUN apt-get install -y python3 python3-pip psmisc git libturbojpeg
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools tox tox-current-env pytest jupyter matplotlib tensorboard \
    PyYAML==5.4.1 numpy==1.19.5 scipy==1.5.4 scikit-image==0.17.2 scikit-learn==0.24.2 Pillow==8.3.1 \
    torch==1.9.1 torchvision==0.10.1 pretrainedmodels==0.7.4 catalyst==21.9 mxnet==1.8.0.post0 optuna==2.10.0 \
    wandb faiss-cpu==1.7.0 jpeg4py==0.1.4 gitpython
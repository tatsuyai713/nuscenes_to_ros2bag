FROM ros:humble-ros-core

RUN apt-get update
RUN apt-get install -y git python3-pip python3-tf2-ros libgl1 libgeos-dev ros-humble-ros2bag ros-humble-rosbag2* ros-humble-foxglove-msgs  ros-humble-tf2-tools ros-humble-tf-transformations
RUN apt install -y python3-colcon-common-extensions

RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install shapely numpy==1.26.4 nuscenes-devkit mcap tqdm requests


RUN apt-get update && apt-get install -y \
    git \
    sed

# pypcdリポジトリをクローン
RUN git clone https://github.com/DanielPollithy/pypcd.git /pypcd

# 作業ディレクトリに移動
WORKDIR /pypcd

# np.floatをfloatに置換するスクリプトを作成
RUN echo '#!/bin/sh\n\
sed -i "s/np.float/float/g" pypcd/numpy_pc2.py' > replace_np_float.sh

# スクリプトに実行権限を付与
RUN chmod +x replace_np_float.sh

# スクリプトを実行して置換
RUN ./replace_np_float.sh

# 修正されたpypcdをインストール
RUN pip install .

#RUN pip3 install git+https://github.com/DanielPollithy/pypcd.git


COPY . /work

WORKDIR /work

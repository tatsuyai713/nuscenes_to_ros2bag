import argparse

import rclpy
from rclpy.duration import Duration
from rclpy.serialization import serialize_message

from std_msgs.msg import String
from sensor_msgs.msg import CameraInfo, CompressedImage, Imu, NavSatFix, PointCloud2, PointField
from builtin_interfaces.msg import Time 
from pypcd import numpy_pc2, pypcd

from tqdm import tqdm
import time 
from pathlib import Path
import os
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
import rosbag2_py

DATA_DIR = Path('work/data')
TOPIC_NAME = "/chatter"

def get_num_sample_data(nusc: NuScenes, scene):
    num_sample_data = 0
    sample = nusc.get("sample", scene["first_sample_token"])
    for sample_token in sample["data"].values():
        sample_data = nusc.get("sample_data", sample_token)
        while sample_data is not None:
            num_sample_data += 1
            sample_data = nusc.get("sample_data", sample_data["next"]) if sample_data["next"] != "" else None
    return num_sample_data

def get_time(data):
    t = Time()
    t.sec, msec = divmod(data["timestamp"], 1_000_000)
    t.nanosec = msec * 1000

    return t

def to_nano(stamp):
    return stamp.sec * 1000000000 + stamp.nanosec

def get_radar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    pc = pypcd.PointCloud.from_path(pc_filename)
    msg = numpy_pc2.array_to_pointcloud2(pc.pc_data)
    msg.header.frame_id = frame_id
    msg.header.stamp = get_time(sample_data)
    return msg

def get_lidar(data_path, sample_data, frame_id):
    pc_filename = data_path / sample_data['filename']
    pc_filesize = os.stat(pc_filename).st_size

    with open(pc_filename, 'rb') as pc_file:
        msg = PointCloud2()
        msg.header.frame_id = frame_id
        msg.header.stamp = get_time(sample_data)

        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
            PointField(name='ring', offset=16, datatype=PointField.FLOAT32, count=1),
        ]

        msg.is_bigendian = False
        msg.is_dense = True
        msg.point_step = len(msg.fields) * 4 # 4 bytes per field
        msg.row_step = pc_filesize
        msg.width = round(pc_filesize / msg.point_step)
        msg.height = 1 # unordered
        msg.data = pc_file.read()
        return msg

def write_scene(nusc, nusc_can, scene, output_path: str):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene['name']} Sample Data", leave=False)

    # writer.create_topic(
    #     rosbag2_py.TopicMetadata(
    #         name=TOPIC_NAME, type="std_msgs/msg/String", serialization_format="cdr"
    #     )
    # )
    #define sensor topics  

    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/lidar", type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/radar", type="sensor_msgs/msg/PointCloud2", serialization_format="cdr"
        )
    )
    #loop through smaples
    cur_sample = nusc.get("sample", scene["first_sample_token"])

    while cur_sample is not None:
        sample_lidar = nusc.get("sample_data", cur_sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", sample_lidar["ego_pose_token"])
        stamp = get_time(ego_pose)
        data_path = Path(nusc.dataroot)
        # iterate sensors
        for (sensor_id, sample_token) in cur_sample["data"].items():
            pbar.update(1)
            sample_data = nusc.get("sample_data", sample_token)
            topic = "/" + sensor_id
            if sample_data["sensor_modality"] == "radar":
                    msg = get_radar(data_path, sample_data, sensor_id)
                    writer.write("/radar", serialize_message(msg), to_nano(stamp))
            if sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    writer.write("/lidar", serialize_message(msg), to_nano(stamp))
        # move to the next sample
        cur_sample = nusc.get("sample", cur_sample["next"]) if cur_sample.get("next") != "" else None

    
    # start_time = 0
    # for i in range(10):
    #     msg = String()
    #     msg.data = f"Chatter #{i}"
    #     timestamp = start_time + (i * 100)
    #     writer.write(TOPIC_NAME, serialize_message(msg), timestamp)

    del writer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="output directory to create and write to")

    args = parser.parse_args()

    nusc = NuScenes(version="v1.0-mini", dataroot="/work/data", verbose=True)
    nusc_can = NuScenesCanBus(dataroot="/work/data")

    write_scene(nusc, nusc_can, nusc.scene[0], str(args.output))


if __name__ == "__main__":
    main()
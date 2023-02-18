from utils import * 
from sensor_utils import * 
from annotation_utils import *
from map_utils import * 

def write_scene(nusc, nusc_can, scene, output_path: str):
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )

    pbar = tqdm(total=get_num_sample_data(nusc, scene), unit="sample_data", desc=f"{scene['name']} Sample Data", leave=False)

    #loop through smaples
    cur_sample = nusc.get("sample", scene["first_sample_token"])

    #define sensor topics  
    create_topics(nusc, scene, writer)
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
                    writer.write(topic, serialize_message(msg), to_nano(stamp))
            elif sample_data["sensor_modality"] == "lidar":
                    msg = get_lidar(data_path, sample_data, sensor_id)
                    writer.write(topic, serialize_message(msg), to_nano(stamp))
            elif sample_data["sensor_modality"] == "camera":
                    msg = get_camera(data_path, sample_data, sensor_id)
                    writer.write(topic + "/image_rect_compressed", serialize_message(msg), to_nano(stamp))
                    msg = get_camera_info(nusc, sample_data, sensor_id)
                    writer.write(topic + "/camera_info", serialize_message(msg), to_nano(stamp))

            if sample_data['sensor_modality'] == 'camera':
                    msg = get_lidar_imagemarkers(nusc, sample_lidar, sample_data, sensor_id)
                    writer.write(topic + '/image_markers_lidar', serialize_message(msg), to_nano(stamp))
                    write_boxes_imagemarkers(nusc, writer, cur_sample['anns'], sample_data, sensor_id, topic, to_nano(stamp))
        
        # publish /pose
        pose_stamped = get_pose(stamp)
        writer.write('/pose', serialize_message(pose_stamped), to_nano(stamp))

        # collect all sensor frames after this sample but before the next sample
        non_keyframe_sensor_msgs = []
        for (sensor_id, sample_token) in cur_sample['data'].items():
            topic = '/' + sensor_id

            next_sample_token = nusc.get('sample_data', sample_token)['next']
            while next_sample_token != '':
                next_sample_data = nusc.get('sample_data', next_sample_token)
                # if next_sample_data['is_key_frame'] or get_time(next_sample_data).to_nsec() > next_stamp.to_nsec():
                #     break
                if next_sample_data['is_key_frame']:
                    break

                pbar.update(1)
                if next_sample_data['sensor_modality'] == 'radar':
                    msg = get_radar(data_path, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((to_nano(msg.header.stamp), topic, msg))
                elif next_sample_data['sensor_modality'] == 'lidar':
                    msg = get_lidar(data_path, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((to_nano(msg.header.stamp), topic, msg))
                elif next_sample_data['sensor_modality'] == 'camera':
                    msg = get_camera(data_path, next_sample_data, sensor_id)
                    camera_stamp_nsec = to_nano(msg.header.stamp)
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_rect_compressed', msg))

                    msg = get_camera_info(nusc, next_sample_data, sensor_id)
                    non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/camera_info', msg))

                    closest_lidar = find_closest_lidar(nusc, cur_sample["data"]["LIDAR_TOP"], camera_stamp_nsec)
                    if closest_lidar is not None:
                        msg = get_lidar_imagemarkers(nusc, closest_lidar, next_sample_data, sensor_id)
                        non_keyframe_sensor_msgs.append(
                            (
                                to_nano(msg.header.stamp),
                                topic + "/image_markers_lidar",
                                msg,
                            )
                        )

                    # Delete all image markers on non-keyframe camera images
                    # msg = get_remove_imagemarkers(sensor_id, 'LIDAR_TOP', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_lidar', msg))
                    # msg = get_remove_imagemarkers(sensor_id, 'annotations', msg.header.stamp)
                    # non_keyframe_sensor_msgs.append((camera_stamp_nsec, topic + '/image_markers_annotations', msg))

                next_sample_token = next_sample_data['next']

        # sort and publish the non-keyframe sensor msgs
        non_keyframe_sensor_msgs.sort(key=lambda x: x[0])
        for (_, topic, msg) in non_keyframe_sensor_msgs:
            writer.write(topic, serialize_message(msg), to_nano(msg.header.stamp))
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
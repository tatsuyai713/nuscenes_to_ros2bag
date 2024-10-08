# nuscenes_to_ros2bag

## Usage

### Converting nuScenes data to ROS2 bag
1. Download the [nuScenes mini dataset](https://nuscenes.org/nuscenes). You will need to make
   an account and agree to the terms of use.
2. Extract the following files into the `data/` directory:
    1. `can_bus.zip` to `data/`
    2. `nuScenes-map-expansion-v1.3.zip` to `data/maps`
    3. `v1.0-mini.tgz` to `data/`
3. Build and run the converter container with `./convert_to_ros2bag.sh`

The file of `/data` structure is the same as that of [foxglove/nuscenes2mcap](https://github.com/foxglove/nuscenes2mcap)
```
/data
├── can_bus
├── maps
│   ├── basemap
│   ├── expansion
│   └── prediction
├── samples
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   ├── LIDAR_TOP
│   ├── RADAR_BACK_LEFT
│   ├── RADAR_BACK_RIGHT
│   ├── RADAR_FRONT
│   ├── RADAR_FRONT_LEFT
│   └── RADAR_FRONT_RIGHT
├── sweeps
│   ├── CAM_BACK
│   ├── CAM_BACK_LEFT
│   ├── CAM_BACK_RIGHT
│   ├── CAM_FRONT
│   ├── CAM_FRONT_LEFT
│   ├── CAM_FRONT_RIGHT
│   ├── LIDAR_TOP
│   ├── RADAR_BACK_LEFT
│   ├── RADAR_BACK_RIGHT
│   ├── RADAR_FRONT
│   ├── RADAR_FRONT_LEFT
│   └── RADAR_FRONT_RIGHT
└── v1.0-mini
```



---
From [foxglove/nuscenes2mcap](https://github.com/foxglove/nuscenes2mcap)
# nuscenes2mcap

> _Convert [nuScenes](https://www.nuscenes.org/) data into [MCAP](https://mcap.dev/) format_

## Introduction

nuScenes is a large-scale dataset of autonomous driving in urban environments, provided free for non-commercial use. This project provides helper scripts to download the nuScenes dataset and convert scenes into [MCAP](https://mcap.dev) files for easy viewing in tools such as [Foxglove Studio](https://foxglove.dev/).

## Usage

### Converting the nuScenes data to MCAP
1. Download the [nuScenes mini dataset](https://nuscenes.org/nuscenes). You will need to make
   an account and agree to the terms of use.
1. Extract the following files into the `data/` directory:
    1. `can_bus.zip` to `data/`
    1. `nuScenes-map-expansion-v1.3.zip` to `data/maps`
    1. `v1.0-mini.tgz` to `data/`
1. Build and run the converter container with `./convert_mini_scenes.sh`

### Uploading data and events to Foxglove Data Platform

If you have a Foxglove Data Platform API key, you can use it to upload your scene data with:
```
docker build -t mcap_converter .
export FOXGLOVE_DATA_PLATFORM_TOKEN=<your secret token>
docker run -e FOXGLOVE_DATA_PLATFORM_TOKEN -v $(pwd)/output:/output \
    mcap_converter python3 upload_mcap.py /output
```

This repo also contains a script that can create synthetic events from the MCAP data.
```
docker run -e FOXGLOVE_DATA_PLATFORM_TOKEN -v $(pwd)/output:/output \
    mcap_converter python3 upload_events.py /output
```

### Updating Protobuf definitions
```
pip install mypy-protobuf
protoc --python_out=. --mypy_out=. --proto_path /path/to/foxglove/schemas/schemas/proto/ /path/to/foxglove/schemas/schemas/proto/foxglove/*.proto 
```

## License

nuscenes2mcap is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Stay in touch

Join our [Slack channel](https://foxglove.dev/join-slack) to ask questions, share feedback, and stay up to date on what our team is working on.

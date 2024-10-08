from utils import *

def get_imu_msg(imu_data):
    msg = Imu()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = get_utime(imu_data)
    msg.angular_velocity.x = imu_data['rotation_rate'][0];
    msg.angular_velocity.y = imu_data['rotation_rate'][1];
    msg.angular_velocity.z = imu_data['rotation_rate'][2];

    msg.linear_acceleration.x = imu_data['linear_accel'][0];
    msg.linear_acceleration.y = imu_data['linear_accel'][1];
    msg.linear_acceleration.z = imu_data['linear_accel'][2];

    msg.orientation.w = imu_data['q'][0];
    msg.orientation.x = imu_data['q'][1];
    msg.orientation.y = imu_data['q'][2];
    msg.orientation.z = imu_data['q'][3];
    
    return (msg.header.stamp, '/imu', msg)

def get_odom_msg(pose_data):
    msg = Odometry()
    msg.header.frame_id = 'map'
    msg.header.stamp = get_utime(pose_data)
    msg.child_frame_id = 'base_link'
    msg.pose.pose.position.x = pose_data['pos'][0]
    msg.pose.pose.position.y = pose_data['pos'][1]
    msg.pose.pose.position.z = pose_data['pos'][2]
    msg.pose.pose.orientation.w = pose_data['orientation'][0]
    msg.pose.pose.orientation.x = pose_data['orientation'][1]
    msg.pose.pose.orientation.y = pose_data['orientation'][2]
    msg.pose.pose.orientation.z = pose_data['orientation'][3]
    msg.twist.twist.linear.x = pose_data['vel'][0]
    msg.twist.twist.linear.y = pose_data['vel'][1]
    msg.twist.twist.linear.z = pose_data['vel'][2]
    msg.twist.twist.angular.x = pose_data['rotation_rate'][0]
    msg.twist.twist.angular.y = pose_data['rotation_rate'][1]
    msg.twist.twist.angular.z = pose_data['rotation_rate'][2]
    
    return (msg.header.stamp, '/odom', msg)

def get_basic_can_msg(name, diag_data):
    values = []
    for (key, value) in diag_data.items():
        if key != 'utime':
            values.append(KeyValue(key=key, value=str(round(value, 4))))

    msg = DiagnosticArray()
    msg.header.stamp = get_utime(diag_data)
    msg.status.append(DiagnosticStatus(name=name, level=DiagnosticStatus.OK, message='OK', values=values))

    return (msg.header.stamp, '/diagnostics', msg)
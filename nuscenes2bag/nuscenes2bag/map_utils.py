from utils import *

def get_pose(stamp):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.pose.orientation.w = 1.0
    return msg
from utils import *

EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}

def get_pose(stamp):
    msg = PoseStamped()
    msg.header.frame_id = 'base_link'
    msg.header.stamp = stamp
    msg.pose.orientation.w = 1.0
    return msg

def get_coordinate(ref_lat, ref_lon, bearing, dist):
    """
    Using a reference coordinate, extract the coordinates of another point in space given its distance and bearing
    to the reference coordinate. For reference, please see: https://www.movable-type.co.uk/scripts/latlong.html.
    :param ref_lat: Latitude of the reference coordinate in degrees, ie: 42.3368.
    :param ref_lon: Longitude of the reference coordinate in degrees, ie: 71.0578.
    :param bearing: The clockwise angle in radians between target point, reference point and the axis pointing north.
    :param dist: The distance in meters from the reference point to the target point.
    :return: A tuple of lat and lon.
    """
    lat, lon = math.radians(ref_lat), math.radians(ref_lon)
    angular_distance = dist / EARTH_RADIUS_METERS

    target_lat = math.asin(math.sin(lat) * math.cos(angular_distance) + math.cos(lat) * math.sin(angular_distance) * math.cos(bearing))
    target_lon = lon + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat),
        math.cos(angular_distance) - math.sin(lat) * math.sin(target_lat),
    )
    return math.degrees(target_lat), math.degrees(target_lon)


def derive_latlon(location, pose):
    """
    For each pose value, extract its respective lat/lon coordinate and timestamp.

    This makes the following two assumptions in order to work:
        1. The reference coordinate for each map is in the south-western corner.
        2. The origin of the global poses is also in the south-western corner (and identical to 1).
    :param location: The name of the map the poses correspond to, ie: 'boston-seaport'.
    :param poses: All nuScenes egopose dictionaries of a scene.
    :return: A list of dicts (lat/lon coordinates and timestamps) for each pose.
    """
    assert location in REFERENCE_COORDINATES.keys(), f"Error: The given location: {location}, has no available reference."

    reference_lat, reference_lon = REFERENCE_COORDINATES[location]
    x, y = pose["translation"][:2]
    bearing = math.atan(x / y)
    distance = math.sqrt(x**2 + y**2)
    lat, lon = get_coordinate(reference_lat, reference_lon, bearing, distance)
    return lat, lon

def get_transform(data):
    t = Transform()
    t.translation.x = data['translation'][0]
    t.translation.y = data['translation'][1]
    t.translation.z = data['translation'][2]
    
    t.rotation.w = data['rotation'][0]
    t.rotation.x = data['rotation'][1]
    t.rotation.y = data['rotation'][2]
    t.rotation.z = data['rotation'][3]
    
    return t

def get_gps(location, ego_pose, stamp):
    gps = NavSatFix()
    gps.header.frame_id = 'base_link'
    gps.header.stamp = stamp
    gps.status.status = 1
    gps.status.service = 1
    lat, lon = derive_latlon(location, ego_pose)
    gps.latitude = lat
    gps.longitude = lon
    gps.altitude = get_transform(ego_pose).translation.z
    return gps
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import random
import time

client = carla.Client('localhost', 2000)
client.set_timeout(10.0) #seconds
world = client.get_world()
worldMap = world.get_map()
worldBPLibrary = world.get_blueprint_library()

camera = world.get_actors().filter('sensor.camera.*')[0]
print(camera.attributes)
vehicle = camera.parent
print(vehicle.attributes)

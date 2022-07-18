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
town = client.get_available_maps()[0]
print(town)
world = client.load_world(town)
'''
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)
world = client.reload_world(False) # Reload map with same world settings
'''
worldMap = world.get_map()
worldBPLibrary = world.get_blueprint_library()

actorList = []

#weather = carla.WeatherParameters(
#        cloudiness=0.0,
#        precipitation=0.0,
#        sun_altitude_angle=90.0)
world.set_weather(carla.WeatherParameters.WetCloudySunset)
#print(world.get_weather())

#cameraBP
cameraState = carla.Transform(carla.Location(x=230, y=195, z=40), carla.Rotation(yaw=180))
       
vehicleBP = worldBPLibrary.find('vehicle.audi.tt')
vehicleBP.set_attribute('color', '255,0,0') #Strange but carla.Color object doesn't work here, need to format color as string 'RGB'

spawnPoints = worldMap.get_spawn_points()
print(vehicleBP)
#print(f'Number of spawnpoints: {len(spawnPoints)}')
print(spawnPoints[0])
vehicle = world.spawn_actor(vehicleBP, spawnPoints[0])
actorList.append(vehicle)

spectator = world.get_spectator()
refTransform = vehicle.get_transform()
spectator.set_transform(carla.Transform(refTransform.location + carla.Location(z=20), carla.Rotation(pitch=-75)))




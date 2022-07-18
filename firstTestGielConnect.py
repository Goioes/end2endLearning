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

#vehicleBP = random.choice(worldBPLibrary.filter('vehicle.*.*'))
#spawnPoints = worldMap.get_spawn_points()
#print(vehicleBP)

print(world.get_actors())
vehicleList = world.get_actors().filter('vehicle.*.*')
print(vehicleList)
vehicle = vehicleList[0]
print(vehicle.id, vehicle.type_id[8:])

cameraBP = worldBPLibrary.find('sensor.camera.rgb')
cameraBP.set_attribute('image_size_x', '600')
cameraBP.set_attribute('image_size_y', '600')
cameraBP.set_attribute('sensor_tick', '0.5')
cameraBP.set_attribute('enable_postprocess_effects', 'True')
cameraFront = world.spawn_actor(cameraBP, carla.Transform(carla.Location(z=2), carla.Rotation()), attach_to=vehicle)
cameraFront.listen(lambda image: image.save_to_disk(f'output/testCamera/Front/Image{image.frame}_{image.timestamp}.png'))
cameraRear = world.spawn_actor(cameraBP, carla.Transform(carla.Location(z=2), carla.Rotation(yaw=180)), attach_to=vehicle)
cameraRear.listen(lambda image: image.save_to_disk(f'output/testCamera/Rear/Image{image.frame}_{image.timestamp}.png'))

vehicle.apply_control(carla.VehicleControl(throttle=1))
time.sleep(1)
vehicle.set_light_state(carla.VehicleLightState.Position)
vehicle.set_autopilot(True)
time.sleep(20)




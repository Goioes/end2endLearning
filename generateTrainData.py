import glob
import os
import sys
import random
import time
import queue
import pygame
import csv

# Need to append path to .egg file to import carla
try:
    sys.path.append(glob.glob('/home/fhasanabadi/Git/carla/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print('Error finding the egg file')
import carla

# Connect to server and switch map
client = carla.Client('localhost', 2000)
client.set_timeout(10.0) #seconds
town = client.get_available_maps()[0]
print(town)
world = client.load_world(town)
pygame.init()
pygameClock = pygame.time.Clock()

# Initialize settings and prepare scenario
try:
    # Settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Get map and stuff
    worldMap = world.get_map()
    worldBPLibrary = world.get_blueprint_library()
    world.set_weather(carla.WeatherParameters.WetCloudySunset)
    actorList = []
    sensorList = []

    # Generate vehicle
    vehicleBP = worldBPLibrary.find('vehicle.audi.tt')
    vehicleBP.set_attribute('color', '255,0,0') # Strange but carla.Color object doesn't work here, need to format color as string 'RGB'
    vehicle = world.spawn_actor(vehicleBP, worldMap.get_spawn_points()[0])
    actorList.append(vehicle)
    vehicle.set_autopilot(True)
    print(vehicle.id, vehicle.type_id[8:])
    spawnTime = 80 # Give some time for vehicle to spawn
    for step in range(spawnTime):
        world.tick()
        pygameClock.tick()

    # Set the view
    spectator = world.get_spectator()
    refTransform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(refTransform.location + carla.Location(z=20), carla.Rotation(pitch=-75)))

    # Prepare and spawn cameras
    cameraBP = worldBPLibrary.find('sensor.camera.rgb')
    cameraBP.set_attribute('image_size_x', '600')
    cameraBP.set_attribute('image_size_y', '600')
    cameraBP.set_attribute('sensor_tick', '0.5')
    cameraBP.set_attribute('enable_postprocess_effects', 'True')
    cameraFront = world.spawn_actor(cameraBP, carla.Transform(carla.Location(z=2), carla.Rotation()), attach_to=vehicle)
    cameraLeft = world.spawn_actor(cameraBP, carla.Transform(carla.Location(z=2), carla.Rotation(yaw=45)), attach_to=vehicle)
    cameraRight = world.spawn_actor(cameraBP, carla.Transform(carla.Location(z=2), carla.Rotation(yaw=-45)), attach_to=vehicle)
    sensorList.extend([cameraFront, cameraLeft, cameraRight])

    # Prepare queues to store incoming images
    imageQueueFront = queue.Queue()
    cameraFront.listen(imageQueueFront.put)
    imageQueueLeft = queue.Queue()
    cameraLeft.listen(imageQueueLeft.put)
    imageQueueRight = queue.Queue()
    cameraRight.listen(imageQueueRight.put)
    
    # Prepare csv file to store controls
    csvHeader = ['step', 'throttle', 'steer']
    with open('Data/controls.csv', 'w', newline='') as f:
        csvWriter = csv.writer(f)
        csvWriter.writerow(csvHeader)

        # Run simulation and gather some data
        simulationSteps = 1000 # Simulate for 100 seconds
        for step in range(simulationSteps):
            world.tick()
            pygameClock.tick()
            try:
                imageFront = imageQueueFront.get(block=False)
                imageFront.save_to_disk(f'Data/FrontImages/Image{step}.png')
                imageLeft = imageQueueLeft.get(block=False)
                imageLeft.save_to_disk(f'Data/LeftImages/Image{step}.png')
                imageRight = imageQueueRight.get(block=False)
                imageRight.save_to_disk(f'Data/RightImages/Image{step}.png')
                controls = vehicle.get_control()
                csvWriter.writerow([step, controls.throttle, controls.steer])
                print(f'Controls: {controls.throttle}, {controls.steer}')
            except queue.Empty:
                pass # Cameras dont listen at every tick

# Leave everything clean and nice
finally:
    settings.synchronous_mode = False
    world.apply_settings(settings)
    for actor in actorList:
        actor.destroy
    for sensor in sensorList:
        sensor.destroy
    pygame.quit()
    print("I'm done here for today")

from djitellopy import Tello

# Initialize the Tello object
tello = Tello()

try:
    # Connect to the drone
    tello.connect()
    print(f"Battery life: {tello.get_battery()}%")

   

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # End the session
    tello.end()

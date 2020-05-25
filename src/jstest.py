"""Simple gamepad/joystick test example."""

from __future__ import print_function
import math
import sys
sys.path.append("~/jetbot/jetbot")
sys.path.append("~/jetcam/jetcam")

import cv2
import inputs
from jetcam.csi_camera import CSICamera
from jetbot import Robot, Camera


EVENT_ABB = (
    # D-PAD, aka HAT
    ('Absolute-ABS_HAT0X', 'HX'),
    ('Absolute-ABS_HAT0Y', 'HY'),

    # Face Buttons
    ('Key-BTN_NORTH', 'N'),
    ('Key-BTN_EAST', 'E'),
    ('Key-BTN_SOUTH', 'S'),
    ('Key-BTN_WEST', 'W'),

    # Other buttons
    ('Key-BTN_THUMBL', 'THL'),
    ('Key-BTN_THUMBR', 'THR'),
    ('Key-BTN_TL', 'TL'),
    ('Key-BTN_TR', 'TR'),
    ('Key-BTN_TL2', 'TL2'),
    ('Key-BTN_TR2', 'TR3'),
    ('Key-BTN_MODE', 'M'),
    ('Key-BTN_START', 'ST'),

    # PiHUT SNES style controller buttons
    ('Key-BTN_TRIGGER', 'N'),
    ('Key-BTN_THUMB', 'E'),
    ('Key-BTN_THUMB2', 'S'),
    ('Key-BTN_TOP', 'W'),
    ('Key-BTN_BASE3', 'SL'),
    ('Key-BTN_BASE4', 'ST'),
    ('Key-BTN_TOP2', 'TL'),
    ('Key-BTN_PINKIE', 'TR')
)

    

# This is to reduce noise from the PlayStation controllers
# For the Xbox controller, you can set this to 0
MIN_ABS_DIFFERENCE = 5

def normalize(x):
    max_x, min_x = 255, 0
    return ((x - min_x) / (max_x - min_x))


class JSTest(object):
    """Simple joystick test class."""
    def __init__(self, robot,camera, gamepad=None, abbrevs=EVENT_ABB):
        self.camera = camera
        self.robot = robot
        self.btn_state = {}
        self.old_btn_state = {}
        self.speed = 0.1
        self.abs_state = {}
        self.old_abs_state = {}
        self.counter = 0
        self.abbrevs = dict(abbrevs)
        for key, value in self.abbrevs.items():
            if key.startswith('Absolute'):
                self.abs_state[value] = 0
                self.old_abs_state[value] = 0
            if key.startswith('Key'):
                self.btn_state[value] = 0
                self.old_btn_state[value] = 0
        self._other = 0
        self.gamepad = gamepad
        if not gamepad:
            self._get_gamepad()


    def forward(self):
        print("inc")
        self.speed += 0.1
        self.robot.forward(self.speed)

    def backward(self):
        print("dec")
        self.speed -= 0.1
        self.robot.backward(self.speed)
        
    def stop(self):
        print("stop")
        self.robot.stop()

    def steer_x(self, status):
        norm = normalize(status)
        if norm < 0.5:
            self.robot.left(self.speed)
            self.robot.forward(self.speed)
        else:
            self.robot.right(self.speed)
            self.robot.forward(self.speed)
        print("steer_x: ", norm)
        print("steer_x: ", status)

    def steer_y(self, status):
        norm = normalize(status)
        if norm > 0.51:
            self.robot.backward(self.speed)
        else:
            self.robot.forward(self.speed)
        print("steer_y: ", norm)
        print("steer_y: ", status)

    def start_recording(self):
        image = self.camera.read()
        button_state_X = self.abs_state.get("A0", 0)
        button_state_Y = self.abs_state.get("A1", 0)
        base_string = f"images/drive_{self.counter}_{button_state_X}_{button_state_Y}.jpg"
        cv2.imwrite(base_string, image)
        self.counter += 1

    def stop_recording(self):
        #self.camera.stop()
        pass

    def get_x(self):
        pass

    def _get_gamepad(self):
        """Get a gamepad object."""
        try:
            self.gamepad = inputs.devices.gamepads[0]
        except IndexError:
            raise inputs.UnpluggedError("No gamepad found.")

    def handle_unknown_event(self, event, key):
        """Deal with unknown events."""
        if event.ev_type == 'Key':
            new_abbv = 'B' + str(self._other)
            self.btn_state[new_abbv] = 0
            self.old_btn_state[new_abbv] = 0
        elif event.ev_type == 'Absolute':
            new_abbv = 'A' + str(self._other)
            self.abs_state[new_abbv] = 0
            self.old_abs_state[new_abbv] = 0
        else:
            return None

        self.abbrevs[key] = new_abbv
        self._other += 1

        return self.abbrevs[key]

    def handle_event(self, event):
        event_code = event.code
        event_state = event.state

        if event_code == "BTN_TL":
            self.forward()
        elif event_code == "BTN_TR":
            self.backward()
        elif event_code == "BTN_EAST":
            self.stop()
        elif event_code == "ABS_X":
            self.steer_x(event_state)
        elif event_code == "ABS_Y":
            self.steer_y(event_state)
        elif event_code == "BTN_WEST":
            self.stop_recording()
        elif event_code == "BTN_NORTH":
            self.start_recording()

        else:
            print("Invalid")

    def process_event(self, event):
        """Process the event into a state."""
        if event.ev_type == 'Sync':
            return
        if event.ev_type == 'Misc':
            return
        key = event.ev_type + '-' + event.code
        try:
            abbv = self.abbrevs[key]
        except KeyError:
            abbv = self.handle_unknown_event(event, key)
            if not abbv:
                return
        if event.ev_type == 'Key':
            self.old_btn_state[abbv] = self.btn_state[abbv]
            self.btn_state[abbv] = event.state
        if event.ev_type == 'Absolute':
            self.old_abs_state[abbv] = self.abs_state[abbv]
            self.abs_state[abbv] = event.state
        self.output_state(event.ev_type, abbv)
        self.handle_event(event)

    def format_state(self):
        """Format the state."""
        output_string = ""
        for key, value in self.abs_state.items():
            output_string += key + ':' + '{:>4}'.format(str(value) + ' ')

        for key, value in self.btn_state.items():
            output_string += key + ':' + str(value) + ' '

        return output_string

    def output_state(self, ev_type, abbv):
        """Print out the output state."""
        if ev_type == 'Key':
            if self.btn_state[abbv] != self.old_btn_state[abbv]:
                print(self.format_state())
                return

        if abbv[0] == 'H':
            print(self.format_state())
            return

        difference = self.abs_state[abbv] - self.old_abs_state[abbv]
        if (abs(difference)) > MIN_ABS_DIFFERENCE:
            print(self.format_state())

    def process_events(self):
        """Process available events."""
        try:
            events = self.gamepad.read()
        except EOFError:
            events = []
        for event in events:
            self.process_event(event)



def main():
    """Process all events forever."""
    robot = Robot()
    camera = CSICamera(width=224, height=224, capture_width=3280, capture_height=2464, capture_fps=21)
    jstest = JSTest(robot, camera)
    while 1:
        jstest.process_events()


if __name__ == "__main__":
    main()

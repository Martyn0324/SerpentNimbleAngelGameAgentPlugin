from serpent.game_agent import GameAgent
import collections
import serpent.utilities
from serpent.sprite_locator import SpriteLocator
from serpent.sprite import Sprite
import numpy as np
from datetime import datetime
from serpent.frame_grabber import FrameGrabber
import skimage.color
import skimage.measure
import serpent.cv
import serpent.utilities
import serpent.ocr
import gc
import re

from serpent.input_controller import InputController
from serpent.input_controller import MouseEvent, MouseEvents, MouseButton

from serpent.config import config

from redis import StrictRedis



import pickle

from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent
from serpent.machine_learning.reinforcement_learning.agents.ppo_agent import PPOAgent
#from serpent.machine_learning.reinforcement_learning.ddqn import DDQN

import time

from serpent.input_controller import KeyboardEvent, KeyboardEvents
from serpent.input_controller import MouseEvent, MouseEvents

from serpent.config import config

from serpent.analytics_client import AnalyticsClient



class Environment:

    def __init__(self, name, game_api=None, input_controller=None):
        self.name = name

        self.game_api = game_api
        self.input_controller = input_controller

        self.game_state = dict()

        self.analytics_client = AnalyticsClient(project_key=config["analytics"]["topic"])

        self.reset()

    @property
    def episode_duration(self):
        return time.time() - self.episode_started_at

    @property
    def episode_over(self):
        if self.episode_maximum_steps is not None:
            return self.episode_steps >= self.episode_maximum_steps
        else:
            return False

    @property
    def new_episode_data(self):
        return dict()

    @property
    def end_episode_data(self):
        return dict()

    def new_episode(self, maximum_steps=None, reset=False):
        self.episode_steps = 0
        self.episode_maximum_steps = maximum_steps

        self.episode_started_at = time.time()

        if not reset:
            self.episode += 1

        self.analytics_client.track(
            event_key="NEW_EPISODE",
            data={
                "episode": self.episode,
                "episode_data": self.new_episode_data,
                "maximum_steps": self.episode_maximum_steps
            }
        )

    def end_episode(self):
        self.analytics_client.track(
            event_key="END_EPISODE",
            data={
                "episode": self.episode,
                "episode_data": self.end_episode_data,
                "episode_steps": self.episode_steps,
                "maximum_steps": self.episode_maximum_steps
            }
        )

    def episode_step(self):
        self.episode_steps += 1
        self.total_steps += 1

        self.analytics_client.track(
            event_key="EPISODE_STEP",
            data={
                "episode": self.episode,
                "episode_step": self.episode_steps,
                "total_steps": self.total_steps
            }
        )

    def reset(self):
        self.total_steps = 0

        self.episode = 0
        self.episode_steps = 0

        self.episode_maximum_steps = None

        self.episode_started_at = None

    def update_game_state(self, game_frame):
        raise NotImplementedError()

    def perform_input(self, actions):
        discrete_keyboard_keys = set()
        discrete_keyboard_labels = set()

        for label, game_input, value in actions:
            # Discrete Space
            if value is None:
                if len(game_input) == 0:
                    discrete_keyboard_labels.add(label)
                    continue

                for game_input_item in game_input:
                    if isinstance(game_input_item, KeyboardEvent):
                        if game_input_item.event == KeyboardEvents.DOWN:
                            discrete_keyboard_keys.add(game_input_item.keyboard_key)
                            discrete_keyboard_labels.add(label)

        discrete_keyboard_keys_sent = False

        for label, game_input, value in actions:
            # Discrete
            if value is None:
                # Discrete - Keyboard
                if (len(discrete_keyboard_keys) == 0 and len(discrete_keyboard_labels) > 0) or isinstance(game_input[0] if len(game_input) else None, KeyboardEvent):
                    if not discrete_keyboard_keys_sent:
                        self.input_controller.handle_keys(list(discrete_keyboard_keys))

                        self.analytics_client.track(
                            event_key="GAME_INPUT",
                            data={
                                "keyboard": {
                                    "type": "DISCRETE",
                                    "label": " - ".join(sorted(discrete_keyboard_labels)),
                                    "inputs": sorted([keyboard_key.value for keyboard_key in discrete_keyboard_keys])
                                },
                                "mouse": {}
                            }
                        )

                        discrete_keyboard_keys_sent = True
                # Discrete - Mouse
                elif isinstance(game_input[0], MouseEvent):
                    for event in game_input:
                        if event.event == MouseEvents.CLICK:
                            self.input_controller.click(button=event.button, x=event.x, y=event.y)
                        elif event.event == MouseEvents.CLICK_DOWN:
                            self.input_controller.click_down(button=event.button)
                        elif event.event == MouseEvents.CLICK_UP:
                            self.input_controller.click_up(button=event.button)
                        elif event.event == MouseEvents.CLICK_SCREEN_REGION:
                            screen_region = event.kwargs["screen_region"]
                            self.input_controller.click_screen_region(button=event.button, screen_region=screen_region)
                        elif event.event == MouseEvents.SCROLL:
                            self.input_controller.scroll(direction=event.direction)

                        self.analytics_client.track(
                            event_key="GAME_INPUT",
                            data={
                                "keyboard": {},
                                "mouse": {
                                    "type": "DISCRETE",
                                    "label": label,
                                    "label_technical": event.as_label,
                                    "input": event.as_input,
                                    "value": value
                                }
                            }
                        )
            # Continuous
            else:
                if isinstance(game_input[0], KeyboardEvent):
                    self.input_controller.tap_keys(
                        [event.keyboard_key for event in game_input],
                        duration=value
                    )

                    self.analytics_client.track(
                        event_key="GAME_INPUT",
                        data={
                            "keyboard": {
                                "type": "CONTINUOUS",
                                "label": label,
                                "inputs": sorted([event.keyboard_key.value for event in game_input]),
                                "duration": value
                            },
                            "mouse": {}
                        }
                    )
                elif isinstance(game_input[0], MouseEvent):
                    for event in game_input:
                        if event.event == MouseEvents.CLICK_SCREEN_REGION:
                            screen_region = event.kwargs["screen_region"]
                            self.input_controller.click_screen_region(button=event.button, screen_region=screen_region)
                        elif event.event == MouseEvents.MOVE:
                            self.input_controller.move(x=event.x, y=event.y)
                        elif event.event == MouseEvents.MOVE_RELATIVE:
                            self.input_controller.move(x=event.x, y=event.y, absolute=False)
                        elif event.event == MouseEvents.DRAG_START:
                            screen_region = event.kwargs.get("screen_region")
                            coordinates = self.input_controller.ratios_to_coordinates(value, screen_region=screen_region)

                            self.input_controller.move(x=coordinates[0], y=coordinates[1], duration=0.1)
                            self.input_controller.click_down(button=event.button)
                        elif event.event == MouseEvents.DRAG_END:
                            screen_region = event.kwargs.get("screen_region")
                            coordinates = self.input_controller.ratios_to_coordinates(value, screen_region=screen_region)

                            self.input_controller.move(x=coordinates[0], y=coordinates[1], duration=0.1)
                            self.input_controller.click_up(button=event.button)

                        self.analytics_client.track(
                            event_key="GAME_INPUT",
                            data={
                                "keyboard": {},
                                "mouse": {
                                    "type": "CONTINUOUS",
                                    "label": label,
                                    "label_technical": event.as_label,
                                    "input": event.as_input,
                                    "duration": value
                                }
                            }
                        )


    def clear_input(self):
        self.input_controller.handle_keys([])




import enum

from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
import os

class InputControlTypes(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1


class SerpentNimbleAngelGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None

        self._reset_game_state()

    def _reset_game_state(self):
        self.game_state = {
            "hp": 3,
            "score": 0,
            "bombs": 3,
            "run_reward": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "run_predicted_actions": 0,
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "run_timestamp": datetime.utcnow(),
        }


    def setup_play(self):

        '''input_mapping = {
            "UP": [KeyboardKey.KEY_UP],
            "DOWN": [KeyboardKey.KEY_DOWN],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "RIGHT": [KeyboardKey.KEY_RIGHT],
            "UP-LEFT": [KeyboardKey.KEY_UP, KeyboardKey.KEY_LEFT],
            "UP-RIGHT": [KeyboardKey.KEY_UP, KeyboardKey.KEY_RIGHT],
            "DOWN-LEFT": [KeyboardKey.KEY_DOWN, KeyboardKey.KEY_LEFT],
            "DOWN-RIGHT": [KeyboardKey.KEY_DOWN, KeyboardKey.KEY_RIGHT],
            "Shoot" : [KeyboardKey.KEY_Z],
            "Aura": [KeyboardKey.KEY_X]
        }'''
        
        '''self.key_mapping = {
            KeyboardKey.KEY_UP.name: "UP",
            KeyboardKey.KEY_LEFT.name: "LEFT",
            KeyboardKey.KEY_DOWN.name: "DOWN",
            KeyboardKey.KEY_RIGHT.name: "RIGHT",
            KeyboardKey.KEY_Z.name: "Shoot",
            KeyboardKey.KEY_X.name: "Aura"
        }'''
        '''input_mapping0 = {
            "A,1": [InputController.move(self, x=5, y=5)],
            "A,2": [InputController.move(self, x=8, y=8)]
        }'''

        '''input_mapping1 = {
            "A,1": [controller.move(self, x=5, y=5)],
            "A,2": [controller.move(self, x=8, y=8)]
        }'''


        '''input_mapping2 = {
                "A,1": [MouseEvent(MouseEvents.MOVE,  x=5, y=5)],
                "A,2": [MouseEvent(MouseEvents.MOVE,  x=8, y=8)]
            }'''
        
        '''input_mapping3 = {
            "A,1": [self.input_controller.move(x=5, y=5)],
            "A,2": [self.input_controller.move(x=8, y=8)]
        }'''


        self.game_inputs = [{
            "name" : "Controls",
            "control_type" : InputControlTypes.DISCRETE,
            "inputs" : self.game.api.combine_game_inputs(["MOVEMENT"]),
            "value": None}]
        
        # Move mouse = Control Type Continuous. We should make a second agent
        # responsible for moving the mouse.
        
        self.game_inputs2 = [{
            "name": "Mouse",
            "control_type": InputControlTypes.CONTINUOUS,
            "inputs": self.game.api.combine_game_inputs(["MOUSE"]),
            "value": 0.05
            }]


        # Rainbow DQN fails to generate inputs for mouse movements (game_inputs2) ---- SOLVED! Added "value" key in self.game_inputs
        # AND in RainbowDQNAgent code.
        # Trying PPO - Fail. Damn Pytorch
        # Using DDQN - Success? Attention to cuDNN though

        '''action_space = KeyboardMouseActionSpace(
            action_keys=[None, "A,1", "A,2"]
        )'''

        self.agent_actions = RainbowDQNAgent('Angel_actions', game_inputs=self.game_inputs)
        self.agent_mouse = RainbowDQNAgent("Angel_mouse", game_inputs=self.game_inputs2)
        #self.agent_mouse = PPOAgent("Angel_mouse", game_inputs=self.game_inputs2, input_shape=(12,12,4))
        '''self.agent_movement = DDQN(
            input_shape=(12,12,4),
            input_mapping=input_mapping3,
            action_space=action_space,
            model_file_path=None
                          )'''

    def handle_play(self, game_frame):

        self.game_state["hp"] = self._measure_hp(game_frame)
        self.game_state["score"] = self._measure_score(game_frame)
        self.game_state["bombs"] = self._measure_bomb(game_frame)
        
        # While we are testing mouse inputs, we're gonna use reward = 0
        
        #reward_actions = 0
        #reward_mouse = reward_actions

        reward_actions = self._reward(self.game_state, game_frame)
        reward_mouse = reward_actions
        
        self.agent_actions.observe(reward=reward_actions)
        self.agent_mouse.observe(reward=reward_mouse)
        
        frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        agent_actions = self.agent_actions.generate_actions(frame_buffer)
        agent_mouse = self.agent_mouse.generate_actions(frame_buffer)

        Environment.perform_input(self, actions=agent_actions)
        Environment.perform_input(self, actions=agent_mouse)

        self.game_state['current_run_steps'] += 1

        serpent.utilities.clear_terminal()
        print(f"Current HP: {self.game_state['hp']}")
        print(f"Current Score: {self.game_state['score']}")
        print(f"Bombs: {self.game_state['bombs']}")
        print(f"Current Reward: {self.game_state['run_reward']}")
        print(f"Current Run: {self.game_state['current_run']}")
        print(f"Current Run Step: {self.game_state['current_run_steps']}")
        #print(f"\n\nagent_mouse:\n\n{agent_mouse}\n\n\n")

        self._check_end(game_frame)

        # Let's solve this mouse problem once and for all:

        '''for label, game_input, value in agent_mouse:
            print(f"agent_mouse\nlabel: {label}\ngame_input:{game_input}\nvalue:{value}\n\n")

            for event in game_input:
                print(f"event in game_input: {event}")

        for label, game_input, value in agent_actions:
            print(f"\n\nagent_actions\nlabel: {label}\ngame_input:{game_input}\nvalue:{value}\n\n")

            for event in game_input:
                print(f"event in game_input: {event}")'''

        '''if self.agent_movement.frame_stack is None:
            full_game_frame = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_type="PIPELINE"
            ).frames[0]

            self.agent_movement.build_frame_stack(full_game_frame.frame)

        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0],
                frame_type="PIPELINE"
            )

            if self.agent_movement.mode == "TRAIN":

                self.agent_movement.append_to_replay_memory(
                    game_frame_buffer,
                    reward_mouse,
                    terminal=self.game_state["health"] == 0
                )

                # Every 2000 steps, save latest weights to disk
                if self.agent_movement.current_step % 2000 == 0:
                    self.agent_movement.save_model_weights(
                        file_path_prefix="D:/SerpentAI/datasets/phy_movement"
                    )

                # Every 20000 steps, save weights checkpoint to disk
                if self.agent_movement.current_step % 20000 == 0:
                    self.agent_movement.save_model_weights(
                        file_path_prefix="D:/SerpentAI/datasets/phy_movement",
                        is_checkpoint=True
                    )

            elif self.agent_movement.mode == "RUN":
                self.agent_movement.update_frame_stack(game_frame_buffer)

            if self.game_state["hp"][1] <= 0:
                serpent.utilities.clear_terminal()
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.agent_movement.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.agent_movement.mode == "RUN"
                        }
                else:
                    pass

                self.game_state["current_run_steps"] = 0

                self.input_controller.handle_keys([])

                if self.agent_movement.mode == "TRAIN":
                    for i in range(8):

                        self.agent_movement.train_on_mini_batch()

                if self.agent_movement.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        if self.agent_movement.type == "DDQN":
                            self.agent_movement.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.agent_movement.enter_run_mode()
                    else:
                        self.agent_movement.enter_train_mode()

                return None



        self.agent_movement.pick_action()
        self.agent_movement.generate_action()

        keys = self.agent_movement.get_input_values()

        self.input_controller.handle_keys(keys)

        self.agent_movement.erode_epsilon(factor=2)

        self.agent_movement.next_step()'''




    def _measure_hp(self, game_frame):
        heart3 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart3.png')[..., np.newaxis]
        heart2 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart2.png')[..., np.newaxis]
        heart1 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart1.png')[..., np.newaxis]
        sprite_heart3 = Sprite("3 Lifes", image_data=heart3)
        sprite_heart2 = Sprite("2 Lifes", image_data=heart2)
        sprite_heart1 = Sprite("1 Lifes", image_data=heart1)

        sprite_locator = SpriteLocator()

        search3 = sprite_locator.locate(sprite=sprite_heart3, game_frame=game_frame)
        if search3 is not None:
            return 3
            
        else:
            search2 = sprite_locator.locate(sprite=sprite_heart2, game_frame=game_frame)

            if search2 is not None:
                return 2
            else:
                search1 = sprite_locator.locate(sprite=sprite_heart1, game_frame=game_frame)

                if search1 is not None:
                    return 1
                
                else:
                    return 0

    def _measure_score(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Score"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5)

        # OCR may add some strange characters:

        score_clean = re.sub(r'[^0-9]', '', score)

        try:
            score_clean = int(score_clean)
        except ValueError:
            score_clean = 0

        return score_clean

    def _measure_bomb(self, game_frame):
        bomb3 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb3.png')[..., np.newaxis]
        bomb2 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb2.png')[..., np.newaxis]
        bomb1 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb1.png')[..., np.newaxis]
        sprite_bomb3 = Sprite("3 Lifes", image_data=bomb3)
        sprite_bomb2 = Sprite("2 Lifes", image_data=bomb2)
        sprite_bomb1 = Sprite("1 Lifes", image_data=bomb1)

        sprite_locator = SpriteLocator()

        search3 = sprite_locator.locate(sprite=sprite_bomb3, game_frame=game_frame)
        if search3 is not None:
            return 3
            
        else:
            search2 = sprite_locator.locate(sprite=sprite_bomb2, game_frame=game_frame)

            if search2 is not None:
                return 2
            else:
                search1 = sprite_locator.locate(sprite=sprite_bomb1, game_frame=game_frame)

                if search1 is not None:
                    return 1
                
                else:
                    return 0

    def _reward(self, game_state, game_frame):
        if self.game_state['hp'] is None:
            pass
        elif self.game_state['hp'] == 0:
            return -(1000000 - (self.game_state['score'] * self.game_state['bombs']))
        else:
            return (self.game_state['score'] + (self.game_state['hp'] * self.game_state['bombs']))

    def _check_end(self, game_frame):
        next_level = skimage.io.imread('D:/Python/datasets/bullet_heaven_next.png')[..., np.newaxis]
        restart = skimage.io.imread('D:/Python/datasets/bullet_heaven_restart.png')[..., np.newaxis]

        sprite_next = Sprite("Next Level", image_data=next_level)
        sprite_restart = Sprite("Restart Level", image_data=restart)

        sprite_locator = SpriteLocator()

        search_next = sprite_locator.locate(sprite=sprite_next, game_frame=game_frame)
        
        if search_next is not None:
            search_next_x = (search_next[1] + search_next[3])/2
            search_next_y = (search_next[0] + search_next[2])/2

            self.input_controller.move(x=search_next_x, y=search_next_y)
            self.input_controller.click()

            self.game_state['current_run'] += 1
            self.game_state['current_run_steps'] = 0
        
        else:
            search_restart = sprite_locator.locate(sprite=sprite_restart, game_frame=game_frame)

            if search_restart is not None:
                search_restart_x = (search_restart[1] + search_restart[3])/2
                search_restart_y = (search_restart[0] + search_restart[2])/2

                self.input_controller.move(x=search_restart_x, y=search_restart_y)
                self.input_controller.click()

                self.game_state['current_run'] += 1
                self.game_state['current_run_steps'] = 0
            
            else:
                pass
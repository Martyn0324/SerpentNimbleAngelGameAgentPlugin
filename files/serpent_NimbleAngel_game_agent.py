from serpent.game_agent import GameAgent
import serpent.utilities
#from serpent.sprite_locator import SpriteLocator
#from serpent.sprite import Sprite
import numpy as np
from datetime import datetime
from serpent.frame_grabber import FrameGrabber
import skimage.color
import skimage.measure
import serpent.cv
import serpent.ocr

import re

from serpent.input_controller import MouseEvent, MouseEvents, MouseButton

from serpent.config import config

from serpent.machine_learning.reinforcement_learning.agents.rainbow_dqn_agent import RainbowDQNAgent

from serpent.input_controller import KeyboardEvent, KeyboardEvents

from serpent.analytics_client import AnalyticsClient


class Environment:

    def __init__(self, name, game_api=None, input_controller=None):
        self.name = name

        self.game_api = game_api
        self.input_controller = input_controller

        self.analytics_client = AnalyticsClient(project_key=config["analytics"]["topic"])


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
                        if event.event == MouseEvents.MOVE:
                            self.input_controller.move(x=event.x, y=event.y)

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

class InputControlTypes(enum.Enum):
    DISCRETE = 0
    CONTINUOUS = 1

#sprite_locator = SpriteLocator()

class SerpentNimbleAngelGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.game_state = None

        self._reset_game_state()

    def _reset_game_state(self):
        self.game_state = {
            #"hp": 3,
            "score": 0,
            #"bombs": 3,
            "run_reward": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "total_steps": 0,
        }


    def setup_play(self):

        self.game_inputs = [{
            "name" : "Controls",
            "control_type" : InputControlTypes.DISCRETE,
            "inputs" : self.game.api.combine_game_inputs(["MOVEMENT"]),
            "value": None}]

        #print(f"game_inputs1: {self.game_inputs}\n")
        
        # Move mouse = Control Type Continuous. We should make a second agent
        # responsible for moving the mouse.
        
        self.game_inputs2 = [{
            "name": "Mouse",
            "control_type": InputControlTypes.CONTINUOUS,
            "inputs": self.game.api.game_inputs2["MOUSE"],
            "value": 0.02
            }]
        
        #print(f"game_inputs2: {self.game_inputs2}")

        # Rainbow DQN fails to generate inputs for mouse movements (game_inputs2) ---- SOLVED! Added "value" key in self.game_inputs
        # AND in RainbowDQNAgent code.
        # Trying PPO - Fail. Damn Pytorch
        # Using DDQN - Success? Attention to cuDNN though

        # History = number of channels = number of frames
        # If each frame has 3 RGB channels, then history must be 4*3 = 12

        '''self.agent_actions = RainbowDQNAgent('Angel_actions', game_inputs=self.game_inputs, rainbow_kwargs=dict(history=4*3))
        self.agent_mouse = RainbowDQNAgent("Angel_mouse", game_inputs=self.game_inputs2, rainbow_kwargs=dict(history=4*3))'''

        # Trying to avoid RunTimeError for CUDA:

        rainbow_kwargs = dict(
            #save_steps=100,
            observe_steps=10,
            target_update=500
        )

        self.agent_actions = RainbowDQNAgent('Angel_actions', game_inputs=self.game_inputs, rainbow_kwargs=rainbow_kwargs, mode='train')
        self.agent_mouse = RainbowDQNAgent("Angel_mouse", game_inputs=self.game_inputs2, rainbow_kwargs=rainbow_kwargs, mode='train')

        self.agent_actions.current_episode = self.game_state['current_run']
        self.agent_mouse.current_episode = self.agent_actions.current_episode
        
        self.agent_actions.current_step = self.game_state['total_steps']
        self.agent_mouse.current_step = self.agent_actions.current_step

        #self.agent_actions.set_mode(1)
        #self.agent_mouse.set_mode(1)

    def handle_play(self, game_frame):

        # Saving model each N steps: - Causes great delay between actions during saving process
        if self.game_state['total_steps'] % 100 == 0:
            self.agent_actions.save_model()
            self.agent_mouse.save_model()


        start = datetime.now()

        #self.game_state["hp"] = self._measure_hp(game_frame)
        self.game_state["score"] = self._measure_score(game_frame)
        #self.game_state["bombs"] = self._measure_bomb(game_frame)
        
        # While we are testing mouse inputs, we're gonna use reward = 0
        
        #reward_actions = 0
        #reward_mouse = reward_actions

        reward_actions = self._reward(self.game_state)
        reward_mouse = reward_actions
        
        self.agent_actions.observe(reward=reward_actions)
        self.agent_mouse.observe(reward=reward_mouse)
        
        frame_buffer = FrameGrabber.get_frames([0, 2, 4, 6], frame_type="PIPELINE")
        agent_actions = self.agent_actions.generate_actions(frame_buffer)
        #print(f"agent_actions: {agent_actions}")
        #agent_mouse = self.agent_mouse.generate_actions(frame_buffer)
        x = self.agent_mouse.generate_mouse_coordinates(frame_buffer)
        y = self.agent_mouse.generate_mouse_coordinates(frame_buffer)

        #print(f"\n\n X: {x}\n")
        #print(f" Y: {y}")
        #print(f"\nagent_mouse: {agent_mouse}")

        #mouse_actions = self.agent_mouse.generate_mouse_actions(x, 1920, y, 1080)
        mouse_actions = self.agent_mouse.generate_mouse_actions(x, 1530, y, 1020)

        #print(f"\n\nmouse_actions:{mouse_actions}")

        Environment.perform_input(self, actions=mouse_actions)
        Environment.perform_input(self, actions=agent_actions)


        self.game_state['current_run_steps'] += 1
        self.game_state['total_steps'] += 1

        serpent.utilities.clear_terminal()
        #print(f"Current HP: {self.game_state['hp']}")
        print(f"Current Score: {self.game_state['score']}")
        #print(f"Bombs: {self.game_state['bombs']}")
        print(f"Current Reward: {self.game_state['run_reward']}")
        print(f"Current Run: {self.game_state['current_run']}")
        print(f"Current Run Step: {self.game_state['current_run_steps']}")
        print(f"Total Steps: {self.game_state['total_steps']}")
        #print(f"\n\nagent_mouse:\n\n{agent_mouse}\n\n\n")
        print(f"Model Observe mode: {self.agent_mouse.observe_mode}")
        #print(f"current_state shape: {self.agent_mouse.current_state.shape}")

        print(f"model mode: {self.agent_mouse.mode}")
        print(f"model string: {self.agent_mouse.mode_string}")

        #print(f"game_state_steps: {self.game_state['current_run_steps']}")
        print(f"self.agent_steps: {self.agent_mouse.current_step}")
        print(self.agent_mouse.current_episode)
        #print(f"modified run and steps: {self.agent_mouse.current_run}\n")
        #print(self.agent_mouse.current_steps)

        #print(f"\n\n X: {x}\n")
        #print(f" Y: {y}")
        #print(f"\n\nmouse_actions:{mouse_actions}")
        
        self._check_end(game_frame)

        end = datetime.now()

        print(f"Time spent: {end-start}")

        # Let's solve this mouse problem once and for all:

        '''for label, game_input, value in agent_mouse:
            print(f"agent_mouse\nlabel: {label}\ngame_input:{game_input}\nvalue:{value}\n\n")

            for event in game_input:
                print(f"event in game_input: {event}")

        for label, game_input, value in agent_actions:
            print(f"\n\nagent_actions\nlabel: {label}\ngame_input:{game_input}\nvalue:{value}\n\n")

            for event in game_input:
                print(f"event in game_input: {event}")'''

        # Ok, we got the mouse inputs, but it's impracticable.
        # We're not from OpenAI nor Tesla or Google. We need a way to use those inputs
        # Without the need of borrowing an entire building full of TPU Pods.
        #print(f"\n\n X: {x}\n")
        #print(f" Y: {y}")
        #print(f"\n\nmouse_actions:\n\n{mouse_actions}\n\n\n")

    def _measure_score(self, game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Score"])
            
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")
            
        try:
            score = serpent.ocr.perform_ocr(image=score_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 8 -c tessedit_char_whitelist=Oo0123456789S')

            score = score.replace('O', '0')
            score = score.replace('o', '0')
            score = score.replace('S', '5')
        
        except ValueError:
            score = 0
        
        except np.linalg.LinAlgError:
            score = 0
        
        try:
            score = int(score)
        except ValueError:
            score = 0

        return score

    def _reward(self, game_state):
        
        return self.game_state['score']

    def _check_end(self, game_frame):
        restart_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Restart"])
            
        restart_grayscale = np.array(skimage.color.rgb2gray(restart_area_frame) * 255, dtype="uint8")
            
        try:
            string = serpent.ocr.perform_ocr(image=restart_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 8')
        
        except ValueError:
            string = "0"

        # OCR adds some strange characters. Let's remove them.

        string = re.sub(r'[^A-Z|a-z]', '', string)

        if string == 'Restart':
            self.input_controller.move(x=768, y=992)
            self.input_controller.click()

            self.game_state['current_run'] += 1
            self.game_state['current_run_steps'] = 0
        
        else:
            next_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["Next Level"])
            next_grayscale = np.array(skimage.color.rgb2gray(next_area_frame) * 255, dtype="uint8")

            try:
                string = serpent.ocr.perform_ocr(image=next_grayscale, scale=10, order=5, horizontal_closing=10, vertical_closing=5, config='--psm 8')
            
            except ValueError:
                string = "0"

            string = re.sub(r'[^A-Z|a-z]', '', string)

            if string == 'NextLevel':
                self.input_controller.move(x=1326, y=982)
                self.input_controller.click()

                self.game_state['current_run'] += 1
                self.game_state['current_run_steps'] = 0
            
            else:
                pass
        
        return None


# ******CURRENTLY UNUSED (yet interesting) FUNCTIONS******

    '''def _measure_hp(self, game_frame):
        heart3 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart3.png')[..., np.newaxis]
        heart2 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart2.png')[..., np.newaxis]
        heart1 = skimage.io.imread('D:/Python/datasets/bullet_heaven_heart1.png')[..., np.newaxis]
        sprite_heart3 = Sprite("3 Lifes", image_data=heart3)
        sprite_heart2 = Sprite("2 Lifes", image_data=heart2)
        sprite_heart1 = Sprite("1 Lifes", image_data=heart1)

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

    def _measure_bomb(self, game_frame):
        bomb3 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb3.png')[..., np.newaxis]
        bomb2 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb2.png')[..., np.newaxis]
        bomb1 = skimage.io.imread('D:/Python/datasets/bullet_heaven_bomb1.png')[..., np.newaxis]
        sprite_bomb3 = Sprite("3 Bombs", image_data=bomb3)
        sprite_bomb2 = Sprite("2 Bombs", image_data=bomb2)
        sprite_bomb1 = Sprite("1 Bomb", image_data=bomb1)

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


    def _check_end(self, game_frame):
        next_level = skimage.io.imread('D:/Python/datasets/bullet_heaven_next.png')[..., np.newaxis]
        restart = skimage.io.imread('D:/Python/datasets/bullet_heaven_restart.png')[..., np.newaxis]

        sprite_next = Sprite("Next Level", image_data=next_level)
        sprite_restart = Sprite("Restart Level", image_data=restart)

        search_next = sprite_locator.locate(sprite=sprite_next, game_frame=game_frame)
        
        if search_next != None:
            search_next_x = (search_next[1] + search_next[3])/2
            search_next_y = (search_next[0] + search_next[2])/2

            self.input_controller.move(x=search_next_x, y=search_next_y)
            self.input_controller.click()

            self.game_state['current_run'] += 1
            self.game_state['current_run_steps'] = 0
        
        else:
            search_restart = sprite_locator.locate(sprite=sprite_restart, game_frame=game_frame)

            if search_restart != None:
                search_restart_x = (search_restart[1] + search_restart[3])/2
                search_restart_y = (search_restart[0] + search_restart[2])/2

                self.input_controller.move(x=search_restart_x, y=search_restart_y)
                self.input_controller.click()

                self.game_state['current_run'] += 1
                self.game_state['current_run_steps'] = 0
            
            else:
                pass'''

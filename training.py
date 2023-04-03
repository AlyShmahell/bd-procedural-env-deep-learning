import json

import numpy as np
import pygame

from SLAMRobot import SLAMAgent
from utils import Check_Collisions, Game_Object, Room, Agent, ExitException


class Training:
    def __init__(self, env_width, env_heigth, multiplier, environment):
        self._type_to_sprite = None
        self._screen = None
        self._agent = Agent(9999, 9999, 8, 8, 0, 'agent', 90)
        self._objective = Game_Object(9800, 9800, 15, 15, 0, 'objective')
        self._rooms = []
        self._env_width = env_width
        self._env_height = env_heigth
        self._multiplier = multiplier
        self._environment = environment
        self._agent_start_x = 198
        self._agent_start_y = 268
        self._checker = Check_Collisions()
        self._is_agent_looking = False
        self._floor = None

    def run_training(self, render_on=False):
        state_size = 40
        slam_agent = SLAMAgent(state_size, 3)
        speed = 2
        frames = 1500
        for i in range(1, 10001):
            self._screen = pygame.display.set_mode((int(self._env_width), int(self._env_height)))
            done = False
            frame_count = 0
            self.reset_agent()
            self._environment._agent = self._agent
            self._environment._objective = self._objective
            self._environment._floor = self._floor
            self._environment._screen = self._screen
            self._environment.reset_objective()
            state = np.reshape(self._environment.project_segments()[0], [1, state_size, 3])
            last_dist_from_spawn = 0
            random_actions = 0

            score = 0

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            slam_agent.save("test")
                    if event.type == pygame.QUIT:
                        pygame.display.quit()
                        pygame.quit()
                        return
                if render_on:
                    frame_count += 1
                    if frame_count == frames:
                        done = True
                    self._screen.fill((30, 30, 30))
                    action, was_it_random = slam_agent.act(state)
                    if was_it_random:
                        random_actions += 1
                    if action == 0:
                        self._agent.targetRot = (self._agent.targetRot + 45) % 360
                    elif action == 1:
                        self._agent.targetRot = (self._agent.targetRot - 45) % 360
                    elif action == 2:
                        if self._agent.targetRot == 90:
                            self._agent.y -= speed
                        elif self._agent.targetRot == 270:
                            self._agent.y += speed
                        elif self._agent.targetRot == 180:
                            self._agent.x -= speed
                        elif self._agent.targetRot == 0:
                            self._agent.x += speed
                        elif self._agent.targetRot == 45:
                            self._agent.y -= speed
                            self._agent.x += speed
                        elif self._agent.targetRot == 135:
                            self._agent.x -= speed
                            self._agent.y -= speed
                        elif self._agent.targetRot == 225:
                            self._agent.x -= speed
                            self._agent.y += speed
                        elif self._agent.targetRot == 315:
                            self._agent.x += speed
                            self._agent.y += speed

                    self._environment._rooms = self._rooms
                    self._environment._screen = self._screen
                    self._environment.draw_model()
                    pygame.display.update()

                    if self.is_agent_colliding():
                        current_min_distance = state[0][0][1]
                        print("len(state[0]): " + str(len(state[0])))
                        for k in range(len(state[0])):
                            if state[0][k][1] < current_min_distance and not state[0][k][2]:
                                print("state[0][k][1]: " + str(state[0][k][1]))
                                current_min_distance = state[0][k][1]
                        print("current_min_distance: " + str(current_min_distance))
                        print("distances: " + str(state))
                        done = True
                    current_dist_from_spawn = self._checker.point_point_distance(
                        (self._agent.sprite.rect.x, self._agent.sprite.rect.y),
                        (self._agent_start_x, self._agent_start_y))

                    if current_dist_from_spawn > last_dist_from_spawn:
                        last_dist_from_spawn = current_dist_from_spawn
                    else:
                        pass
                    reward = 0
                    if self._agent.sprite.rect.colliderect(self._objective.sprite.rect):
                        self._environment.reset_objective()
                        reward += 1
                    next_state = np.reshape(self._environment.project_segments()[0], [1, state_size, 3])
                    score += reward

                    slam_agent.remember(state, action, reward, next_state, done)
                    state = next_state
                else:
                    frame_count += 1
                    if frame_count == 600:
                        done = True
                    reward = 0
                    action = slam_agent.act(state)
                    if action == 0:
                        self._agent.targetRot = 90
                        self._agent.y -= 2
                    elif action == 1:
                        self._agent.targetRot = 270
                        self._agent.y += 2
                    elif action == 2:
                        self._agent.targetRot = 180
                        self._agent.x -= 2
                    elif action == 3:
                        self._agent.targetRot = 0
                        self._agent.x += 2
                    if self.is_agent_colliding():
                        done = True

                    reward += 1
                    next_state = np.reshape(self._environment.project_segments()[0], [1, state_size, 2])

                    if self._environment.project_segments()[1]:
                        reward += 3
                        new_dist_to_objective = self._checker.point_point_distance((self._agent.x, self._agent.y),
                                                                                   (self._objective.x, self._objective.y))
                        if new_dist_to_objective < dist_to_objective:
                            dist_to_objective = new_dist_to_objective
                            reward += 5
                    slam_agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    if done:
                        break
            print("Episode {}/{}: end at frame {}/{}, score: {}, random: {}%".format(i, 10000, frame_count, frames,
                                                                                     score, int((
                                                                                                            random_actions * 100) / frame_count)))
            
            print("Start _agent replay.")
            try:
                slam_agent.replay(500)
                print("Agent replay completed.")
            except ExitException as e:
                print(e)
                return

        print("Weights saving...")
        slam_agent.save("test")
        print("Weights saving completed.")
        pygame.quit()

    def reset_agent(self):
        self._agent.targetRot = 90
        self._agent.x = self._agent_start_x
        self._agent.y = self._agent_start_y
        self._agent.sprite.rect.x = self._agent_start_x
        self._agent.sprite.rect.y = self._agent_start_y

    def is_agent_colliding(self):
        is_agent_in_room = False
        for room in self._rooms:
            if not room.sprite.rect.contains(self._agent.sprite.rect):
                if self._agent.sprite.rect.colliderect(room.sprite.rect):
                    if room.door.width == 0:
                        if not (self._agent.sprite.rect.y >= room.door.sprite.rect.y and self._agent.sprite.rect.y +
                                self._agent.sprite.rect.height <= room.door.sprite.rect.y + room.door.sprite.rect.height):
                            print("Collision with room with vertical door")
                            return True
                    else:
                        if not (self._agent.sprite.rect.x >= room.door.sprite.rect.x and self._agent.sprite.rect.x +
                                self._agent.sprite.rect.width <= room.door.sprite.rect.x + room.door.sprite.rect.width):
                            print("Collision with room with horizontal door")
                            return True
            else:
                is_agent_in_room = True
                for room_child in room.children:
                    if self._agent.sprite.rect.colliderect(room_child.sprite.rect):
                        print("Collision due to a room's object")
                        return True
                    for child in room_child.children:
                        if self._agent.sprite.rect.colliderect(child.sprite.rect):
                            print("Collision due to an object's object")
                            return True
        if not is_agent_in_room:
            if not self._floor.sprite.rect.contains(self._agent.sprite.rect):
                print("Collision with floor")
                return True
        return False

    def load_model(self, file_path):
        with open("./environments/" + file_path, 'r') as infile:
            json_string = infile.read()

        deserialized_environment_dict = json.loads(json_string)
        room_number = deserialized_environment_dict["roomNumber"]
        floor_dict = deserialized_environment_dict["floor"]
        
        self._env_width = floor_dict["width"] + (8 * room_number * self._multiplier)
        self._env_height = floor_dict["height"] + (8 * room_number * self._multiplier)

        self._screen = pygame.display.set_mode((int(self._env_width), int(self._env_height)))
        self._rooms = []
        self._type_to_sprite = dict(hall=pygame.image.load('textures/hall_texture.png').convert_alpha(),
                                    kitchen=pygame.image.load('textures/kitchen_texture.png').convert_alpha(),
                                    bedroom=pygame.image.load('textures/bedroom_texture.png').convert_alpha(),
                                    bathroom=pygame.image.load('textures/bathroom_texture.png').convert_alpha(),
                                    door=pygame.image.load('textures/door_texture.png').convert_alpha(),
                                    toilet=pygame.image.load('textures/toilet_texture.png').convert_alpha(),
                                    shower=pygame.image.load('textures/shower_texture.png').convert_alpha(),
                                    bed=pygame.image.load('textures/green_bed_texture.png').convert_alpha(),
                                    bedside=pygame.image.load('textures/bedside_texture.png').convert_alpha(),
                                    sofa=pygame.image.load('textures/sofa_texture.png').convert_alpha(),
                                    hall_table=pygame.image.load('textures/hall_table_texture.png').convert_alpha(),
                                    table=pygame.image.load('textures/table_texture.png').convert_alpha(),
                                    chair=pygame.image.load('textures/chair_texture.png').convert_alpha(),
                                    desk=pygame.image.load('textures/desk_texture.png').convert_alpha(),
                                    sink=pygame.image.load('textures/sink_texture.png').convert_alpha(),
                                    wardrobe=pygame.image.load('textures/wardrobe_texture.png').convert_alpha(),
                                    cupboard=pygame.image.load('textures/wardrobe_texture.png').convert_alpha(),
                                    floor=pygame.image.load('textures/floor_texture.png').convert_alpha(),
                                    agent=pygame.image.load('textures/agent_texture_mockup.png').convert_alpha(),
                                    objective=pygame.image.load('textures/objective_texture_mockup.png').convert_alpha())

        floor_sprite = pygame.sprite.Sprite()
        floor_sprite.image = pygame.transform.scale(self._type_to_sprite['floor'],
                                                    (int(floor_dict["width"]), int(floor_dict["height"])))
        floor_sprite.rect = pygame.Rect(floor_dict["x"], floor_dict["y"], floor_dict["width"], floor_dict["height"])
        self._floor = Game_Object(floor_dict["x"], floor_dict["y"], floor_dict["width"], floor_dict["height"],
                                  floor_sprite, 'floor')
        for i in range(0, room_number):
            room_dict = deserialized_environment_dict["R" + str(i)]
            room_sprite = pygame.sprite.Sprite()
            room_sprite.image = pygame.transform.scale(self._type_to_sprite[room_dict["type"]],
                                                       (int(room_dict["width"]), int(room_dict["height"])))
            room_sprite.rect = pygame.Rect(room_dict["x"], room_dict["y"], room_dict["width"], room_dict["height"])
            deserialized_room = Room(room_dict["x"], room_dict["y"], room_dict["width"], room_dict["height"], i,
                                     room_sprite, room_dict["type"])
            door_dict = room_dict["door"]
            door_sprite = pygame.sprite.Sprite()
            if door_dict["width"] != 0:
                door_sprite.image = pygame.transform.scale(pygame.transform.rotate(self._type_to_sprite['door'], 90),
                                                           (int(2.5 * self._multiplier), int(1.0 * self._multiplier)))
            else:
                door_sprite.image = pygame.transform.scale(self._type_to_sprite['door'],
                                                           (int(1.0 * self._multiplier), int(2.5 * self._multiplier)))
            door_sprite.rect = pygame.Rect(door_dict["x"], door_dict["y"], door_dict["width"], door_dict["height"])
            deserialized_room.door = Game_Object(door_dict["x"], door_dict["y"], door_dict["width"],
                                                 door_dict["height"], door_sprite, 'door')
            for child_dict in room_dict["children"]:
                child_rotation = 0
                if child_dict["orientation"] == "W":
                    child_rotation = -90
                elif child_dict["orientation"] == "N":
                    child_rotation = 180
                elif child_dict["orientation"] == "E":
                    child_rotation = 90
                child_sprite = pygame.sprite.Sprite()
                child_sprite.image = pygame.transform.scale(
                    pygame.transform.rotate(self._type_to_sprite[child_dict["type"]], child_rotation),
                    (int(child_dict["width"]), int(child_dict["height"])))
                child_sprite.rect = pygame.Rect(child_dict["x"], child_dict["y"], child_dict["width"],
                                                child_dict["height"])
                deserialized_child = Game_Object(child_dict["x"], child_dict["y"], child_dict["width"],
                                                 child_dict["height"], child_sprite, child_dict["type"])
                deserialized_room.children.append(deserialized_child)

                for childchild_dict in child_dict["children"]:
                    childchild_rotation = 0
                    if childchild_dict["orientation"] == "W":
                        childchild_rotation = -90
                    elif childchild_dict["orientation"] == "N":
                        childchild_rotation = 180
                    elif childchild_dict["orientation"] == "E":
                        childchild_rotation = 90
                    childchild_sprite = pygame.sprite.Sprite()

                    childchild_sprite.image = pygame.transform.scale(
                        pygame.transform.rotate(self._type_to_sprite[childchild_dict["type"]], childchild_rotation),
                        (int(childchild_dict["width"]), int(childchild_dict["height"])))
                    childchild_sprite.rect = pygame.Rect(childchild_dict["x"], childchild_dict["y"],
                                                         childchild_dict["width"], childchild_dict["height"])
                    deserialized_child_child = Game_Object(childchild_dict["x"], childchild_dict["y"],
                                                           childchild_dict["width"], childchild_dict["height"],
                                                           childchild_sprite, childchild_dict["type"])
                    deserialized_child.children.append(deserialized_child_child)
            self._rooms.append(deserialized_room)

        agent_sprite = pygame.sprite.Sprite()
        agent_sprite.image = pygame.transform.scale(self._type_to_sprite['agent'], (int(self._agent.width),
                                                                                    int(self._agent.height)))
        agent_sprite.rect = pygame.Rect(self._agent.x, self._agent.y, self._agent.width, self._agent.height)
        self._agent.sprite = agent_sprite
        self._agent.image = self._agent.sprite.image

        objective_sprite = pygame.sprite.Sprite()
        objective_sprite.image = pygame.transform.scale(self._type_to_sprite['objective'], (int(self._objective.width),
                                                                                            int(self._objective.height)))
        objective_sprite.rect = pygame.Rect(self._objective.x, self._objective.y, self._objective.width,
                                            self._objective.height)
        self._objective.sprite = objective_sprite
        self.multiplier = 1.0

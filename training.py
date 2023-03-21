import json

import numpy as np
import pygame

from SLAMRobot import SLAMAgent
from utils import Check_Collisions, Game_Object, Room, Agent


class Training:
    def __init__(self, env_width, env_heigth, multiplier, environment):
        self.type_to_sprite = None
        self.screen = None
        self.agent = Agent(9999, 9999, 8, 8, 0, 'agent', 90)
        self.objective = Game_Object(9800, 9800, 15, 15, 0, 'objective')
        self.rooms = []
        self.env_width = env_width
        self.env_height = env_heigth
        self.multiplier = multiplier
        self.environment = environment
        self.agent_start_x = 198
        self.agent_start_y = 268
        self.checker = Check_Collisions()
        self.is_agent_looking = False

    def run_training(self):
        state_size = 40
        slam_agent = SLAMAgent(state_size, 3)
        speed = 2
        frames = 1500
        for i in range(1, 10001):
            self.screen = pygame.display.set_mode((int(self.env_width), int(self.env_height)))
            done = False
            frame_count = 0
            self.reset_agent()
            self.environment._agent = self.agent
            self.environment._objective = self.objective
            self.environment._floor = self.floor
            self.environment.screen = self.screen
            self.environment.reset_objective()
            state = np.reshape(self.environment.project_segments()[0], [1, state_size, 3])
            last_dist_from_spawn = 0
            random_actions = 0

            score = 0

            while not done:

                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:
                            slam_agent.save("test")
                render_on = True
                if render_on:
                    frame_count += 1
                    if frame_count == frames:
                        done = True
                    self.screen.fill((30, 30, 30))
                    action, was_it_random = slam_agent.act(state)
                    if was_it_random:
                        random_actions += 1
                    if action == 0:
                        self.agent.targetRot = (self.agent.targetRot + 45) % 360
                    elif action == 1:
                        self.agent.targetRot = (self.agent.targetRot - 45) % 360
                    elif action == 2:
                        if self.agent.targetRot == 90:
                            self.agent.y -= speed
                        elif self.agent.targetRot == 270:
                            self.agent.y += speed
                        elif self.agent.targetRot == 180:
                            self.agent.x -= speed
                        elif self.agent.targetRot == 0:
                            self.agent.x += speed
                        elif self.agent.targetRot == 45:
                            self.agent.y -= speed
                            self.agent.x += speed
                        elif self.agent.targetRot == 135:
                            self.agent.x -= speed
                            self.agent.y -= speed
                        elif self.agent.targetRot == 225:
                            self.agent.x -= speed
                            self.agent.y += speed
                        elif self.agent.targetRot == 315:
                            self.agent.x += speed
                            self.agent.y += speed

                    self.environment._rooms = self.rooms
                    self.environment.screen = self.screen
                    self.environment.draw_model()
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
                    current_dist_from_spawn = self.checker.point_point_distance(
                        (self.agent.sprite.rect.x, self.agent.sprite.rect.y),
                        (self.agent_start_x, self.agent_start_y))

                    if current_dist_from_spawn > last_dist_from_spawn:
                        last_dist_from_spawn = current_dist_from_spawn
                    else:
                        pass
                    reward = 0
                    if self.agent.sprite.rect.colliderect(self.objective.sprite.rect):
                        self.environment.reset_objective()
                        reward += 1
                    next_state = np.reshape(self.environment.project_segments()[0], [1, state_size, 3])
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
                        self.agent.targetRot = 90
                        self.agent.y -= 2
                    elif action == 1:
                        self.agent.targetRot = 270
                        self.agent.y += 2
                    elif action == 2:
                        self.agent.targetRot = 180
                        self.agent.x -= 2
                    elif action == 3:
                        self.agent.targetRot = 0
                        self.agent.x += 2
                    if self.is_agent_colliding():
                        done = True

                    reward += 1
                    next_state = np.reshape(self.environment.project_segments()[0], [1, state_size, 2])

                    if self.environment.project_segments()[1]:
                        reward += 3
                        new_dist_to_objective = self.checker.point_point_distance((self.agent.x, self.agent.y),
                                                                                  (self.objective.x, self.objective.y))
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
            print("Start agent replay.")
            slam_agent.replay(500)
            print("Agent replay completed.")

        print("Weights saving...")
        slam_agent.save("test")
        print("Weights saving completed.")
        pygame.quit()

    def reset_agent(self):
        self.agent.targetRot = 90
        self.agent.x = self.agent_start_x
        self.agent.y = self.agent_start_y
        self.agent.sprite.rect.x = self.agent_start_x
        self.agent.sprite.rect.y = self.agent_start_y

    def is_agent_colliding(self):
        is_agent_in_room = False
        for room in self.rooms:
            if not room.sprite.rect.contains(self.agent.sprite.rect):
                if self.agent.sprite.rect.colliderect(room.sprite.rect):
                    if room.door.width == 0:
                        if not (self.agent.sprite.rect.y >= room.door.sprite.rect.y and self.agent.sprite.rect.y +
                                self.agent.sprite.rect.height <= room.door.sprite.rect.y + room.door.sprite.rect.height):
                            print("Collision with room with vertical door")
                            return True
                    else:
                        if not (self.agent.sprite.rect.x >= room.door.sprite.rect.x and self.agent.sprite.rect.x +
                                self.agent.sprite.rect.width <= room.door.sprite.rect.x + room.door.sprite.rect.width):
                            print("Collision with room with horizontal door")
                            return True
            else:
                is_agent_in_room = True
                for room_child in room.children:
                    if self.agent.sprite.rect.colliderect(room_child.sprite.rect):
                        print("Collision due to a room's object")
                        return True
                    for child in room_child.children:
                        if self.agent.sprite.rect.colliderect(child.sprite.rect):
                            print("Collision due to an object's object")
                            return True
        if not is_agent_in_room:
            if not self.floor.sprite.rect.contains(self.agent.sprite.rect):
                print("Collision with floor")
                return True
        return False

    def load_model(self, file_path):
        with open("./environments/" + file_path, 'r') as infile:
            json_string = infile.read()

        deserialized_environment_dict = json.loads(json_string)
        room_number = deserialized_environment_dict["roomNumber"]

        self.env_width = self.env_width + (8.5 * room_number * self.multiplier)
        self.env_height = self.env_height + (8.5 * room_number * self.multiplier)

        self.screen = pygame.display.set_mode((int(self.env_width), int(self.env_height)))

        self.type_to_sprite = dict(hall=pygame.image.load('textures/hall_texture.png').convert_alpha(),
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

        floor_dict = deserialized_environment_dict["floor"]
        floor_sprite = pygame.sprite.Sprite()
        floor_sprite.image = pygame.transform.scale(self.type_to_sprite['floor'],
                                                    (int(floor_dict["width"]), int(floor_dict["height"])))
        floor_sprite.rect = pygame.Rect(floor_dict["x"], floor_dict["y"], floor_dict["width"], floor_dict["height"])
        self.floor = Game_Object(floor_dict["x"], floor_dict["y"], floor_dict["width"], floor_dict["height"],
                                 floor_sprite, 'floor')
        for i in range(0, room_number):
            room_dict = deserialized_environment_dict["R" + str(i)]
            room_sprite = pygame.sprite.Sprite()
            room_sprite.image = pygame.transform.scale(self.type_to_sprite[room_dict["type"]],
                                                       (int(room_dict["width"]), int(room_dict["height"])))
            room_sprite.rect = pygame.Rect(room_dict["x"], room_dict["y"], room_dict["width"], room_dict["height"])
            deserialized_room = Room(room_dict["x"], room_dict["y"], room_dict["width"], room_dict["height"], i,
                                     room_sprite, room_dict["type"])
            door_dict = room_dict["door"]
            door_sprite = pygame.sprite.Sprite()
            if door_dict["width"] != 0:
                door_sprite.image = pygame.transform.scale(pygame.transform.rotate(self.type_to_sprite['door'], 90),
                                                           (int(2.5 * self.multiplier), int(1.0 * self.multiplier)))
            else:
                door_sprite.image = pygame.transform.scale(self.type_to_sprite['door'],
                                                           (int(1.0 * self.multiplier), int(2.5 * self.multiplier)))
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
                    pygame.transform.rotate(self.type_to_sprite[child_dict["type"]], child_rotation),
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
                        pygame.transform.rotate(self.type_to_sprite[childchild_dict["type"]], childchild_rotation),
                        (int(childchild_dict["width"]), int(childchild_dict["height"])))
                    childchild_sprite.rect = pygame.Rect(childchild_dict["x"], childchild_dict["y"],
                                                         childchild_dict["width"], childchild_dict["height"])
                    deserialized_child_child = Game_Object(childchild_dict["x"], childchild_dict["y"],
                                                           childchild_dict["width"], childchild_dict["height"],
                                                           childchild_sprite, childchild_dict["type"])
                    deserialized_child.children.append(deserialized_child_child)
            self.rooms.append(deserialized_room)

        agent_sprite = pygame.sprite.Sprite()
        agent_sprite.image = pygame.transform.scale(self.type_to_sprite['agent'], (int(self.agent.width),
                                                                                   int(self.agent.height)))
        agent_sprite.rect = pygame.Rect(self.agent.x, self.agent.y, self.agent.width, self.agent.height)
        self.agent.sprite = agent_sprite
        self.agent.image = self.agent.sprite.image

        objective_sprite = pygame.sprite.Sprite()
        objective_sprite.image = pygame.transform.scale(self.type_to_sprite['objective'], (int(self.objective.width),
                                                                                           int(self.objective.height)))
        objective_sprite.rect = pygame.Rect(self.objective.x, self.objective.y, self.objective.width,
                                            self.objective.height)
        self.objective.sprite = objective_sprite

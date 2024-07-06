import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import pygame
from pygame.locals import QUIT
import math
import random
from time import sleep
import numpy as np

class VPushSimulation:
    def __init__(self, object_type='circle', v_angle=np.pi/3, use_gui=True):
        self.v_angle = v_angle
        self.object_type = object_type
        self.use_gui = use_gui
        self.setup()

    def setup(self):
        if self.use_gui:
            # Initialize Pygame
            pygame.init()
            self.width, self.height = 800, 600
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption('Box2D with Pygame - Robot Pushing Object')

            # Colors
            self.colors = {
                'background': (255, 255, 255),
                'robot': (0, 0, 255),
                'circle': (255, 0, 0),
                'table': (128, 128, 0),
                'polygon': (0, 255, 50),
                'goal': (0, 128, 128)
            }

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True)

        # Create the dynamic object
        self.object_position = np.array([random.uniform(-20, 20), random.uniform(20, 40)])
        if self.object_type == 'circle':
            self.circle_rad = 3
            self.object_body = self.create_circle(self.world, self.object_position)
        elif self.object_type == 'polygon':
            self.poly_rad = 3
            vertices = [(-self.poly_rad, -self.poly_rad),
                        (self.poly_rad, -self.poly_rad),
                        (self.poly_rad, self.poly_rad),
                        (-self.poly_rad, self.poly_rad)]
            self.object_body = self.create_polygon(self.world, self.object_position, vertices)

        # Create the goal region
        self.goal_radius = 3
        while True:
            self.goal_position = np.array([random.uniform(-20, 20), random.uniform(20, 40)])
            if np.linalg.norm(self.goal_position - self.object_position) > self.goal_radius * 2:
                break

        # Create a dynamic V-shaped robot
        direction_to_goal = self.goal_position - self.object_position
        self.direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
        pre_position = self.object_position - self.direction_to_goal * 15  # Adjust distance as needed
        self.v_length = 10
        self.object_rad = self.circle_rad if self.object_type == 'circle' else self.poly_rad
        self.thres_v_frame_x = self.v_length * math.cos(self.v_angle / 2) + self.object_rad
        self.thres_v_frame_y = self.v_length * math.sin(self.v_angle / 2) + self.object_rad
        self.robot_body = self.create_v_shape(self.world, pre_position, self.v_length, 1, self.v_angle)
        self.robot_body.position = pre_position
        self.robot_body.angle = math.atan2(self.direction_to_goal[1], self.direction_to_goal[0])

        # Simulation parameters
        self.timeStep = 1.0 / 60
        self.vel_iters, self.pos_iters = 6, 2

    def create_circle(self, world, position):
        body = world.CreateDynamicBody(position=position)
        body.CreateCircleFixture(radius=self.circle_rad, density=0.1, friction=1)
        return body

    def create_polygon(self, world, position, vertices):
        body = world.CreateDynamicBody(position=position)
        body.CreatePolygonFixture(vertices=vertices, density=0.1, friction=1)
        return body

    def create_v_shape(self, world, position, length, thickness, angle):
        # Create a dynamic body for the V-shape
        position = (0,30)
        v_shape_body = world.CreateDynamicBody(position=position)

        # Calculate the vertices for the two rectangles
        vertices1 = [(0, 0),
                     (length * math.cos(angle / 2), length * math.sin(angle / 2)),
                     (length * math.cos(angle / 2) - thickness * math.sin(angle / 2),
                      length * math.sin(angle / 2) + thickness * math.cos(angle / 2)),
                     (-thickness * math.sin(angle / 2), thickness * math.cos(angle / 2))]
        
        # Flip rectangle 1 around x axis to create the rectangle 2
        vertices2 = [(0, 0),
                        (length * math.cos(angle / 2), -length * math.sin(angle / 2)), # TODO: V-shape not symmetric
                        (length * math.cos(angle / 2) - thickness * math.sin(angle / 2),
                        -length * math.sin(angle / 2) - thickness * math.cos(angle / 2)),
                        (-thickness * math.sin(angle / 2), -thickness * math.cos(angle / 2))]

        # Attach the shapes to the body
        v_shape_body.CreatePolygonFixture(vertices=vertices1, density=1, friction=0.6)
        v_shape_body.CreatePolygonFixture(vertices=vertices2, density=1, friction=0.6)

        return v_shape_body

    def to_pygame(self, p):
        """Convert Box2D coordinates to Pygame coordinates."""
        return int(p[0] * 10 + self.width // 2), int(self.height - p[1] * 10) # [-40,40], [0,60]

    def draw(self):
        # Draw goal region
        self.draw_goal_region()

        # Draw robot
        for fixture in self.robot_body.fixtures:
            self.draw_polygon(fixture.shape, self.robot_body, fixture, self.colors['robot'])

        # Draw object
        if self.object_type == 'circle':
            for fixture in self.object_body.fixtures:
                self.draw_circle(fixture.shape, self.object_body, fixture, self.colors['circle'])
        elif self.object_type == 'polygon':
            for fixture in self.object_body.fixtures:
                self.draw_polygon(fixture.shape, self.object_body, fixture, self.colors['polygon'])

    def draw_polygon(self, polygon, body, fixture, color):
        vertices = [(body.transform * v) for v in polygon.vertices]
        vertices = [self.to_pygame(v) for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)

    def draw_circle(self, circle, body, fixture, color):
        position = self.to_pygame(body.position)
        pygame.draw.circle(self.screen, color, position, int(circle.radius * 10))

    def draw_goal_region(self):
        pygame.draw.circle(self.screen, self.colors['goal'], self.to_pygame(self.goal_position), self.goal_radius * 10)

    def eval_robustness(self):
        # Calculate the object position in the robot frame
        object_pos = self.object_body.position
        robot_pos = self.robot_body.position
        robot_angle = self.robot_body.angle
        relative_pos = object_pos - robot_pos
        relative_pos = np.array([relative_pos[0] * math.cos(robot_angle) + relative_pos[1] * math.sin(robot_angle),
                                 -relative_pos[0] * math.sin(robot_angle) + relative_pos[1] * math.cos(robot_angle)])
        
        if abs(relative_pos[1]) < self.thres_v_frame_y and relative_pos[0] > 0:
            robustness = max(0, self.thres_v_frame_x - relative_pos[0])
        else:
            robustness = 0
        return robustness
    
    def run(self, num_episodes=1):
        avg_score = 0
        for i in range(num_episodes):
            self.setup()
            score = self.run_onetime()
            print('Episode %d: %.2f' % (i, score))
            avg_score += score
        avg_score /= num_episodes
        return avg_score
    
    def run_onetime(self):
        running = True
        target_reached = False
        num_steps = 0
        avg_robustness = 0
        while running:
            if self.use_gui:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False

            # Calculate the direction vector from the robot to the object
            object_pos = self.object_body.position
            distance_to_goal = (object_pos - self.goal_position).length
            if distance_to_goal < self.goal_radius:
                target_reached = True
                break

            # Push object towards goal using velocity control with noise
            self.robot_body.linearVelocity = 0.1 * self.direction_to_goal * (distance_to_goal+2*self.goal_radius) + np.random.normal(0, 0.5, 2)

            # Step the world
            self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
            self.world.ClearForces()

            # Evaluate robustness
            rob = self.eval_robustness()
            # Record average robustness every 10 steps
            if num_steps % 10 == 0:
                avg_robustness += rob

            if self.use_gui:
                # Clear screen
                self.screen.fill(self.colors['background'])

                # Draw table and objects
                self.draw()
                # pygame.draw.circle(self.screen, (10, 0, 0), self.to_pygame(pre_position), 5)

                # Flip screen
                pygame.display.flip()

                # Cap the frame rate
                pygame.time.Clock().tick(60)

            num_steps += 1
            if num_steps >= 1000:
                break

        if self.use_gui:
            pygame.quit()

        avg_robustness = 0 if num_steps == 0 else avg_robustness / (num_steps // 10)
        final_score = 0.8 if target_reached else 0
        final_score += avg_robustness * 0.1

        return final_score
        
# # Example usage
# simulation = VPushSimulation('polygon', 1.39777, use_gui=1) # polygon or circle
# final_score = simulation.run(3)

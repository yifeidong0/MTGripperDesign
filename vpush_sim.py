import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody)
import pygame
from pygame.locals import QUIT
import math
import random
from time import sleep
import numpy as np

class Box2DSimulation:
    def __init__(self, v_angle, object_type='circle'):
        self.v_angle = v_angle
        self.object_type = object_type
        self.setup()

    def setup(self):
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
            'table': (128, 128, 128),
            'polygon': (0, 255, 50),
            'goal': (0, 255, 0)
        }

        # Create the world
        self.world = world(gravity=(0, 0), doSleep=True)

        # Create a static table body
        self.table_body = self.world.CreateStaticBody(
            position=(0, -1),
            shapes=polygonShape(box=(50, 1)),
        )

        # Create the dynamic object
        self.object_position = np.array([random.uniform(-20, 20), random.uniform(20, 40)])
        if self.object_type == 'circle':
            self.object_body = self.create_circle(self.world, self.object_position)
        elif self.object_type == 'polygon':
            vertices = [(0, 0), (-2, 3), (5, 1), (1, 5), (-3, 1)]
            self.object_body = self.create_polygon(self.world, self.object_position, vertices)

        # Create the goal region
        self.goal_position = np.array([random.uniform(-20, 20), random.uniform(20, 40)])
        self.goal_radius = 5

        # Create a dynamic V-shaped robot
        direction_to_goal = self.goal_position - self.object_position
        # normalize the np vector
        self.direction_to_goal = direction_to_goal / np.linalg.norm(direction_to_goal)
        pre_position = self.object_position - self.direction_to_goal * 15  # Adjust distance as needed
        self.robot_body = self.create_v_shape(self.world, pre_position, 10, 1, self.v_angle)
        self.robot_body.position = pre_position
        self.robot_body.angle = math.atan2(self.direction_to_goal[1], self.direction_to_goal[0]) 

        # Simulation parameters
        self.timeStep = 1.0 / 60
        self.vel_iters, self.pos_iters = 6, 2


    def create_circle(self, world, position):
        body = world.CreateDynamicBody(position=position)
        body.CreateCircleFixture(radius=3, density=0.1, friction=0.2)
        return body

    def create_polygon(self, world, position, vertices):
        body = world.CreateDynamicBody(position=position)
        body.CreatePolygonFixture(vertices=vertices, density=0.1, friction=0.2)
        return body

    def create_v_shape(self, world, position, length, thickness, angle):
        # Create a dynamic body for the V-shape
        v_shape_body = world.CreateDynamicBody(position=position)

        # Calculate the vertices for the two rectangles
        vertices1 = [(0, 0),
                     (length * math.cos(angle / 2), length * math.sin(angle / 2)),
                     (length * math.cos(angle / 2) - thickness * math.sin(angle / 2),
                      length * math.sin(angle / 2) + thickness * math.cos(angle / 2)),
                     (-thickness * math.sin(angle / 2), thickness * math.cos(angle / 2))]

        vertices2 = [(0, 0),
                     (length * math.cos(-angle / 2), length * math.sin(-angle / 2)),
                     (length * math.cos(-angle / 2) - thickness * math.sin(-angle / 2),
                      length * math.sin(-angle / 2) + thickness * math.cos(-angle / 2)),
                     (-thickness * math.sin(-angle / 2), thickness * math.cos(-angle / 2))]

        # Attach the shapes to the body
        v_shape_body.CreatePolygonFixture(vertices=vertices1, density=1, friction=0.2)
        v_shape_body.CreatePolygonFixture(vertices=vertices2, density=1, friction=0.2)

        return v_shape_body

    def to_pygame(self, p):
        """Convert Box2D coordinates to Pygame coordinates."""
        return int(p[0] * 10 + self.width // 2), int(self.height - p[1] * 10) # [-40,40], [0,60]

    def draw(self):
        # Draw table
        for fixture in self.table_body.fixtures:
            self.draw_polygon(fixture.shape, self.table_body, fixture, self.colors['table'])

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

    def run(self):
        running = True
        target_reached = False
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False

            # Calculate the direction vector from the robot to the object
            # robot_pos = self.robot_body.position
            object_pos = self.object_body.position
            distance_to_goal = (object_pos - self.goal_position).length
            if distance_to_goal < self.goal_radius:
                target_reached = True
                break

            # push object towards goal using velocity control
            self.robot_body.linearVelocity = 0.1 * self.direction_to_goal * distance_to_goal

            # Step the world
            self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
            self.world.ClearForces()

            # Clear screen
            self.screen.fill(self.colors['background'])

            # Draw table and objects
            self.draw()
            # pygame.draw.circle(self.screen, (10, 0, 0), self.to_pygame(pre_position), 5)

            # Flip screen
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(60)

        pygame.quit()
        
# Example usage
simulation = Box2DSimulation(v_angle=math.pi / 3, object_type='circle') # polygon or circle
simulation.run()

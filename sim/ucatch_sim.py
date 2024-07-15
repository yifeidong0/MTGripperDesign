import Box2D
from Box2D.b2 import (world, polygonShape, circleShape, staticBody, dynamicBody, revoluteJoint)
import pygame
from pygame.locals import QUIT
import math
import random
from time import sleep
import numpy as np
from shapely.geometry import Polygon, Point

class UCatchSimulation:
    def __init__(self, object_type='circle', design_param=[5, 5, 5, np.pi/2, np.pi/2], use_gui=True):
        self.object_type = object_type
        self.d0, self.d1, self.d2, self.alpha0, self.alpha1 = design_param
        self.use_gui = use_gui
        self.setup()

    def setup(self, reset_task_and_design=0):
        if not reset_task_and_design:
            if self.use_gui:
                # Initialize Pygame
                pygame.init()
                self.width, self.height = 800, 600
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption('Box2D with Pygame - U Catch')

                # Colors
                self.colors = {
                    'background': (255, 255, 255),
                    'robot': (0, 0, 255),
                    'circle': (255, 0, 0),
                    'polygon': (0, 255, 50),
                    'goal': (0, 128, 128),
                    'table': (128, 128, 0),
                }

            # Create the world
            self.g = 9.81
            self.world = world(gravity=(0, -self.g), doSleep=True)

        # Randomize the object velocity, open-loop control
        self.object_position = [-30, 50]
        self.robot_position = [25, 5]  # Middle rectangle, top left corner
        self.object_velocity_true_vx = random.normalvariate(10, 3)
        self.object_velocity_true = [self.object_velocity_true_vx, 0] # Initial rightward velocity
        self.object_velocity = [self.object_velocity_true_vx+random.normalvariate(0, 3), 0]

        # Compute the robot desired velocity
        self.time_to_fall = math.sqrt(2*(self.object_position[1]-(self.robot_position[1]+5))/self.g) # +5 offset
        self.distance_to_travel = self.robot_position[0] - (self.object_position[0]+self.object_velocity_true_vx*self.time_to_fall) 
        self.robot_desired_vx = -self.distance_to_travel / self.time_to_fall
        
        # Create the dynamic object
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
        self.object_body.linearVelocity = self.object_velocity

        # Create the U-shaped robot gripper
        self.robot_body = self.create_u_shape(self.world, self.robot_position, [self.d0, self.d1, self.d2], [self.alpha0, self.alpha1])

        # Create table
        self.robot_shell_width = 0.5
        self.table_body = self.create_table(self.world, (0, self.robot_position[1]-2*self.robot_shell_width), 80, self.robot_shell_width)

        # Simulation parameters
        self.timeStep = 1.0 / 60
        self.vel_iters, self.pos_iters = 6, 2

    def reset_task_and_design(self, new_task, new_design):
        """Reset the task and design parameters for MTBO or RL training."""
        self.object_type = new_task
        self.d0, self.d1, self.d2, self.alpha0, self.alpha1 = new_design

        # Remove the old object and robot
        self.world.DestroyBody(self.object_body)
        self.world.DestroyBody(self.robot_body)

        self.setup(reset_task_and_design=1)

    def create_table(self, world, position, width, height):
        table = world.CreateStaticBody(position=position)
        vertices = [(-width/2, 0), (width/2, 0), (width/2, height), (-width/2, height)]
        table.CreatePolygonFixture(vertices=vertices, density=1, friction=0.6)
        return table

    def create_circle(self, world, position):
        body = world.CreateDynamicBody(position=position)
        body.CreateCircleFixture(radius=self.circle_rad, density=0.01, friction=1)
        return body

    def create_polygon(self, world, position, vertices):
        body = world.CreateDynamicBody(position=position)
        body.CreatePolygonFixture(vertices=vertices, density=0.01, friction=1)
        return body

    def create_u_shape(self, world, position, lengths, angles, width=.5):
        d0, d1, d2 = lengths
        alpha0, alpha1 = angles

        # Create the main body for the U-shape
        main_body = world.CreateDynamicBody(position=position)

        # Define the vertices for the U-shape parts
        # Rectangle 1 (bottom part of U)
        vertices1 = [(0, 0), (d0, 0), (d0, -width), (0, -width)]

        # Rectangle 2 (left side of U)
        vertices2 = [(d0, 0), 
                     (d0-d1*math.cos(alpha0), d1*math.sin(alpha0)), 
                     (d0-d1*math.cos(alpha0)+width*math.sin(alpha0), d1*math.sin(alpha0)+width*math.cos(alpha0)), 
                     (d0+width*math.sin(alpha0), width*math.cos(alpha0))]

        # # Rectangle 3 (right side of U)
        vertices3 = [(0, 0), 
                     (d2*math.cos(alpha1), d2*math.sin(alpha1)), 
                     (d2*math.cos(alpha1)-width*math.sin(alpha1), d2*math.sin(alpha1)+width*math.cos(alpha1)),
                        (-width*math.sin(alpha1), width*math.cos(alpha1))]
        
        # Attach the shapes to the main body
        main_body.CreatePolygonFixture(vertices=vertices1, density=1, friction=0.6)
        main_body.CreatePolygonFixture(vertices=vertices2, density=1, friction=0.6)
        main_body.CreatePolygonFixture(vertices=vertices3, density=1, friction=0.6)

        return main_body

    def create_rectangle(self, world, position, length, thickness):
        body = world.CreateDynamicBody(position=position)
        vertices = [(0, 0), (length, 0), (length, thickness), (0, thickness)]
        body.CreatePolygonFixture(vertices=vertices, density=1, friction=0.6)
        return body

    def to_pygame(self, p):
        """Convert Box2D coordinates to Pygame coordinates."""
        return int(p[0] * 10 + self.width // 2), int(self.height - p[1] * 10)

    def draw(self):
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

        # Draw table
        for fixture in self.table_body.fixtures:
            self.draw_polygon(fixture.shape, self.table_body, fixture, self.colors['table'])

    def draw_polygon(self, polygon, body, fixture, color):
        vertices = [(body.transform * v) for v in polygon.vertices]
        vertices = [self.to_pygame(v) for v in vertices]
        pygame.draw.polygon(self.screen, color, vertices)

    def draw_circle(self, circle, body, fixture, color):
        position = self.to_pygame(body.position)
        pygame.draw.circle(self.screen, color, position, int(circle.radius * 10))

    def eval_robustness(self):
        # Calculate the object position in the robot frame
        object_pos = self.object_body.position
        thickness = self.poly_rad if self.object_type == 'polygon' else self.circle_rad
        if self.check_end_condition(slack=0.9*thickness):
            robot_y_max = min(self.robot_vertices[2][1], self.robot_vertices[3][1])
            robustness = max(robot_y_max + thickness - object_pos[1], 0.0)
        else:
            robustness = 0
        # print(f"Robustness: {robustness}")
        return robustness
    
    def check_end_condition(self, slack=2):
        """Check if the object is caught by the robot. 
            It means object center is within the quadrilateral formed by the robot.
        """
        object_pos = self.object_body.position
        robot_pos = self.robot_body.position
        d0, d1, d2 = self.d0, self.d1, self.d2
        alpha0, alpha1 = self.alpha0, self.alpha1
        self.robot_vertices = [(robot_pos[0], robot_pos[1]), 
                          (robot_pos[0] + d0, robot_pos[1]),
                          (robot_pos[0] + d0 - d1*math.cos(alpha0), robot_pos[1] + d1*math.sin(alpha0)),
                          (robot_pos[0] + d2*math.cos(alpha1), robot_pos[1] + d2*math.sin(alpha1))]
        self.robot_vertices = np.array(self.robot_vertices)
        object_pos = np.array(object_pos)
        if self.is_point_inside_polygon(object_pos, self.robot_vertices, slack):
            # print("Object caught by the robot!")
            return True
        return False

    def is_point_inside_polygon(self, point, vertices, slack=2):
        polygon = Polygon(vertices)
        point = Point(point)
        if polygon.contains(point):
            return True
        if polygon.distance(point) <= slack:
            return True

        return False
        
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
        num_end_steps = 0
        avg_robustness = 0
        rob_count = 0
        while running:
            if self.use_gui:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        running = False

            # Check if object is caught by the robot
            robot_pos = self.robot_body.position
            if self.check_end_condition():
                num_end_steps += 1
                target_reached = True if num_end_steps >= 100 else False
                if target_reached:
                    break

            # Robot position limits
            if robot_pos[0] < -40 or robot_pos[0] > 40:
                break

            # Move the robot horizontally
            self.robot_body.linearVelocity = [self.robot_desired_vx + random.normalvariate(0,1), 0]  # Move rightward

            # Step the world
            self.world.Step(self.timeStep, self.vel_iters, self.pos_iters)
            self.world.ClearForces()

            # Evaluate robustness
            robustness = self.eval_robustness()
            if num_steps > 130 and num_steps % 10 == 0:
                avg_robustness += robustness
                rob_count += 1

            if self.use_gui:
                # Clear screen
                self.screen.fill(self.colors['background'])

                # Draw table and objects
                self.draw()

                # Flip screen
                pygame.display.flip()

                # Cap the frame rate
                pygame.time.Clock().tick(60)

            num_steps += 1
            if num_steps >= 1000:
                break

        if self.use_gui:
            pygame.quit()

        final_score = 1.0 if target_reached else 0.0
        avg_robustness = 0 if rob_count == 0 else avg_robustness / rob_count
        final_score = final_score + 0.1*avg_robustness
        
        return final_score


if __name__ == "__main__":
    simulation = UCatchSimulation('polygon', [ 7.28517609 ,9.71980838, 9.29659018 ,2.07382565 ,2.30108103], use_gui=True)  # polygon or circle
    for i in range(3):
        task = random.choice(['circle', 'polygon'])
        design = [random.uniform(5, 10), random.uniform(5, 10), random.uniform(5, 10), 
                 random.uniform(np.pi/2, np.pi), random.uniform(np.pi/2, np.pi)]
        simulation.reset_task_and_design(task, design)
        final_score = simulation.run(1)

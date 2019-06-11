import pygame, math, car1D

from pygame.locals import *

SCREEN_COLOR = (36, 156, 126)
METER2PIXEL_w = 0.01
METER2PIXEL_h = 0.01
METER2PIXEL_polar = math.sqrt(METER2PIXEL_w**2+METER2PIXEL_h**2)

class Visualization:

    screen_w = 800
    screen_h = 480

    def __init__(self,car):

        self.screen = pygame.display.set_mode((self.screen_w,self.screen_h), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()
        self.car = car

        pygame.init()
        pygame.display.set_caption("1D Car sim")


    def reset_display(self,car):

        pygame.display.flip()
        state = (0.0,0.0,0.0,0.0,0.0,0.0)

        car_pos = self.car.wheel.get_pos(state)
        pygame.draw.circle(self.screen, self.car.wheel.color, (int(car_pos[0]*METER2PIXEL_w), int(car_pos[2]*METER2PIXEL_h)),
                           int(self.car.wheel.radius*METER2PIXEL_polar))

    def render(self,state,t):

        # keep figure alive
        #  self.clock.tick(60)  #fps

        # x_mouse,y_mouse = pygame.mouse.get_pos()

        # clear the screen
        self.screen.fill(SCREEN_COLOR)

        # update car body pos:
        car_pos = self.car.wheel.get_pos(state)
        pygame.draw.circle(self.screen, self.car.wheel.color, (int(car_pos[0] * METER2PIXEL_w), int(car_pos[1] * METER2PIXEL_h)),
                           int(self.car.wheel.radius * METER2PIXEL_polar))

        # update figure
        pygame.display.update()  # update or flip?


    def get_events(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.terminate()
                stop_simulation = True
            else:
                stop_simulation = False

        return stop_simulation

    def terminate(self):
        pygame.quit()

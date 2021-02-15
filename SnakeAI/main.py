import pygame
import math
import numpy as np
import random

class SnakeEnv():
    WIDTH = 500
    HEIGHT = 500
    ROWS = 20
    TW = WIDTH/ROWS

    # Initiliser
    def __init__(self, frameRate=10):
        self.frameRate = frameRate
        self.ep_count = 0
        self.step_count = 0
        self.apple_count = 30
        self.render = True
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.reset()

        
    # Spawning Random Foods
    def randomFood(self):
        food = [random.randint(0, self.ROWS-1), random.randint(0, self.ROWS-1)]

        for i in self.snake:
            if i == food:
                return self.randomFood()

        return food

    # This is for playing the game yourself.
    def play(self):
        self.reset()
        self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        while True:      
            self.clock.tick(self.frameRate) 
            self.window.fill((0,0,0))

            if self.snake[0][0] in [-1, self.WIDTH/self.TW] or self.snake[0][1]  in [-1, self.HEIGHT/self.TW] or self.snake[0] in self.snake[1:]:
                pygame.quit()
                quit()

            new_head = self.snake[0].copy()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        self.vel = [0, -1]
                    elif event.key == pygame.K_DOWN:
                        self.vel = [0, 1]
                    elif event.key == pygame.K_LEFT:
                        self.vel = [-1, 0]
                    elif event.key == pygame.K_RIGHT:
                        self.vel = [1, 0]

            new_head[0] += self.vel[0]
            new_head[1] += self.vel[1]
            self.snake.insert(0, new_head)

            #print(food)
            append = True

            for food in self.foods:
                # print(self.snake, food)
                if self.snake[0] == food:
                    self.foods.remove(food)
                    self.foods.append(self.randomFood())
                    append = False
            if append:
                self.snake.pop()

            for cell in self.snake:
                # counter +=1
                pygame.draw.rect(self.window, (255,255,255), [cell[0]*self.TW, cell[1]*self.TW, self.TW, self.TW])

            for food in self.foods:
                pygame.draw.rect(self.window, (255,0,0), [food[0]*self.TW, food[1]*self.TW, self.TW, self.TW])

            pygame.display.update()

    # Finding reward, for distance allocations
    def find_distance(self):
        point = [self.snake[0][0] * self.TW, self.snake[0][1] * self.TW]
        lengths = []
        for pt in self.foods:
            p1 = abs(point[0] - pt[0] * self.TW) * abs(point[0] - pt[0] * self.TW)
            p2 = abs(point[1] - pt[1] * self.TW) * abs(point[1] - pt[1] * self.TW)
            lengths.append(math.sqrt(p1 + p2))

        return lengths if len(lengths) < 2 else min(lengths)

    # Resetting
    def reset(self):
        #self.window = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        #self.clock = pygame.time.Clock()
        self.snake = [
            [int(self.ROWS/2), int(self.ROWS/2)],
            [int((self.ROWS/2)-1), int(self.ROWS/2)],
            [int((self.ROWS/2)-2), int(self.ROWS/2)]
        ]

        self.foods = [self.randomFood() for i in range(self.apple_count)]
        self.done = False
        self.vel = [1,0]

        self.temp_dist = math.inf
        self.last_grid = np.zeros((2, self.ROWS, self.ROWS))
        self.grid = np.zeros((2, self.ROWS, self.ROWS))

        self.update_grid()
        return self.grid

    def update_grid(self):
        self.grid = np.zeros((2, self.ROWS, self.ROWS))
        for food in self.foods:
            self.grid[0, food[0], food[1]] = 1

        for snake in self.snake:
            self.grid[1, snake[0], snake[1]] = 1



    #----------Old Code--------------#

    #
    # def findClosest(self):
    #     point = [self.snake[0][0]*self.TW, self.snake[0][1]*self.TW]
    #     lengths = []
    #     for pt in self.foods:
    #         p1 = abs(point[0]-pt[0]*self.TW)*abs(point[0]-pt[0]*self.TW)
    #         p2 = abs(point[1]-pt[1]*self.TW)*abs(point[1]-pt[1]*self.TW)
    #         lengths.append(math.sqrt(p1+p2))
    #
    #     ind = lengths.index(min(lengths))
    #     return self.foods[ind]

    # -------------------------------#


    def step(self, action):
        reward = 0
        done = False
        if not self.done:

            # Alloting the reward, based on whether it went near the food.
            #cur_dist = self.find_distance()[0]
            #if cur_dist < self.temp_dist:
            #    reward = 1
            #elif cur_dist > self.temp_dist:
            #    reward = -1
            #self.temp_dist = cur_dist



            # seeing if we collided or not
            if self.snake[0][0] in [-1, (self.WIDTH/self.TW) - 1] or self.snake[0][1]  in [-1, (self.HEIGHT/self.TW) - 1] or self.snake[0] in self.snake[1:]:
                reward = 0
                self.ep_count += 1
                self.done = True
                done = True

            # Moving the Snake
            new_head = self.snake[0].copy()
            copy_vel = self.vel
            if action == 0:
                self.vel = [0, -1]
            elif action == 1:
                self.vel = [0, 1]
            elif action == 2:
                self.vel = [-1, 0]
            elif action == 3:
                self.vel = [1, 0]
            elif action == 4:
                self.vel = copy_vel

            # Snake Game Logic
            new_head[0] += self.vel[0]
            new_head[1] += self.vel[1]
            self.snake.insert(0, new_head)

            # More Snake Game logic
            append = True
            for food in self.foods:
                if self.snake[0] == food:
                    self.foods.remove(food)
                    self.foods.append(self.randomFood())
                    reward += 3
                    append = False

            if append:
                self.snake.pop()

            # Render
            if self.render:
                self.window.fill((0, 0, 0))
                for cell in self.snake:
                    # counter +=1
                    pygame.draw.rect(self.window, (255,255,255), [cell[0]*self.TW, cell[1]*self.TW, self.TW, self.TW])

                for food in self.foods:
                    pygame.draw.rect(self.window, (255,0,0), [food[0]*self.TW, food[1]*self.TW, self.TW, self.TW])

                pygame.display.update()

            reward = reward if not done else -1
            self.last_grid = self.grid
            # Declaring the inputs to the agent
            
            if not done:
                self.update_grid() 
                return self.grid, reward, done
            else:
                #print('cuased errors me thinks')
                return self.last_grid, reward, done

            return 'alo','balo','jalo'
            
            
if __name__ == "__main__":
    env = SnakeEnv()
    env.play()

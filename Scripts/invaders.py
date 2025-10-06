import random
import pygame
import math
import numpy as np
from enum import Enum
from math import pi, cos, sin, radians, sqrt

import pprint

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BROWN = (139, 69, 19)
AMBER = (255, 191, 0)


# Simulation settings
WIDTH, HEIGHT = 1024, 800
G = 0.1  # Gravitational constant
FPS = 30  # Frames per second
FPS_DEFAULT = 30 #Actual run speed (make higher to get smoother motions)

# Shelter definitions
SHELTER_BLOCKS = 4
SHELTER_WIDTH = 30
SHELTER_THICKNESS = 20
SHELTER_ALTITUDE = HEIGHT - 130

# Invader definitions
INVADER_COUNT_MAX = 20
INVADER_COUNT_INITIAL = 10
INVADER_BODY = 5
INVADER_HEAD = 2
INVADER_MASS = 100
INVADER_FUEL = 40
INVADER_FUEL_USAGE = 0.002
INVADER_NUM_BOMBS = 5
INVADER_SIGMA = 5
INVADER_BOMB_COUNTER = 10
INVADER_SPEED = 3 # Nominal forward speed for lift calculation
INVADER_MAX_THRUST = 10
INVADER_AIR_RESISTANCE = 0.05
INVADER_AIR_RESISTANCE2 = 2
INVADER_LIFT = 3
INVADER_ROLL_LIMIT = 2
INVADER_DIVE_TIMER_MAX = 500
INVADER_TARGETED_TIME = 30  # Frames to stay targeted (for evasive action)
INVADER_TARGET_THRESHOLD = 200  # Distance at which invader is considered targeted
INVADER_JOIN_SWARM_DISTANCE = 100  # Distance to nearest invader to join swarm
INVADER_JOIN_SWARM_NUMBER = 3  # Number of nearby invaders to join swarm
INVADER_SWARM_MAX_TRACK = 5  # Number of nearest invaders to track in swarm
INVADER_SWARM_SEPARATION = 60  # Desired separation distance in swarm
INVADER_SWARM_LENGTH_SCALE = 80  # Length scale for swarm attraction
INVADER_SWARM_STRENGTH = 1

# Bomb definitions
BOMB_MASS = 5
BOMB_AIR_RESISTANCE = 0.05


# Spaceship definitions
SPACESHIP_MASS = 200
SPACESHIP_FIRERATE = 1  # Frames between shots
SPACESHIP_EMPTY_CLIP_TIME = 8  # Frames to recognize empty clip
SPACESHIP_AMMOCLIP = 24
SPACESHIP_RELOAD = 100  # Frames to reload
SPACESHIP_NUM_MISSILE = 200
SPACESHIP_MISSILE_RATE = 20  # Frames between missiles
SPACESHIP_SIZE = 10
SPACESHIP_SPEED = 5
SPACESHIP_THRUST = 2
SPACESHIP_RESISTANCE = 0.2
SPACESHIP_ALTITUDE = 100
SPACESHIP_LIVES = 3

# Factory Definitions
FACTORY_NUM = 2
FACTORY_PRODUCTION_RATE = 1000 # Frames for MISSILE Proction
FACTORY_SHIELD = 3
FACTORY_SHIELD_RESTORE = 50 # Frames to restore shield unit
FACTORY_ALTITUDE = HEIGHT - 40
FACTORY_SIZE = 16
FACTORY_SHIELD_THICKNESS = 3

# Bullet definitions
BULLET_SPEED = 7
BULLET_SCATTER = 3
BULLET_RADIUS = 1
BULLET_NUM_SOUNDS = 8

# MISSILE definitions
MISSILE_THRUST = 0.7
MISSILE_BOOSTER_TIME = 250  # Frames of booster thrust
MISSILE_MASS = 3
MISSILE_AIR_RESISTANCE = 0.01
MISSILE_AIR_RESISTANCE_SIDE = 0.25
MISSILE_RADAR_SCAN_ANGLE = 30  # Degrees
MISSILE_HEAT_SCAN_ANGLE = 15  # Degrees
MISSILE_TURN_RATE = 4  # Degrees per frame
MISSILE_PROXIMITY_RANGE = 30
MISSILE_FRAG_RANGE = 60

EXPLOSION_RADIUS = 5
EXPLOSION_NUM_PARTICLES = 12

# Controller mappings
JOYAXIS_X = 0
BUTTON_A = 0
BUTTON_B = 1
BUTTON_X = 2
BUTTON_Y = 3
BUTTON_R = 10

# Spaceship shape
sspoly = [(0,0),(-SPACESHIP_SIZE,2*SPACESHIP_SIZE),(0,1.4*SPACESHIP_SIZE),(SPACESHIP_SIZE,2*SPACESHIP_SIZE)]
mpoly = [(0,0),(0,-3),(1,-4),(0,-3),(-1,-4),(0,-3)]
fpoly = [(0,0),(3,3),(4,-3),(8,-3),(9,3),(9,15),(-9,15),(-9,3),(-6,0),(-3,3),(0,0)]

# Invader shape: wings and landing gear
iline1 = [(0,0),(2,0),(6,4),(12,4),(6,4),(8,6),(16,6)]
iline2 = [(4,-5),(4,-4),(0,0),(-4,-4),(-4,-5)]
ithick = 2

class STATE(Enum):
    FORMUP = 0
    WALK = 1
    SWARM = 2
    DIVE = 3
    ROLL = 4
    RIGHT = 5
    CRASH = 6

class STATE2(Enum):
    READY = 0
    RELOAD = 1

class TYPE(Enum):
    RADAR = 0
    HEATSEEK = 1
    FRAGMENTATION = 2

class TYPE2(Enum):
    MISSILE_FACTORY = 0
    AUTOCANNON = 1
    SILO = 2

# Set up (global) shelter, smoke, heat, kill and land matrices
shelter = np.zeros((WIDTH, HEIGHT), dtype=int)
smoke = np.zeros((WIDTH, HEIGHT), dtype=float)
heat = np.zeros((WIDTH, HEIGHT), dtype=float)
Nkills = 0
Nland = 0

# Load explosion sound
pygame.mixer.pre_init(44100, -16, 1, 512)
pygame.mixer.init()
explosion_sound = pygame.mixer.Sound("Invaders\\Scripts\\explosion-312361.mp3")
nuke_sound = pygame.mixer.Sound("Invaders\\Scripts\\nuclear-explosion.mp3")
    
bullet_sound_1 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-1.mp3")
bullet_sound_2 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-2.mp3")
bullet_sound_3 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-3.mp3")
bullet_sound_4 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-4.mp3")
bullet_sound_5 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-5.mp3")
bullet_sound_6 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-6.mp3")
bullet_sound_7 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-7.mp3")
bullet_sound_8 = pygame.mixer.Sound("Invaders\\Scripts\\gun-fire-8.mp3")
bullet_sounds = [bullet_sound_1,bullet_sound_2,bullet_sound_3,bullet_sound_4,bullet_sound_5,bullet_sound_6,bullet_sound_7,bullet_sound_8]
empty_clip_sound = pygame.mixer.Sound("Invaders\\Scripts\\empty-magazine.mp3")
reload_sound = pygame.mixer.Sound("Invaders\\Scripts\\reload.mp3")

launch_sound = pygame.mixer.Sound("Invaders\\Scripts\\rocket-launch.mp3")

class Alien:
    def __init__(self):
        self.active = True
        self.state = STATE.FORMUP
        self.x = random.uniform(0, WIDTH)
        self.y = 0
        self.heading = random.uniform(45, 135)
        self.attitude = 0
        self.dattitude = 0
        self.bombs = INVADER_NUM_BOMBS
        self.bomb_counter = 0
        self.swarm_count = 0
        self.swarm_list = [-1 for _ in range(INVADER_SWARM_MAX_TRACK)]
        self.swarm_dist = [0 for _ in range(INVADER_SWARM_MAX_TRACK)]
        self.swarm_x = [0 for _ in range(INVADER_SWARM_MAX_TRACK)]
        self.swarm_y =  [0 for _ in range(INVADER_SWARM_MAX_TRACK)]
        self.holding_altitude = random.uniform(50,200)
        self.speed = 1
        self.dive_counter = 20
        self.fuel = INVADER_FUEL
        self.mass = INVADER_MASS + BOMB_MASS * self.bombs + self.fuel
        self.thrust = 0
        self.targeted = False
        self.targeted_counter = 0
        self.acquired_target = False
        self.acquired_target_xspeed = 0

    def update(self):
        global Nkills,bombs,smoke

        #Update mass
        self.mass = INVADER_MASS + BOMB_MASS * self.bombs + self.fuel

        #Find nearest aliens, add to swarm list
        self.update_swarm()

        # Countdown targeted status
        if self.targeted:
            self.targeted_counter -= 1
            if self.targeted_counter <= 0:
                self.targeted = False
        
        if self.bomb_counter > 0:
            self.bomb_counter -= (FPS/FPS_DEFAULT)

        bomb_flag = False
       
        match self.state:
            case STATE.FORMUP:
                # Enter the screen and form up
                thrust_x = 0
                thrust_y = 0#-self.mass*G*0.5
                    
                if self.y >= self.holding_altitude:
                    self.state = STATE.WALK
                    if self.heading < 90:
                        self.heading = 0
                    else:
                        self.heading = 180
                        
            case STATE.WALK:
                # Walk left and right, occasionally dropping bombs
                thrust_x = self.mass*(G/10)*cos(math.radians(self.heading)) 
                thrust_y = -self.mass*(G - (self.holding_altitude-self.y)*(G/50))

                bomb_flag = True

                if random.random() < 0.001 or (self.targeted and random.random() < 0.05):
                    self.state = STATE.ROLL
                    if self.x > WIDTH*0.8:
                        self.heading = 180
                    elif self.x < WIDTH*0.2:
                        self.heading = 0
                    else:
                        self.heading = random.choice([0,180])
                    if self.heading == 0:
                        self.dattitude = INVADER_ROLL_LIMIT
                    else:
                        self.dattitude = -INVADER_ROLL_LIMIT

                # Count number of nearby invaders
                elif self.swarm_count >= INVADER_JOIN_SWARM_NUMBER:
                    self.state = STATE.SWARM
                        
            case STATE.SWARM:
                thrust_x = 0
                thrust_y = 0
                for n in range(self.swarm_count):
                    thrust = -INVADER_SWARM_STRENGTH*cos(pi*self.swarm_dist[n]/(2*INVADER_SWARM_SEPARATION))*math.exp(-self.swarm_dist[n]/INVADER_SWARM_LENGTH_SCALE)
                    theta = math.atan2(self.swarm_y[n],self.swarm_x[n])
                    thrust_x += thrust*cos(theta)
                    thrust_y += thrust*sin(theta)
                    
                thrust_y -= self.mass*(G - (self.holding_altitude-self.y)*(G/50))

                if self.swarm_count == 0:
                    self.state = STATE.WALK
                    if self.heading > 90 and self.heading < 270:
                        self.heading = 180
                    else:
                        self.heading = 0

            case STATE.ROLL:
                #Roll and lose altitude
                thrust_x = 0
                thrust_y = 0
                if (abs(self.attitude) >= 180-INVADER_ROLL_LIMIT) and (abs(self.attitude) <= 180+INVADER_ROLL_LIMIT):
                    self.state = STATE.DIVE
                    self.dive_counter = random.randint(0,INVADER_DIVE_TIMER_MAX)
                    self.dattitude = 0

            case STATE.DIVE:
                self.dive_counter -= (FPS/FPS_DEFAULT)
                if self.dive_counter <= 0 or self.y >= HEIGHT*(1/2):
                    # drop bomb
                    if self.bombs > 0:
                        bomb = Bomb(self.x,self.y,self.heading,self.speed)
                        bombs.append(bomb)
                        self.bombs -= 1
                        self.state = STATE.RIGHT
                        self.dattitude = random.choice([-1,1])*INVADER_ROLL_LIMIT/2


                if self.acquired_target == False:
                    # pick a target: spaceship or factory
                    target = random.choice(factories + [spaceship])
                    delta = (target.x - self.x)
                    delta_alt = (target.y - self.y)
                    time_to_target = delta_alt/self.speed
                    self.acquired_target = True
                    self.acquired_target_xspeed = 5 * delta/(1+time_to_target)

                if self.acquired_target:
                    thrust_x = (self.acquired_target_xspeed - self.speed*cos(math.radians(self.heading)))*self.mass*0.01
                else:
                    thrust_x = 0


                    
                # Dive towards the ground, then roll and climb back to holding altitude
                #thrust_x = 0
                thrust_y = 0

            case STATE.RIGHT:
                if abs(self.attitude) < INVADER_ROLL_LIMIT:
                    self.state = STATE.WALK
                    self.dattitude = 0 

                bomb_flag = True

                thrust_x = 0#self.mass*(G/10)*cos(math.radians(self.heading))
                thrust_y = -self.mass*(G - (self.holding_altitude-self.y)*(G/50))

            case STATE.CRASH:
                thrust_x = 0
                thrust_y = 0
                self.dattitude += random.gauss(0,0.1)
                leave_trail(self.x,self.y,1,smoke)

        thrust = math.hypot(thrust_x,thrust_y)
        throttle = max(1,thrust/INVADER_MAX_THRUST)
        thrust_x = thrust_x / throttle
        thrust_y = thrust_y / throttle

        # Deplete fuel
        self.fuel -=  math.hypot(thrust_x,thrust_y) * (FPS/FPS_DEFAULT) * INVADER_FUEL_USAGE
        if self.fuel <= 0:
            self.state = STATE.CRASH
            if self.bombs > 0:
                # Drop all remaining bombs
                for _ in range(self.bombs):
                    bomb = Bomb(self.x, self.y, random.uniform(0, 360),self.speed)
                    bombs.append(bomb)
                self.bombs = 0

        # Check for targets
        if bomb_flag and self.bombs > 0 and self.bomb_counter <= 0:
            delta = self.x + self.speed*cos(math.radians(self.heading))*math.sqrt(2*(HEIGHT-self.y)/G) - spaceship.x
            prob = 0.1*math.exp(-0.5*(delta/INVADER_SIGMA)**2)

            for factory in factories:
                delta = self.x + self.speed*cos(math.radians(self.heading))*math.sqrt(2*(HEIGHT-self.y)/G) - factory.x
                prob += 0.1*math.exp(-0.5*(delta/INVADER_SIGMA)**2)

            if random.random() < prob:
                # Drop a bomb
                bomb = Bomb(self.x,self.y,self.heading,self.speed)
                bombs.append(bomb)
                self.bombs -= 1
                self.bomb_counter = INVADER_BOMB_COUNTER

        self.move(thrust_x,thrust_y)

    def move(self,thrust_x,thrust_y):
        global invaders,Nland,Nkills,heat,spaceship,factories

        # Alien physics
        airres_x = -(INVADER_AIR_RESISTANCE2*abs(self.speed) + INVADER_AIR_RESISTANCE)*self.speed*cos(math.radians(self.heading))
        airres_y = -(INVADER_AIR_RESISTANCE2*abs(self.speed) + INVADER_AIR_RESISTANCE)*self.speed*sin(math.radians(self.heading))
        lift_x = INVADER_LIFT*sqrt(INVADER_SPEED**2+self.speed**2)*sin(math.radians(self.attitude))
        lift_y = -INVADER_LIFT*sqrt(INVADER_SPEED**2+self.speed**2)*cos(math.radians(self.attitude))
        force_x = thrust_x + airres_x + lift_x
        force_y = thrust_y + airres_y + lift_y + self.mass*G
        speed_x = self.speed * cos(math.radians(self.heading)) + force_x / self.mass
        speed_y = self.speed * sin(math.radians(self.heading)) + force_y / self.mass
        self.heading = math.degrees(math.atan2(speed_y,speed_x))
        self.speed = math.hypot(speed_x,speed_y)
        self.x += (FPS/FPS_DEFAULT) * self.speed * cos(math.radians(self.heading))
        self.y += (FPS/FPS_DEFAULT) * self.speed * sin(math.radians(self.heading))
        self.attitude += (FPS/FPS_DEFAULT) * self.dattitude
        self.attitude = self.attitude % 360

        # Detect collision with spaceship
        if math.hypot(self.x - spaceship.x, self.y - spaceship.y) < INVADER_BODY:
            # Hit an invader
            spaceship.active = False
            self.active = False
            Nkills += 1
            pygame.mixer.Sound.play(explosion_sound)

        # Detect edge of the screen and change direction
        if self.x < 0 or self.x >= WIDTH:
            if self.state == STATE.WALK:
                self.heading = (self.heading + 180) % 360
                self.x = max(0, min(WIDTH-1, self.x))
            else:
                self.active = False
                if self.state == STATE.CRASH:
                    Nkills += 1
                return
            
        if self.y >= FACTORY_ALTITUDE:
            for factory in factories:
                if math.hypot(self.x - factory.x, self.y - factory.y) < FACTORY_SHIELD_THICKNESS*factory.shield+FACTORY_SIZE:
                    # Hit the factory
                    self.active = False
                    Nkills += 1
                    factory.hit()
                    return

        # Detect ground impact
        if self.y >= HEIGHT:
            self.active = False
            if self.state != STATE.CRASH:
                Nland += 1
            else:
                Nkills += 1
            return

        self.thrust = math.hypot(thrust_x,thrust_y)
        if self.thrust > 0:
            leave_trail(self.x,self.y,self.thrust,heat)

    def update_swarm(self):
        # This function updates the swarm list with the nearest aliens
        global invaders
        distances = []
        swarm_count = 0 
        for i, invader in enumerate(invaders):
            if invader is not self:
                dist_x = invader.x - self.x
                dist_y = invader.y - self.y
                dist = math.hypot(dist_x,dist_y)
                distances.append((dist,i,dist_x,dist_y))
                if dist < INVADER_JOIN_SWARM_DISTANCE:
                    swarm_count += 1
        distances.sort()
        self.swarm_count = min(swarm_count,INVADER_SWARM_MAX_TRACK)
        self.swarm_list = [idx for _,idx,_,_ in distances[:INVADER_SWARM_MAX_TRACK]]
        self.swarm_dist = [dist for dist,_,_,_ in distances[:INVADER_SWARM_MAX_TRACK]]
        self.swarm_x = [dx for _,_,dx,_ in distances[:INVADER_SWARM_MAX_TRACK]]    
        self.swarm_y = [dy for _,_,_,dy in distances[:INVADER_SWARM_MAX_TRACK]]

    def hit(self):
        global Nkills,invaders,smoke

        if random.random() < (0.2+self.bombs/(2*INVADER_NUM_BOMBS)):
            self.active = False
            Nkills += 1
            pygame.mixer.Sound.play(explosion_sound)

            # Explosion effect
            for _ in range(EXPLOSION_NUM_PARTICLES):
                dx = random.gauss(self.x,EXPLOSION_RADIUS)
                dy = random.gauss(self.y,EXPLOSION_RADIUS)
                smoke[int(dx)%WIDTH,int(dy)%HEIGHT] += 1
            
        else:
            self.state = STATE.CRASH
            self.dattitude = random.uniform(-INVADER_ROLL_LIMIT,INVADER_ROLL_LIMIT)
            if self.bombs > 0:
                # Drop all remaining bombs
                for _ in range(self.bombs):
                    bomb = Bomb(self.x, self.y, random.uniform(0, 360), random.uniform(1, 3))
                    bombs.append(bomb)
                self.bombs = 0

    def draw(self, display):
        match self.state:
            case STATE.FORMUP:
                color = GREEN
            case STATE.WALK:
                color = YELLOW
            case STATE.DIVE:
                color = RED
            case STATE.ROLL:
                color = BLUE
            case STATE.CRASH:
                color = BROWN
            case STATE.SWARM:
                color = AMBER
            case STATE.RIGHT:
                color = BLUE
                
        pygame.draw.circle(display,color,(self.x,self.y),INVADER_BODY)
        # Rotate invader shape
        c, s = cos(math.radians(self.attitude-180)), sin(math.radians(self.attitude-180))
        R = np.array(((c, -s), (s, c)))
        riline1 = [(px*R[0,0]+py*R[0,1],px*R[1,0]+py*R[1,1]) for px, py in iline1]
        pygame.draw.lines(display,WHITE,False,[(self.x + px, self.y + py) for px, py in riline1],ithick)
        riline1 = [(-px*R[0,0]+py*R[0,1],-px*R[1,0]+py*R[1,1]) for px, py in iline1]
        pygame.draw.lines(display,WHITE,False,[(self.x + px, self.y + py) for px, py in riline1],ithick)
        riline2 = [(px*R[0,0]+py*R[0,1],px*R[1,0]+py*R[1,1]) for px, py in iline2]
        pygame.draw.lines(display,WHITE,False,[(self.x + px, self.y + py) for px, py in riline2],2)

        if self.targeted:
            pygame.draw.circle(display,RED,(self.x,self.y),INVADER_BODY+2,1)


class Bomb:
    def __init__(self,x,y,heading,speed):
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = speed
        self.active = True

    def update(self):
        global spaceship,shelter,Nkills,factories

        # Bomb physics
        force_x = -BOMB_AIR_RESISTANCE * self.speed * cos(math.radians(self.heading))
        force_y = BOMB_MASS * G - BOMB_AIR_RESISTANCE * self.speed * sin(math.radians(self.heading))
        speed_x = self.speed * cos(math.radians(self.heading)) + force_x / BOMB_MASS
        speed_y = self.speed * sin(math.radians(self.heading)) + force_y / BOMB_MASS
        self.heading = math.degrees(math.atan2(speed_y, speed_x))
        self.speed = math.hypot(speed_x, speed_y)
        self.x += (FPS/FPS_DEFAULT) * self.speed * cos(math.radians(self.heading))
        self.y += (FPS/FPS_DEFAULT) * self.speed * sin(math.radians(self.heading))

        if self.y >= HEIGHT or self.x < 0 or self.x >= WIDTH:
            self.active = False
            return
        
        if shelter[int(self.x),int(self.y)] == 1:
            # Hit the shelter, damage it
            for dx in range(-SHELTER_THICKNESS//2, SHELTER_THICKNESS//2):
                for dy in range(-SHELTER_THICKNESS//2, SHELTER_THICKNESS//2):
                    if 0 <= int(self.x)+dx < WIDTH and 0 <= int(self.y)+dy < HEIGHT:
                        shelter[int(self.x)+dx, int(self.y)+dy] = 0
            self.active = False
            return
        
        if math.hypot(self.x - spaceship.x, self.y - spaceship.y) < SPACESHIP_SIZE:
            # Hit the spaceship
            self.active = False
            spaceship.active = False
            return
        
        if self.y >= FACTORY_ALTITUDE:
            for factory in factories:
                if math.hypot(self.x - factory.x, self.y - factory.y) < FACTORY_SHIELD_THICKNESS*factory.shield+FACTORY_SIZE:
                    # Hit the factory
                    self.active = False
                    factory.hit()

    def draw(self, display):
        pygame.draw.circle(display, RED, (int(self.x), int(self.y)), 3)
                            
class Spaceship:
    def __init__(self):
        self.active = True
        self.state = STATE2.READY
        self.x = WIDTH // 2
        self.y = HEIGHT - SPACESHIP_ALTITUDE
        self.speed = 0
        self.thrust = 0
        self.ammo = SPACESHIP_AMMOCLIP
        self.reload_counter = 0
        self.firerate_counter = 0
        self.missiles = SPACESHIP_NUM_MISSILE
        self.missilerate_counter = SPACESHIP_MISSILE_RATE
    
    def update(self):
        if self.firerate_counter > 0:
            self.firerate_counter -= (FPS/FPS_DEFAULT)
        if self.missilerate_counter > 0:
            self.missilerate_counter -= (FPS/FPS_DEFAULT)

        match self.state:
            case STATE2.READY:
                if self.ammo == 0:
                    self.state = STATE2.RELOAD
                    self.reload_counter = SPACESHIP_RELOAD  # Frames to reload
            case STATE2.RELOAD:
                self.reload_counter -= 1 
                if self.reload_counter <= 0:
                    self.state = STATE2.READY
                    self.ammo = SPACESHIP_AMMOCLIP
                    pygame.mixer.Sound.play(reload_sound)
        
        self.speed += (self.thrust - SPACESHIP_RESISTANCE*self.speed) * (FPS/FPS_DEFAULT)
        self.speed = max(-SPACESHIP_SPEED, min(SPACESHIP_SPEED, self.speed))
        self.thrust = 0  # Reset thrust after move
        self.x += (FPS/FPS_DEFAULT) * self.speed
        self.x = max(0, min(WIDTH - 1, self.x))

    def draw(self, display):
        pygame.draw.polygon(display,WHITE,[(self.x + px, self.y + py) for px, py in sspoly])


class Bullet:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed =  BULLET_SPEED
        self.active = True
        self.heading = random.gauss(270,BULLET_SCATTER) # Straight up


    def update(self):
        global shelter,Nkills,invaders
        self.x += (FPS/FPS_DEFAULT) * self.speed * cos(math.radians(self.heading))
        self.y += (FPS/FPS_DEFAULT) * self.speed * sin(math.radians(self.heading))
        if self.y < 0:
            self.active = False
        else:
            # Check for collision with invaders
            for invader in invaders:
                if math.hypot(self.x - invader.x, self.y - invader.y) < INVADER_BODY:
                    # Hit an invader
                    self.active = False
                    invader.hit()

    def draw(self, display):
        pygame.draw.circle(display, GREEN, (int(self.x), int(self.y)), BULLET_RADIUS)



class MISSILE:
    def __init__(self, x, y,initial_speed_x,type):
        self.active = True
        self.type = type
        self.x = x
        self.y = y
        self.speed = abs(initial_speed_x)
        if initial_speed_x >= 0:
            self.heading = 0
        elif initial_speed_x < 0:
            self.heading = 180
        self.attitude = 270 # Straight up
        self.booster_counter = MISSILE_BOOSTER_TIME

        # Play launch sound
        pygame.mixer.Sound.play(launch_sound)
       

    def update(self):
        global shelter,Nkills,invaders,smoke,detonations
        
        match self.type:
            case TYPE.RADAR:
                # Update radar guided targeting
                max_radar_signal = 0
                target_invader = False
                for invader in invaders:
                    dist = math.hypot(self.x - invader.x, self.y - invader.y)
                    heading = math.degrees(math.atan2(invader.y - self.y, invader.x - self.x))
                    dheading = (heading - self.attitude + 180) % 360 - 180
                    if abs(dheading) < MISSILE_RADAR_SCAN_ANGLE:  # Only consider invaders roughly in front
                        signal = invader.mass / (dist**2 + 1)  # Simple signal strength model
                        if signal > max_radar_signal:
                            max_radar_signal = signal
                            target_invader = True
                            target_dheading = dheading
                        if (1/(dist**2 + 1)) >= (1/(INVADER_TARGET_THRESHOLD**2)):
                            invader.targeted = True
                            invader.targeted_counter = INVADER_TARGETED_TIME

            case TYPE.HEATSEEK:
                # Update heat seeking targeting
                max_heat_signal = 0
                target_invader = False
                for invader in invaders:
                    dist = math.hypot(self.x - invader.x, self.y - invader.y)
                    heading = math.degrees(math.atan2(invader.y - self.y, invader.x - self.x))
                    dheading = (heading - self.attitude + 180) % 360 - 180
                    if abs(dheading) <= MISSILE_HEAT_SCAN_ANGLE:  # Only consider invaders roughly in front
                        signal = invader.thrust / (dist**2 + 1)  # Simple signal strength model
                        if signal > max_heat_signal:
                            max_heat_signal = signal
                            target_invader = True
                            target_dheading = dheading

            case TYPE.FRAGMENTATION:
                target_invader = False          
                # Proximity detonation
                for invader in invaders:
                    dist = math.hypot(self.x - invader.x, self.y - invader.y)
                    if dist <= MISSILE_PROXIMITY_RANGE:
                        detonation = Detonation(self.x,self.y,MISSILE_FRAG_RANGE)
                        detonations.append(detonation)
                        self.active = False
                        pygame.mixer.Sound.play(nuke_sound)
                        return

        if target_invader:
            # Adjust heading towards target invader
            target_dheading *= (FPS/FPS_DEFAULT)
            self.attitude += max(-MISSILE_TURN_RATE, min(MISSILE_TURN_RATE, target_dheading))
            self.attitude = self.attitude % 360

        if self.booster_counter > 0:
            self.booster_counter -= (FPS/FPS_DEFAULT)
            thrust_x = cos(math.radians(self.attitude)) * MISSILE_THRUST
            thrust_y = sin(math.radians(self.attitude)) * MISSILE_THRUST
            thrust = math.hypot(thrust_x,thrust_y)
            leave_trail(self.x,self.y,thrust,heat)
        else:
            thrust_x = 0
            thrust_y = 0

        # MISSILE physics
        airres_x = (abs(cos(radians(self.heading-self.attitude)))*MISSILE_AIR_RESISTANCE + abs(sin(radians(self.heading-self.attitude)))*MISSILE_AIR_RESISTANCE_SIDE)*self.speed*cos(math.radians(self.heading))
        airres_y = (abs(cos(radians(self.heading-self.attitude)))*MISSILE_AIR_RESISTANCE + abs(sin(radians(self.heading-self.attitude)))*MISSILE_AIR_RESISTANCE_SIDE)*self.speed*sin(math.radians(self.heading))
        force_x = thrust_x - airres_x
        force_y = MISSILE_MASS * G + thrust_y - airres_y
        speed_x = self.speed * cos(math.radians(self.heading)) + force_x / MISSILE_MASS
        speed_y = self.speed * sin(math.radians(self.heading)) + force_y / MISSILE_MASS
        self.heading = math.degrees(math.atan2(speed_y, speed_x))
        self.speed = math.hypot(speed_x, speed_y)
        self.x += (FPS/FPS_DEFAULT) * self.speed * cos(math.radians(self.heading))
        self.y += (FPS/FPS_DEFAULT) * self.speed * sin(math.radians(self.heading))

        if self.y < 0 or self.y >= HEIGHT or self.x < 0 or self.x >= WIDTH:
            self.active = False
            return

        # Check for collision with invaders
        for invader in invaders:
            if math.hypot(self.x - invader.x, self.y - invader.y) < INVADER_BODY:
                # Hit an invader
                self.active = False
                invader.hit()

    def draw(self, display):
        # Rotate missile shape
        c, s = cos(math.radians(self.attitude-90)), sin(math.radians(self.attitude-90))
        R = np.array(((c, -s), (s, c)))
        rmpoly = [(px*R[0,0]+py*R[0,1],px*R[1,0]+py*R[1,1]) for px, py in mpoly]
        pygame.draw.polygon(display,WHITE,[(self.x + px, self.y + py) for px, py in rmpoly])

class Factory:
    def __init__(self,type):
        self.active = True
        self.type = type
        self.x = random.randint(20,WIDTH-20)
        self.y = FACTORY_ALTITUDE
        self.missile_counter = FACTORY_PRODUCTION_RATE
        self.shield = FACTORY_SHIELD
        self.shield_counter = 0

    def update(self):
        global spaceship

        if self.shield_counter > 0:
            self.shield_counter -= (FPS/FPS_DEFAULT)
            if self.shield_counter <= 0:
                self.shield += 1
                if self.shield < FACTORY_SHIELD:
                    self.shield_counter = FACTORY_SHIELD_RESTORE

        match self.type:
            case TYPE2.MISSILE_FACTORY:
                if self.missile_counter > 0:
                    self.missile_counter -= (FPS/FPS_DEFAULT)
                if self.missile_counter <= 0:
                    spaceship.missiles += 1
                    self.missile_counter = FACTORY_PRODUCTION_RATE
    
    def hit(self):
        if self.shield > 0:
            self.shield -= 1
            self.shield_counter = FACTORY_SHIELD_RESTORE
        else:
            self.active = False
        return

    def draw(self,display):
        # Factory outline
        pygame.draw.polygon(display,WHITE,[(self.x + px, self.y + py) for px, py in fpoly],width=1)
        # Shield
        for n in range(self.shield):
            pygame.draw.circle(display,GREEN,(self.x,self.y+6),FACTORY_SHIELD_THICKNESS*n+FACTORY_SIZE,1)
        match self.type:
            case TYPE2.MISSILE_FACTORY:
                pygame.draw.polygon(display,WHITE,[(self.x + px, self.y - py + 6) for px, py in mpoly],width=1)

class Detonation:
    def __init__(self,x,y,range):
        self.active = True
        self.x = x
        self.y = y
        self.range = range
        self.inflate = 0

    def update(self):
        self.inflate += 3*(FPS/FPS_DEFAULT)
        for invader in invaders:
            dist = math.hypot(self.x - invader.x, self.y - invader.y)
            if dist <= self.inflate:
                invader.hit()

        if self.inflate >= self.range:
            self.active = False
        
    def draw(self,display):
        if self.inflate % 2 == 0:
            pygame.draw.circle(display,RED,(self.x,self.y),self.inflate)
        else:
            pygame.draw.circle(display,WHITE,(self.x,self.y),self.inflate)

def leave_trail(x, y, magnitude,smoke):
    
    # Leave a smoke trail
    fx0 = int(x)
    fx1 = min(fx0 + 1, WIDTH - 1)
    fy0 = int(y)
    fy1 = min(fy0 + 1, HEIGHT - 1)
    dx = x - fx0
    dy = y - fy0
    # weight the smoke marker trail lay on four matrix cells near the crashing invader
    A00 = (1 - dx) * (1 - dy)
    A01 = (1 - dx) * dy
    A10 = dx * (1 - dy)
    A11 = dx * dy

    smoke[fx0,fy0] += A00 * magnitude
    smoke[fx0,fy1] += A01 * magnitude
    smoke[fx1,fy0] += A10 * magnitude               
    smoke[fx1,fy1] += A11 * magnitude


def main():

    global shelter,invaders,Nkills,bombs,spaceship,smoke,heat,factories,detonations
    
    pygame.init()
    display = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Space Invaders")

    pygame.joystick.init()
    controller = pygame.joystick.Joystick(0)
    controller.init()


    # Create spaceship
    Nlives = SPACESHIP_LIVES
    spaceship = Spaceship()
    missiles = []
    bullets = []
    detonations = []

    # Create factories
    factories = [Factory(TYPE2.MISSILE_FACTORY) for _ in range(FACTORY_NUM)]

    # Create invaders
    invaders = [Alien() for _ in range(INVADER_COUNT_INITIAL)]
    bombs = []

    # Build Shelters
    for n in range(SHELTER_BLOCKS):
        x = (n+1)*WIDTH//(SHELTER_BLOCKS+1)
        y = SHELTER_ALTITUDE
        shelter[x-SHELTER_WIDTH//2:x+SHELTER_WIDTH//2, y:y+SHELTER_THICKNESS] = 1

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            #if event.type == pygame.JOYAXISMOTION:
                #axis_data[event.axis] = round(event.value,2)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            spaceship.thrust = -SPACESHIP_THRUST
        if keys[pygame.K_RIGHT]:
            spaceship.thrust = SPACESHIP_THRUST
        if controller.get_axis(JOYAXIS_X) != 0:
            spaceship.thrust = SPACESHIP_THRUST * controller.get_axis(0)
        if keys[pygame.K_SPACE] or controller.get_button(BUTTON_A):
            # Fire a bullet
            if spaceship.ammo > 0 and spaceship.firerate_counter <=   0:
                bullet = Bullet(spaceship.x, spaceship.y)
                bullets.append(bullet)
                spaceship.ammo -= 1             
                spaceship.firerate_counter = SPACESHIP_FIRERATE
                pygame.mixer.Sound.play(bullet_sounds[random.randint(0,BULLET_NUM_SOUNDS-1)])
            elif spaceship.ammo == 0 and spaceship.firerate_counter <= 0:
                pygame.mixer.Sound.play(empty_clip_sound)
                spaceship.firerate_counter = SPACESHIP_EMPTY_CLIP_TIME
        if controller.get_axis(5) > 0.5:
            # Reload clip
            spaceship.ammo == 0
        if keys[pygame.K_m] or controller.get_button(BUTTON_B):
            # Fire a heat seeking missile
            if spaceship.missiles > 0 and spaceship.missilerate_counter <= 0:
                missile = MISSILE(spaceship.x,spaceship.y,spaceship.speed,TYPE.HEATSEEK)
                missiles.append(missile)
                spaceship.missiles -= 1 
                spaceship.missilerate_counter = SPACESHIP_MISSILE_RATE
        if keys[pygame.K_n] or controller.get_button(BUTTON_Y):
            # Fire a radar guided missile
            if spaceship.missiles > 0 and spaceship.missilerate_counter <= 0:
                missile = MISSILE(spaceship.x,spaceship.y,spaceship.speed,TYPE.RADAR)
                missiles.append(missile)
                spaceship.missiles -= 1 
                spaceship.missilerate_counter = SPACESHIP_MISSILE_RATE
        if keys[pygame.K_b] or controller.get_button(BUTTON_X):
            # Fire a proximity detonation missile
            if spaceship.missiles > 0 and spaceship.missilerate_counter <= 0:
                missile = MISSILE(spaceship.x,spaceship.y,spaceship.speed,TYPE.FRAGMENTATION)
                missiles.append(missile)
                spaceship.missiles -= 1 
                spaceship.missilerate_counter = SPACESHIP_MISSILE_RATE
        
        #pprint.pprint(controller.get_button(0))        
        
        # Blank screen
        display.fill((255, 255, 255))

        # Update shelter display
        map = np.zeros((WIDTH, HEIGHT, 3), dtype=np.uint8)
        map[:, :, 0] = 50*heat  # Red channel
        map[:, :, 1] = 255*shelter  # Gre    en channel
        map[:, :, 2] = 255*smoke  # Blue channel
        surf = pygame.surfarray.make_surface(map) 

        display.blit(surf, (0, 0))
          
        # Update and draw aliens
        for invader in invaders:
            invader.update()
            invader.draw(display)
        invaders = [invader for invader in invaders if invader.active]  

        if len(invaders) < INVADER_COUNT_MAX and random.random() < 0.005:
            invader = Alien()
            invaders.append(invader)

        if len(invaders) == 0:
            # All invaders destroyed, start a new wave
            invaders = [Alien() for _ in range(INVADER_COUNT_INITIAL)]

        # Update and draw bombs
        for bomb in bombs:
            bomb.update()
            if bomb.active:
                bomb.draw(display)
        bombs = [bomb for bomb in bombs if bomb.active]

        # Update and draw spaceship
        if spaceship.active:    
            spaceship.update()
            spaceship.draw(display)
        else:
            Nlives -= 1
            spaceship = Spaceship()

        if Nlives <= 0:
            #Game Over
            running = False

        for bullet in bullets:
            bullet.update()
            if bullet.active:
                bullet.draw(display)
        bullets = [bullet for bullet in bullets if bullet.active]

        for missile in missiles:
            missile.update()
            if missile.active:
                missile.draw(display)
        missiles = [missile for missile in missiles if missile.active]

        for detonation in detonations:
            detonation.update()
            if detonation.active:
                detonation.draw(display)
        detonations = [detonation for detonation in detonations if detonation.active]
    
        for factory in factories:
            factory.update()
            if factory.active:
                factory.draw(display)
        factories = [factory for factory in factories if factory.active]

        fps = clock.get_fps()

        pygame.display.update()
        pygame.display.set_caption("Space Invaders, Nlives: {}, Kills: {}, Landed: {}, FPS: {}, Ammo: {}, Missiles: {}.".format(Nlives,Nkills,Nland,int(fps),spaceship.ammo,spaceship.missiles))

        clock.tick(FPS_DEFAULT)

        smoke *= 0.995  # Decay smoke     
        heat *= 0.95  # Decay heat 


    pygame.quit()

if __name__ == "__main__":
    main()


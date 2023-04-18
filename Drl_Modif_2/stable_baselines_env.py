import gym
from gym import spaces
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import copy
import rospy
import subprocess
from os import path
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from numpy import inf
import numpy as np
import random
import math
from gazebo_msgs.msg import ModelState
from squaternion import Quaternion
from std_msgs.msg import Float64
from sensor_msgs.msg import LaserScan, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_srvs.srv import Empty
import os
import time
from random import *
from position_case import type_case
from Dynamic_Approach.traject3d import evitement
from Gridy_drone_swipp.Gridy_based import scan
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

COLLISION_DIST = 0.035
GOAL_REACHED_DIST = 0.03



class UnityEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, launchfile, max_episode_length=256):

        #init des different parametre comme positon
        #goal, batterie ect..
        self._max_episode_length = max_episode_length
        self.flag_change_goal=False
        self.flag_change_bat=False
        self.step_counter = 0
        self.position=[0,0,20]
        self.position_depart=[0,0,20]
        self.position_goal=[
                            [0,0,20],
                            [0,0,20],
                            [0,0,20], 
                            [0,0,20],
                            [0,0,20],
                            [0,0,20],
                            [0,0,20]
                           ]
        self.current_dist_to_goal=[
                                    [0],
                                    [0],
                                    [0],
                                    [0],
                                    [0],
                                    [0],
                                    [0]
                                  ]

        self.insertitude=0.20
        self.grid =[
        [0.0, 0], [0.0, 1], [0.0, 2], [0.0, 3], [0.0, 4], 
        [0.5, -0.5], [0.5, 0.5], [0.5, 1.5], [0.5, 2.5], [0.5, 3.5], 
        [0.5, 4.5], [1.0, 0], [1.0, 1], [1.0, 2], [1.0, 3],
        [1.0, 4], [1.5, -0.5], [1.5, 0.5], [1.5, 1.5], [1.5, 2.5],
        [1.5, 3.5], [1.5, 4.5], [2.0, 0], [2.0, 1], [2.0, 2],
        [2.0, 3], [2.0, 4], [2.5, -0.5], [2.5, 0.5], [2.5, 1.5],
        [2.5, 2.5], [2.5, 3.5],[2.5, 4.5], [3.0, 0], [3.0, 1],
        [3.0, 2], [3.0, 3], [3.0, 4], [3.5, -0.5], [3.5, 0.5], 
        [3.5, 1.5], [3.5, 2.5], [3.5, 3.5], [3.5, 4.5], [4.0, 0],
        [4.0, 1], [4.0, 2], [4.0, 3], [4.0, 4], [4.5, -0.5], 
        [4.5, 0.5], [4.5, 1.5], [4.5, 2.5], [4.5, 3.5], [4.5, 4.5], 
        [5.0, 0], [5.0, 1], [5.0, 2], [5.0, 3], [5.0, 4]]

        #fonction qui place le goal au hasard parmis les waypoints de la grille
        self.change_goal()
        self.batterie_start=100
        self.batterie=self.batterie_start
        # definition de l'espace d'action et d'observation
        gym.Env.__init__(self)



        self.action_space =  Dict(
            {
               'mode_trait' : spaces.Box(low=0, high=2, shape=(1,), dtype=np.int_),
               'case_suivante' : spaces.Box(low=0, high=len(self.grid), shape=(1,), dtype=np.int_)
                
            })

        self.observation_space = Dict(
                    {
                        #X Y Z  depart
                        'pos_départ' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,), dtype=np.float32),
                        #X Y Z arrive
                        'intervention' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,7), dtype=np.float32),
                        #X Y Z ma_pos
                        'pos_cur' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3,), dtype=np.float32),
                        #Niveau de batterie 
                        'nivbat' : spaces.Box(low=float("inf"), high=float("inf"), shape=(1,), dtype=np.float32),
                        #X Y Z point de passage des cases
                        'waypnt' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3, 50), dtype=np.float32),
                        #data du velodymne, a determiner 
                        'data_v' : spaces.Box(low=float("inf"), high=float("inf"), shape=(3, 50), dtype=np.float32),
                    }
                  )

        port = '11311'
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node('TD3')
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        self.gzclient_pid = 0      
         
        #inscrition en tant que pub et sub au topic qui va permettre la position a la fin 
        #d'un mode de traiter (sortie de scan ou évitement)
        self.position_sub = rospy.Subscriber("position", Float64, queue_size=1)
        self.position_pub = rospy.Pubscriber("position", Float64, queue_size=1)


    #Actualisaiton de l'observation 
    def get_observation(self):
        
        dict_obs = self.observation_space.sample()
        
        #tirage de l'aléa de baisse subite de la batterie 
        if randint(1,10000)==1 and self.flag_change_bat==False:
            #capacité/2
            self.batterie = self.batterie/2
            self.flag_change_bat=True
        else : 
            #baisse de la batterie simple
            self.batterie = self.batterie-6
        
        #actualisation des position 
        dict_obs['pos_départ'] = np.array([self.position_depart], dtype=int)
        dict_obs['intervention'] = np.array([self.position_goal], dtype=int)
        dict_obs['pos_cur'] = np.array([self.position_sub], dtype=int)

        #baisse de la batterie simple
        dict_obs['nivbat'] = np.array([self.batterie], dtype=int)
        dict_obs['waypnt'] = np.array([self.grid], dtype=int)
        dict_obs['data_v'] = np.array([self.data.v], dtype=int)

        return Dict(dict_obs)
    def reward(self) : 
        
          
        positon_cur= self.position_sub
        positon_goal=self.position_goal

        pre_dist_to_goal =  current_dist_to_goal
        current_dist_to_goal=[[],[],[],[],[],[],[]]
        for i in range(0,7) :
            current_dist_to_goal[i] = np.linalg.norm(positon_goal[i]- positon_cur)

        current_dist_to_start = np.linalg.norm(self.position_depart- positon_cur)
        self.current_dist_to_goal=current_dist_to_goal


        #systeme de recompense doit changer pour la distance avec le goal car plusieur goal maintenant 

        if self.batterie > 50 : 
            #score simple pour la distance du goal
            score_distance = pre_dist_to_goal-current_dist_to_goal*10
        #si batterie infieure a 50 plus de recompense pour aller au goal car doit retourner au depart

        #si goal atteint et bat 50 goal parfait, en dessous de 60 bien mais pas top
        if current_dist_to_goal < GOAL_REACHED_DIST:
            if self.batterie==50 :
                score_goal=1000
            elif self.batterie>50 :
                score_goal=800 
            elif self.batterie<50 :
                score_goal=-1000
                                
        #si batterie 0 alors on check la distance avec le depart 
        if self.batterie==0 :

            if current_dist_to_start < GOAL_REACHED_DIST  :
                score_goal=1000 
            else :
                #malusse si distance trop loin du départ avec bat 0 
                current_distance_to_depart = np.linalg.norm(self.position_depart- positon_cur)
                score_base = current_distance_to_depart*(-10)

        #verif de la case precendante et de laction précédente 
        pso_robot=[self.pos_pre[0],self.pos_pre[1]]
        nbr_case=5

        type, sortie_disp=type_case(nbr_case,pso_robot)


        self.pre_action_tr = action_tr 
        self.pre_action_dep = action_dep

        for i in range(0,4) : 
            if self.pre_action_dep==sortie_disp[i] :
                reward_dep=20
            else :
                reward_dep=-100


        action_tr= self.cur_action_tr
        action_dep= self.cur_action_dep  
        if type==self.pre_action :
            score_traitement=10
        else :
            score_traitement=-10


        self.score = score_distance+score_goal+score_traitement+score_base+reward_dep
        return self.score




    def step(self, action):

        #nombre de step+1
        self.step_counter += 1
        #on recupère les infos contenue dans le dic "action", deux données, mode de traitement et coordonées voulu apres avoir fini le mode de traitement 
        sample_acttion=action
        action_trait=sample_acttion['mode_trait']
        action_dep=sample_acttion['case_suivante']

        self.cur_action_tr = action_trait
        self.cur_action_dep = self.grid[action_dep-1]
        #position avant de bouger 
        self.pos_pre=self.position_sub

        

        #mise en route du mode de traitement choisi
        if action_trait==1 :
            evitement(self.pos_pre, action_dep)
        elif action_trait==2 :
            scan(self.pos_pre, action_dep)
        # else :
        #     #recalibrage()
        
        self.cur_pos=self.position_sub
        #tirage de l'aléa de changement du point d'arrivé
        if randint(1,10000)==1 and self.flag_change_goal==False:
            self.change_goal()
            self.flag_change_goal=True


        self.insertitude = self.insertitude +   np.linalg.norm(self.pos_pre - self.cur_pos)*0.1
                          
        #actualise les infos
        observations = self.get_observation()
        
        #pas capté ce que c'est, mais c'est important 
        info = {}

        #flag de fin d'épisode 
        done = False

        #on teste si on a dépassé le nombre de step max d'un épisode
        if self.step_counter >= self._max_episode_length:
            #si oui fin d'épisode
            done = True

        
        reward=self.reward()
       
        #verifie si une collision a eu lieu 
        collision, min_laser = self.observe_collision(self.velodyne_data)

        #reward negarif si collision + fin d'épisode 
        if collision:
            done = True

       

        #calculer la distance entre le goal et la position actul 
        if self.current_dist_to_goal < GOAL_REACHED_DIST:
            done = True

        #imprime le reward dans ros et le nombre de step
        rospy.loginfo(reward)
        rospy.loginfo(self.step_counter)
        
        #envoi les infos a l'IA
        return observations, reward, done, info



    # Reset the state of the environment to an initial state
    def reset(self):

        
        self.step_counter = 0
        self.position=[0,0,20]
        self.position_depart=[0,0,20]
        self.position_goal=[
                            [0,0,20],
                            [0,0,20],
                            [0,0,20], 
                            [0,0,20],
                            [0,0,20],
                            [0,0,20],
                            [0,0,20]
                           ]
        self.batterie=10000
        self.flag_change=False
        self.position_pub.publish(self.position)

        self.change_goal()

        observations = self.get_observation()


        return observations

    # Place a new goal and check if its lov\cation is not on one of the obstacles
    def change_goal(self):
        n = randint(1,10)
        #self.position_goal=waypoint_case(n)

    
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, min_laser
        return False, min_laser



# Function to put the laser data in bins
def binning(lower_bound, data, quantity):
    width = round(len(data) / quantity)
    quantity -= 1
    bins = []
    for low in range(lower_bound, lower_bound + quantity * width + 1, width):
        bins.append(min(data[low:low + width]))
    return np.array([bins])



def velodyne_callback(self, v):
    data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
    self.velodyne_data = np.ones(self.environment_dim) * 10
    for i in range(len(data)):
        if data[i][2] > -0.2:
            dot = data[i][0] * 1 + data[i][1] * 0
            mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
            mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
            beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
            dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

            for j in range(len(self.gaps)):
                if self.gaps[j][0] <= beta < self.gaps[j][1]:
                    self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                    break



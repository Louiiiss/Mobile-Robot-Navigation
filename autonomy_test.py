#!/usr/bin/env python


import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import sensor_msgs.msg
import numpy as np
import math
from random import randint
import random
<<<<<<< HEAD
=======
import matplotlib.pyplot as plt
import matplotlib.animation as animation
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
import time
import pyqtgraph as pg
import threading
from PyQt4 import QtGui
import Image
from scipy.spatial import distance
<<<<<<< HEAD
import cPickle
=======
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60


rospy.init_node('mcl_localisation')

move_cmd = Twist()

fwd_count = 0
free = True


coords = []

test_coord = []

x = 0.0
y = 0.0
th = 0.0

vx = 0.0
vy = 0.0
vth = 0.0

current_time = 0.0
old_time = 0.0

prev_v = 0.0
prev_ang = 0.0

<<<<<<< HEAD
testImage = Image.open("Megumin.png")
testImage.load()
testImage = Image.open("BinaryOccupancyTestExSmall.png")
=======
# testImage = Image.open("Megumin.png")
# testImage.load()
testImage = Image.open("BinaryOccupancyTestLarge.png")
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
testImage.load()
rawImgData = np.asarray(testImage,dtype="int32")
displayImgData = np.rot90(rawImgData,3)
imageW = rawImgData.shape[1]
imageH = rawImgData.shape[0]
xMult = imageW/7
yMult = imageH/7




def gaussian_prob(sensor_data, particle_data):
    gaussian_base = 1.0/(math.sqrt(2*math.pi))
    mu = sensor_data - particle_data
    exp = -(mu*mu)/2
    return gaussian_base * math.e**exp



def randomParticle():
    global xMult, yMult, rawImgData, imageW, imageH
    xPos = random.uniform(0,imageW)
    yPos = random.uniform(1,imageH)
    theta = random.uniform(0, 2*math.pi)

    return (xPos,yPos,theta)

## THIS IS THE PROBLEM. SORT IT

## address the scaling issue here if it
<<<<<<< HEAD
#def simParticleSensor(particle, beam):
def simParticleSensor(particle):
    global rawImgData, xMult, yMult, imageW, imageH
    x,y,t = particle
    # beam_init = t-0.5
    # fraction = 1/3
    simVal = 5.0
    current_beam = t
    adjX = x
    adjY = y
    x_direction = math.cos(current_beam)
    y_direction = math.sin(current_beam)
    for step in range(xMult*5):
        dx = math.floor(step*x_direction)
        dy = math.floor(step*y_direction)
        if raycastValidHit(adjX+dx,adjY+dy):
            simVal = float(step/float(xMult))
            break

    return simVal
=======
def simParticleSensor(particle, beam):
    global rawImgData, xMult, yMult, imageW, imageH
    x,y,t = particle
    beam_init = t-0.5
    fraction = 1/3
    simVal = 0
    current_beam = beam_init + fraction*beam
    x_direction = math.cos(current_beam)
    y_direction = math.sin(current_beam)
    for step in range((3*xMult)):
        dx = math.floor(step*x_direction)
        dy = math.floor(step*y_direction)
        if raycastValidHit(x+dx,y+dy):
            return float(step/xMult)

    return 5.0
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60


def raycastValidHit(x,y):
    global rawImgData, imageW, imageH
    if x<imageW and x>0 and y<imageH and y>0:
<<<<<<< HEAD
        if np.array_equal(rawImgData[int(imageH-y)][int(x)], [0,0,0]):
=======
        if not np.array_equal(rawImgData[int(imageH-y)][int(x)], [255,255,255]):
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
            return True
        else:
            return False
    else:
<<<<<<< HEAD
        return True




=======
        return False



def particleWeight(particle, sensor_data):
    return sum([gaussian_prob(sensor_data[beam], simParticleSensor(particle,beam))
                for beam in range(3)])
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60

# def particle_weight(particle, scan):
#     global range_ts
#     return sum([gaussian_p(scan[i], mcl_tools.map_range(particle, range_ts[i]))
#                 for i in xrange(5)])


# Take the top ranking particles, then add random samples in, but you'll have to maintain the order
def randomSample(particles, size, weights):
    weights = np.array(weights, dtype=np.double, ndmin=1, copy=0)
    cdf = weights.cumsum()
    cdf /= cdf[-1]
    uniform_samples = np.random.random(size)
    idx = cdf.searchsorted(uniform_samples, side='right')
    return map(lambda n: particles[n], idx)






<<<<<<< HEAD
rotationalSamples = 36
particle_count = 3000
noise = int(particle_count*0.01)
rayCastValues = np.empty([imageH,imageW,rotationalSamples])


particles = [randomParticle() for i in range(particle_count)]#[(150,150,0) for i in range(particle_count)]
print(particles)
weights = np.array([1.0 for p in range(particle_count)])


def particleWeight(particle, sensor_data):
    global rayCastValues, imageH
    # return sum([gaussian_prob(sensor_data[beam], simParticleSensor(particle,beam))
    #              for beam in range(3)])
    x,y,t = particle
    heading = int(round(t*(180/math.pi)/10))
    offset = heading - 2
    summedVals = 0
    for beam in range(5):
        if offset+beam>35:
            angle = offset+beam - 36
        else:
            angle = offset+beam
        xval = int(x)
        yval = int(y)
        # print("prob")
        # print (gaussian_prob(sensor_data[beam], rayCastValues[yval][xval][int(angle)]))
        if (math.isnan(sensor_data[beam])):
            summedVals += gaussian_prob(float(5.0), rayCastValues[yval][xval][int(angle)])
        else:
            summedVals += gaussian_prob(float(sensor_data[beam]-0.1), rayCastValues[yval][xval][int(angle)])

    return summedVals


def initGraph(type):
    global rayCastValues, rawImgData
    radConv = math.pi/180.0
    if(type=="gen"):
        for ys in range(5,rayCastValues.shape[0]-5):
            for xs in range(5,rayCastValues.shape[1]-5):
                for zs in range(rayCastValues.shape[2]):
                    if np.array_equal(rawImgData[int(rawImgData.shape[0]-ys)][int(xs)], [0,0,0]):
                        rayCastValues[ys,xs,zs] = 0
                    else:
                        p = (xs,ys,zs*10*radConv)
                        rayCastValues[ys,xs,zs] = simParticleSensor(p)
        cPickle.dump(rayCastValues, open("raycasts.txt", "wb"))
    else:
        rayCastValues = cPickle.load(open("raycasts.txt", "rb"))



=======

particle_count = 100
noise = int(particle_count*0.1)

particles = [randomParticle() for i in range(particle_count)]
weights = np.array([1.0 for p in range(particle_count)])


>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60


def applyMotionModel(particle,ct):
    global xMult, yMult, prev_v, prev_ang, old_time, current_time, rawImgData, imageH, imageW

    if (old_time==0):
        dt = 0
    else:
        dt = (ct - old_time).to_sec()

    v = prev_v
    w = prev_ang

    x,y,t = particle

    v_dt = v*dt
    w_dt = w*dt
    sigma = math.sqrt(v*v + w*w)/6.0 * dt

<<<<<<< HEAD
    if (dt>5.0):
        new_x = x
        new_y  = y
        new_th  = t
    else:
        new_x = round(x + (random.gauss(v_dt*math.cos(t), sigma)* xMult))
        new_y  = round(y + (random.gauss(v_dt*math.sin(t), sigma)* yMult))
        new_th  = t + random.gauss(w_dt, sigma)

    new_particle = (new_x, new_y, new_th)
    # print("New particle")
    # print(new_particle)

=======
    new_x = math.floor(x + (random.gauss(v_dt*math.cos(t), sigma)* xMult))
    new_y  = math.floor(y + (random.gauss(v_dt*math.sin(t), sigma)* yMult))
    new_th  = t + random.gauss(w_dt, sigma)

    new_particle = (new_x, new_y, new_th)

    print(new_x)
    print(new_y)
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60

    if(new_x>=imageW or new_x<0 or new_y>=imageH or new_y<=0):
        return randomParticle()
    elif (np.array_equal(rawImgData[int(rawImgData.shape[0]-new_y)][int(new_x)],[0,0,0])):
        return randomParticle()
    else:
        return new_particle



<<<<<<< HEAD
### LOCALISATION CALLBACK
def localisation(data):
    global xMult, yMult, prev_v, prev_ang, old_time, current_time, rawImgData, imageH, imageW, prev_v, prev_ang, particles, particle_count, weights, noise

    divisor = 640/5
    sample_data = data.ranges[::divisor]

    # print("localisation Loop Begin")
    # print(current_time)
    # print(old_time)
=======



def localisation(data):
    global xMult, yMult, prev_v, prev_ang, old_time, current_time, rawImgData, imageH, imageW, prev_v, prev_ang, particles, particle_count, weights, noise

    divisor = 640/3
    sample_data = data.ranges[::divisor]


>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60

    current_time = rospy.Time.now()
    particles = [applyMotionModel(p,current_time) for p in particles]
    old_time = rospy.Time.now()


    new_weights = np.array([particleWeight(p,sample_data) for p in particles])

<<<<<<< HEAD
    print("New Weights")
    print(new_weights)

    print("Particles")
    print(particles)

    print("Best Weight")
    print(max(new_weights))
    idx = np.where(new_weights==max(new_weights))
    print("Best Particle")
    bestP = particles[idx[0][0]]
    print(bestP)
    bestSim = simParticleSensor(bestP)
    print("Best Sim")
    print(bestSim)
    print("Current Laser Readings")
    print(sample_data)



    weights *= new_weights
    weights /= weights.sum()
    #wvar = 1.0/sum([w*w for w in weights])
    # if wvar < random.gauss(particle_count*.81, 60):
    particles = randomSample(particles, particle_count-noise, weights) + [randomParticle() for n in range(noise)]
    weights = [1.0 for p in range(particle_count)]
    # else:
    #     pass

    #print(len(particles))
    rate = rospy.Rate(5)
    rate.sleep()

    return
=======
    weights *= new_weights
    weights /= weights.sum()
    wvar = 1.0/sum([w*w for w in weights])
    if wvar < random.gauss(particle_count*.81, 60):
        particles = randomSample(particles, particle_count-noise, weights) + [randomParticle() for n in range(noise)]
        weights = [1.0 for p in range(particle_count)]
    else:
        pass

    return



class Wander():
    global move_cmd, fwd_count, free, pw, coords, x,y,th, vx,vy,vth, current_time, old_time, test_coord, imageW, imageH, xMult, yMult, prev_v, prev_ang





>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60




<<<<<<< HEAD
### MOVEMENT CALLBACK
=======
    def deviate(self,l,c,r):
        print("deviation called")
        r = rospy.Rate(10);
        rc = randint(15,50)
        if(l<1 and r>1):
            count = 0
            move_cmd.linear.x = 0
            move_cmd.angular.z = -0.5
            while (count<rc):
                count = count+1
                self.cmd_vel.publish(move_cmd)
                r.sleep()
        elif(l>1 and r<1):
            count = 0
            move_cmd.linear.x = 0
            move_cmd.angular.z = 0.5
            while (count<rc):
                count = count+1
                self.cmd_vel.publish(move_cmd)
                r.sleep()
        else:
            count = 0
            if(randint(0,1)==0):
                move_cmd.linear.x = 0
                move_cmd.angular.z = 0.5
            else:
                move_cmd.linear.x = 0
                move_cmd.angular.z = -0.5
            while (count<rc):
                count = count+1
                self.cmd_vel.publish(move_cmd)
                r.sleep()
        print("deviation end")
        self.free = True






    def odometryCb(self, msg):
        global old_time, current_time, coords, x,y,th, vx,vy,vth, test_coord, xMult, yMult, rawImgData




    def __init__(self):


        self.fwd_count = 0
        self.free = True


        rospy.loginfo("To stop TurtleBot CTRL + C")

        rospy.on_shutdown(self.shutdown)

        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)


        current_time = rospy.Time.now()
        old_time = rospy.Time.now()


        rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,self.callback)
        rospy.Subscriber('odom',Odometry,self.odometryCb)
        rospy.spin()




>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
def callback(data):
    global old_time, current_time, test_coord, xMult, yMult, prev_v, prev_ang, particles, particle_count, weights, noise, fwd_count, free

    rospy.loginfo(np.nanmean(data.ranges))
    avgrange = np.nanmean(data.ranges)
    right_periphery = data.ranges[0:120]
    centre_vis = data.ranges[120:520]
    left_periphery = data.ranges[520:]
    nm_R = np.nanmean(right_periphery)
    nm_C = np.nanmean(centre_vis)
    nm_L = np.nanmean(left_periphery)
<<<<<<< HEAD
    divisor = 640/5
    sample_data = data.ranges[::divisor]


    # print("Current Laser Readings")
    # print(sample_data)

=======
    divisor = 640/3
    sample_data = data.ranges[::divisor]


>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
    if(free):
        # if(self.fwd_count>50):
        #     print("Perturbation Started")
        #     if (randint(0,50)==1):
        #         self.free = False
        #         self.fwd_count = 0
        #         self.deviate(nm_L, nm_C, nm_R)

        if(nm_L<1 and nm_C>1 and nm_R>1):
            print(1)
            fwd_count = 0
<<<<<<< HEAD
            move_cmd.linear.x = 0.15
            move_cmd.angular.z = -0.25
        elif(nm_L>1 and nm_C>1 and nm_R<1):
            print(2)
            fwd_count = 0
            move_cmd.linear.x = 0.15
            move_cmd.angular.z = 0.25
        elif((nm_L<1.5 and nm_C<1.5 and nm_R>1.5) or max(left_periphery)<1):
            print(3)
            fwd_count = 0
            move_cmd.linear.x = 0
            move_cmd.angular.z = -0.25
=======
            move_cmd.linear.x = 0.2
            move_cmd.angular.z = -0.5
        elif(nm_L>1 and nm_C>1 and nm_R<1):
            print(2)
            fwd_count = 0
            move_cmd.linear.x = 0.2
            move_cmd.angular.z = 0.5
        elif((nm_L<1 and nm_C<1 and nm_R>1) or max(left_periphery)<1):
            print(3)
            fwd_count = 0
            move_cmd.linear.x = 0
            move_cmd.angular.z = -0.5
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
        elif((nm_L>1 and nm_C<1 and nm_R<1) or max(right_periphery)<1):
            print(4)
            fwd_count = 0
            move_cmd.linear.x = 0
<<<<<<< HEAD
            move_cmd.angular.z = 0.3
=======
            move_cmd.angular.z = 0.5
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
        elif(max(data.ranges)<1 and avgrange<1 and np.nanmean(left_periphery)>np.nanmean(right_periphery)):
            print(5)
            fwd_count = 0
            move_cmd.linear.x = 0.0
<<<<<<< HEAD
            move_cmd.angular.z = 0.25
=======
            move_cmd.angular.z = 0.5
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
        elif(max(data.ranges)<1 and avgrange<1 and np.nanmean(left_periphery)<np.nanmean(right_periphery)):
            print(6)
            fwd_count = 0
            move_cmd.linear.x = 0.0
<<<<<<< HEAD
            move_cmd.angular.z = -0.25
        elif((nm_L<1 and nm_C<1 and nm_R<1) or math.isnan(np.nanmean(data.ranges)) or max(centre_vis)<1):
            print(7)
            fwd_count = 0
            move_cmd.linear.x = -0.15
=======
            move_cmd.angular.z = -0.5
        elif((nm_L<1 and nm_C<1 and nm_R<1) or math.isnan(np.nanmean(data.ranges)) or max(centre_vis)<1):
            print(7)
            fwd_count = 0
            move_cmd.linear.x = -0.2
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
            move_cmd.angular.z = 0
        else:
            print(8)
            fwd_count = fwd_count+1
<<<<<<< HEAD
            move_cmd.linear.x = 0.15
=======
            move_cmd.linear.x = 0.2
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
            move_cmd.angular.z = 0


        prev_v = move_cmd.linear.x
        prev_ang = move_cmd.angular.z


        cmd_vel.publish(move_cmd)
<<<<<<< HEAD


=======
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
        return





def shutdown():
    # stop turtlebot
    rospy.loginfo("Stop TurtleBot")
# a default Twist has linear.x of 0 and angular.z of 0.  So it'll stop TurtleBot
    cmd_vel.publish(Twist())
# sleep just makes sure TurtleBot receives the stop command prior to shutting down the script
    rospy.sleep(1)


#def test():

class Graphing():
    global data, curve, line, i, bufferSize, plt, img

    def update(self):
        global coords, test_coord, particles
        self.plt.plot([val[0] for val in particles], [val[1] for val in particles],clear=True, pen=None, symbol='o')
        self.plt.addItem(self.img)

        #print("Angle: " + str(test_coord[0][2]))

        # print ("TEST COORD:")
        # print (test_coord)



    def __init__(self):
        global coords, test_coord, xMult, yMult, imageW,imageH, particles


        app = QtGui.QApplication([])

<<<<<<< HEAD
=======





>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60
        self.plt = pg.plot([val[0] for val in particles], [val[1] for val in particles], pen=None, symbol='o')
        self.img = pg.ImageItem(displayImgData)
        self.img.scale(1, 1)
        self.img.setZValue(-100)
        self.plt.addItem(self.img)
        timer = pg.QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(30) #1000/60 ish
        # test_coord.append([float(3.5*xMult), float(3.5*yMult), float(0.0)])
        # test_coord.append([float(4.0*xMult), float(3.5*yMult)])
        app.exec_()



if __name__ == '__main__':
    try:
<<<<<<< HEAD

        initGraph("load")
        print("Done")


        graphing_thread = threading.Thread(target=Graphing)
        graphing_thread.start()
=======
        graphing_thread = threading.Thread(target=Graphing)
        graphing_thread.start()



>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60

        rospy.loginfo("To stop TurtleBot CTRL + C")

        rospy.on_shutdown(shutdown)

        cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size=10)


<<<<<<< HEAD



        rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,callback=callback,
                queue_size = 1, tcp_nodelay = True)

        current_time = rospy.Time.now()
        old_time = rospy.Time.now()
        rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,callback=localisation,
                queue_size = 1, tcp_nodelay = True)





=======
        current_time = rospy.Time.now()
        old_time = rospy.Time.now()


        rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,callback=callback,
                queue_size = 10, tcp_nodelay = True)
        #rospy.Subscriber('odom',Odometry,self.odometryCb)
        #rospy.spin()
        rospy.Subscriber("/scan",sensor_msgs.msg.LaserScan,callback=localisation,
                queue_size = 10, tcp_nodelay = True)
        #Wander()

        #print(random.gauss(particle_count*.81, 60))

        # print(displayImgData[imageH - 395][189])
        # print(displayImgData[imageH - 394][210])
>>>>>>> be50e6274119fdd462c1c4bfa269afd8faee5b60

    except:
        rospy.loginfo("localisation node terminated.")
    while 1:
       pass

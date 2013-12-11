

import sys
import time
from math import pi, sin, cos

import numpy as np

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.showbase import DirectObject
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode


from environment import Environment
from tasks import BalanceTask

# TODO scaling might really mess up how rotations work.
# TODO implement butt movement (d control).
# TODO flashy environment.
# TODO more colorful bike objects (how come we can't set their color?).

class Game(ShowBase):

    rad2deg = 180. / 3.14
    def __init__(self, agent=None, task=None, noise_mag=0.02,
            show_learning=False,
            slowdown_factor=1.0):
        ShowBase.__init__(self)

        self.show_learning = show_learning
        if self.show_learning:
            self.theta_counter = 0
            self.theta_index = 0
        self.slowdown_factor = slowdown_factor
        self.elapsed_time = 0

        # add some text
        #self.title = OnscreenText(text="Test",
        #                      style=1, fg=(1,1,1,1),
        #                      pos=(0.9,-0.95), scale = .07)

        self.tiltText = self.genLabelText("Bike tilt", 1)
        self.distText = self.genLabelText("Distance to goal", 2)
        self.rewardText = self.genLabelText("Current reward", 3)
        if self.show_learning:
            self.thetaText = self.genLabelText("After 0 episodes...", 10, scale=0.10)

        self.agent = agent
        self.noise_mag = noise_mag
        if task:
            self.task = task
            self.task._butt_disturbance_amplitude = self.noise_mag
            self.bike = self.task.env
            self.bike.reset()
        else:
            # Randlov's bicycle.
            self.bike = Environment()

        #self.bike.h = 0.1

        self.wheel_roll = 0
        self.torque = 0
        self.butt_displacement = 0

        # Load the environment model.
        #self.environ = self.loader.loadModel("models/environment")
        self.environ = self.loader.loadModel("Ground2.egg")
        ## Reparent the model to render.
        self.environ.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        #self.environ.setScale(0.25, 0.25, 0.25)
        #self.environ.setPos(-8, 42, 0)

        # Disable the use of the mouse to control the camera.
        self.disableMouse()

        # "out-of-body experience"; toggles camera control.
        self.accept('o', self.oobe)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.followBikeTask, "FollowBikeTask")
        self.taskMgr.add(self.simulateBicycleTask, "SimulateBicycleTask")

#        self.axes = self.loader.loadModel("misc/xyzAxis.egg")
#        self.axes.reparentTo(self.render)
#        self.axes.setPos(-0, 0, 0)

        self.rear_wheel = self.loader.loadModel("wheel3.egg")
        self.rear_wheel.reparentTo(self.render)
        self.rear_wheel.setPos(0, 0, self.bike.r)

        self.frame = self.loader.loadModel("frame.egg")
        self.frame.reparentTo(self.rear_wheel)
        self.frame.setColor(1, 0, 0)

        self.butt = self.loader.loadModel("frame.egg")
        self.butt.reparentTo(self.frame)
        self.butt.setColor(1, 0, 0)
        self.butt.setScale(1, 0.1, 1)
        self.butt.setZ(1.5 * self.bike.r)
        self.butt.setY(0.3 * self.bike.L)

        # place goal
        self.goalPost = self.loader.loadModel("fork.egg")
        self.goalPost.reparentTo(self.render)
        self.goalPost.setPos(self.bike.x_goal,self.bike.y_goal, 0)

        self.fork = self.loader.loadModel("fork.egg")
        self.fork.reparentTo(self.frame)
        self.fork.setColor(0, 0, 1)
        ## 1 unit in the scaled space of this node is self.bike.L in self.render
        self.fork.setPos(0, self.bike.L, self.bike.r)

        self.front_wheel = self.loader.loadModel("wheel3.egg")
        self.front_wheel.reparentTo(self.fork)
        self.front_wheel.setColor(1, 1, 1)
        self.front_wheel.setPos(0, 0, -self.bike.r)

        self.handlebar = self.loader.loadModel("fork.egg")
        self.handlebar.reparentTo(self.fork)
        self.handlebar.setColor(0, 0, 1)
        self.handlebar.setPos(0, 0, self.bike.r)
        self.handlebar.setHpr(0, 0, 90)

        self.torqueLeftIndicator = self.loader.loadModel("fork.egg")
        self.torqueLeftIndicator.reparentTo(self.fork)
        self.torqueLeftIndicator.setColor(0, 0, 1)
        self.torqueLeftIndicator.setPos(-self.bike.r, 0, self.bike.r)
        self.torqueLeftIndicator.hide()

        self.torqueRightIndicator = self.loader.loadModel("fork.egg")
        self.torqueRightIndicator.reparentTo(self.fork)
        self.torqueRightIndicator.setColor(0, 0, 1)
        self.torqueRightIndicator.setPos(self.bike.r, 0, self.bike.r)
        self.torqueRightIndicator.hide()

#        self.accept('mouse1', self.printHello)

        self.accept('arrow_right', self.torqueRight)
        # When the key is released (lifted 'up').
        self.accept('arrow_right-up', self.noTorque)
        self.accept('arrow_left', self.torqueLeft)
        self.accept('arrow_left-up', self.noTorque)

        self.accept('d', self.buttRight)
        self.accept('d-up', self.noButt)
        self.accept('a', self.buttLeft)
        self.accept('a-up', self.noButt)

        self.camera.setPos(5, -5, 10)

    # Define a procedure to move the camera.
    def followBikeTask(self, task):
        look = self.rear_wheel.getPos()
        self.camera.lookAt(look[0], look[1], look[2] + 1.0)
        self.camera.setPos(look[0] - 1.0, look[1] - 6.0, look[2] + 2.0)

        #self.camera.setPos(*self.rear_wheel.getPos())
        return Task.cont

    def printHello(self):
        print "hello"

    #Macro-like function used to reduce the amount to code needed to create the
    #on screen instructions
    def genLabelText(self, text, i, scale=None):
        if scale == None:
            scale = 0.05
        textObject = OnscreenText(text = text, pos = (-1.3, .95-.05*i), fg=(1,1,0,1),
                      align = TextNode.ALeft, scale = scale)
        return textObject

    def set_theta_to_next_file(self):
        self.theta_counter += 3
        self.theta_index = (self.theta_counter % 21) * 100
        theta = np.loadtxt('balance_thetas/theta_%i.dat' % self.theta_index)
        self.agent.learner._theta = theta

    def simulateBicycleTask(self, task):
        butt_disp_w_noise = self.butt_displacement + self.noise_mag * (2.0 *
                (np.random.rand() - 1.0))

        # update text parameters
        if self.task.goto:
            diststr      = "Distance to goal    = %3.3f" %(self.task.calc_dist_to_goal())
        else:
            diststr      = "Elapsed time        = %3.3f" % (self.elapsed_time)
        tiltstr       = "Tilt                           = %3.3f" %(self.bike.getTilt())
        rewardstr = "Reward                   = %3.3f" %(self.task.getReward())
        self.tiltText.setText(tiltstr)
        self.distText.setText(diststr)
        self.rewardText.setText(rewardstr)
        if self.show_learning:
            thetastr     = "After %i episodes..." % self.theta_index
            self.thetaText.setText(thetastr)

        self.elapsed_time += self.bike.time_step
        if self.show_learning and self.elapsed_time > 7.0:
            self.elapsed_time = 0
            self.set_theta_to_next_file()
            self.bike.reset()
        if self.agent and self.task:
            self.agent.integrateObservation(self.task.getObservation())
            self.task.performAction(self.agent.getAction())
            butt_disp_w_noise = self.bike.actions[1]
        else:
            self.bike.actions = [self.torque, butt_disp_w_noise]
            self.bike.step()
        if abs(self.bike.getTilt()) < BalanceTask.max_tilt:
            self.wheel_roll += self.bike.time_step * self.bike.sigmad
            self.rear_wheel.setPos(self.bike.getXB(), self.bike.getYB(), self.bike.r)
            self.rear_wheel.setP(-self.rad2deg * self.wheel_roll)
            self.rear_wheel.setR(self.rad2deg * self.bike.getTilt())
            self.frame.setP(self.rad2deg * self.wheel_roll)
            self.butt.setX(butt_disp_w_noise)
            self.fork.setH(self.rad2deg * self.bike.getSteer())
            self.front_wheel.setP(-self.rad2deg * self.wheel_roll)
            time.sleep(self.bike.time_step * self.slowdown_factor)
        else:
            time.sleep(1)
            self.bike.reset()
            self.elapsed_time = 0
            if self.show_learning:
                self.set_theta_to_next_file()
        return Task.cont

    def torqueRight(self):
        self.torqueRightIndicator.show()
        self.torque = -2.0

    def torqueLeft(self):
        self.torqueLeftIndicator.show()
        self.torque = 2.0

    def buttRight(self):
        self.butt_displacement = 0.02

    def buttLeft(self):
        self.butt_displacement = -0.02

    def noTorque(self):
        self.torqueRightIndicator.hide()
        self.torqueLeftIndicator.hide()
        self.torque = 0.0

    def noButt(self):
        self.butt_displacement = 0.0

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        slowdown_factor = float(sys.argv[1])
    else:
        slowdown_factor = 1.0
    app = Game(slowdown_factor=slowdown_factor)
    app.run()

from math import pi, sin, cos
 
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.showbase import DirectObject

from environment import Environment
from tasks import BalanceTask

# TODO scaling might really mess up how rotations work.
# TODO implement butt movement (d control).
# TODO flashy environment.
# TODO more colorful bike objects (how come we can't set their color?).
 
class MyApp(ShowBase):

    rad2deg = 180. / 3.14
    def __init__(self):
        ShowBase.__init__(self)

        self.wheel_roll = 0
        self.torque = 0
        self.butt_displacement = 0

        # Randlov's bicycle.
        self.bike = Environment()
 
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

    def simulateBicycleTask(self, task):
        self.bike.actions = [self.torque, self.butt_displacement]
        self.bike.step()
        if abs(self.bike.getTilt()) < BalanceTask.max_tilt:
            self.wheel_roll += self.bike.time_step * self.bike.sigmad
            self.rear_wheel.setPos(self.bike.getXB(), self.bike.getYB(),
                    self.bike.r)
            self.rear_wheel.setP(-self.rad2deg * self.wheel_roll)
            self.rear_wheel.setR(self.rad2deg * self.bike.getTilt())
            self.frame.setP(self.rad2deg * self.wheel_roll)
            self.fork.setH(-self.rad2deg * self.bike.getSteer())
            self.front_wheel.setP(-self.rad2deg * self.wheel_roll)
        else:
            self.bike.reset()
        return Task.cont

    def torqueRight(self):
        self.torque = 2.0

    def torqueLeft(self):
        self.torque = -2.0

    def buttRight(self):
        self.butt_displacement = 0.02

    def buttLeft(self):
        self.butt_displacement = -0.02

    def noTorque(self):
        self.torque = 0.0

    def noButt(self):
        self.butt_displacement = 0.0

app = MyApp()
app.run()

from math import pi, sin, cos
 
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.showbase import DirectObject

from environment import Environment

# TODO scaling might really mess up how rotations work.
 
class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        # Randlov's bicycle.
        self.bike = Environment()
 
        # Load the environment model.
        self.environ = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.environ.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.environ.setScale(0.25, 0.25, 0.25)
        self.environ.setPos(-8, 42, 0)

        # Disable the use of the mouse to control the camera.
        self.disableMouse()

        # "out-of-body experience"; toggles camera control.
        self.accept('o', self.oobe)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.followBikeTask, "FollowBikeTask")
        self.taskMgr.add(self.simulateBicycleTask, "SimulateBicycleTask")

        self.axes = self.loader.loadModel("misc/xyzAxis.egg")
        self.axes.reparentTo(self.render)
        self.axes.setScale(0.1, 0.1, 0.1)
        self.axes.setPos(-0, 0, 0)

        self.rear_wheel = self.loader.loadModel("misc/smiley.egg")
        self.rear_wheel.reparentTo(self.render)
        self.rear_wheel.setScale(self.bike.r, 0.1 * self.bike.r, self.bike.r)
        self.rear_wheel.setPos(0, 0, self.bike.r)
        self.rear_wheel.setHpr(90, 0, 0)

        self.frame = self.loader.loadModel("box.egg")
        self.frame.reparentTo(self.rear_wheel)
        self.frame.setColor(1, 0, 0)
        self.frame.setScale(self.bike.L / self.bike.r, 1, 1)

        self.fork = self.loader.loadModel("box.egg")
        fork_rel_height = 2.0
        fork_rel_width = 0.1
        self.fork.reparentTo(self.frame)
        self.fork.setColor(0, 0, 1)
        self.fork.setScale(
                fork_rel_width * self.bike.r / self.bike.L, 1, fork_rel_height)
        # 1 unit in the scaled space of this node is self.bike.L in self.render
        self.fork.setPos(1, 0, 0)

        self.front_wheel = self.loader.loadModel("misc/smiley.egg")
        self.front_wheel.reparentTo(self.fork)
        self.front_wheel.setColor(1, 1, 1)
        self.front_wheel.setScale(1.0 / fork_rel_width, 1,
                1.0 / fork_rel_height)

        self.handlebar = self.loader.loadModel("box.egg")
        self.handlebar.reparentTo(self.fork)
        # 1 unit in the frame space is 2 * self.bike.r in self.render.
        handlebar_rel_length = 30.0
        handlebar_rel_height = 0.1
        self.handlebar.setScale(1.0 , handlebar_rel_length,
                handlebar_rel_height / fork_rel_height)
        self.handlebar.setPos(0, -0.5 * handlebar_rel_length, 1)

        self.accept('mouse1', self.printHello)
        
        self.accept('arrow_right', self.moveBikeXPos)

        self.camera.setPos(5, -5, 10)

    # Define a procedure to move the camera.
    def followBikeTask(self, task):
        look = self.front_wheel.getPos()
        #self.camera.lookAt(look[0], look[1], look[2] + 1.1)
        self.camera.setPos(self.rear_wheel, -15, 0, 5)
    
        #self.camera.setPos(*self.rear_wheel.getPos())
        return Task.cont

    def printHello(self):
        print "hello"

    def moveBikeXPos(self):
        self.rear_wheel.setPos(self.rear_wheel, 0.1, 0, 0)

    def simulateBicycleTask(self, task):
        self.bike.step()
        self.rear_wheel.setPos(self.bike.getXB(), self.bike.getYB(),
                self.bike.r)
        return Task.cont
 
app = MyApp()
app.run()

from mpl_toolkits.mplot3d import axes3d
import numpy as np
import math
import sys
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


def unitVector(vector):
    """
        Returns the unit vector of a given input vector.

        Params:
            vector -> input vector.

        Returns:
            numpy.array().
    """

    # Divide the input vector by its magnitude.
    return vector / np.linalg.norm(vector)


class Segment2D:

    """
        A part of the FabrikSolver2D to store a part of an inverse kinematics chain.
    """

    def __init__(self, referenceX, referenceY, length, angle):
        """
            Params:
                referenceX -> x component of the reference point.

                referenceY -> y component of the reference point.

                length -> length of the segemnt.

                angle -> initial angle of the segment.
        """

        self.angle = angle

        # Store the length of the segment.
        self.length = length

        # Calculate new coördinates.
        deltaX = math.cos(math.radians(angle)) * length
        deltaY = math.sin(math.radians(angle)) * length

        # Calculate new coördinates with respect to reference.
        newX = referenceX + deltaX
        newY = referenceY + deltaY

        # Store new coördinates.
        self.point = np.array([newX, newY])

    def setPoint(self, a, b, reference):
        # TODO: add high level function for updating point.
        pass


class FabrikSolver2D:
    """
        An inverse kinematics solver in 2D. Uses the Fabrik Inverse Kinematics Algorithm.
    """

    def __init__(self, baseX=0, baseY=0, marginOfError=0.01):
        """
            Params:
                baseX -> x component of the base.

                baseY -> y coördinate of the base.

                marginOfError -> the margin of error for the algorithm.
        """

        # Create the base of the chain.
        self.basePoint = np.array([baseX, baseY])

        # Initialize empty segment array -> [].
        self.segments = []

        # Initialize iterate history
        self.history = []

        # Initialize length of the chain -> 0.
        self.armLength = 0

        # Initialize the margin of error.
        self.marginOfError = marginOfError

    def addSegment(self, length, angle):
        """
            Add new segment to chain with respect to the last segment.

            Params:
                length -> length of the segment.

                angle -> initial angle of the segment.
        """

        if len(self.segments) > 0:

            segment = Segment2D(
                self.segments[-1].point[0], self.segments[-1].point[1], length, angle + self.segments[-1].angle)
        else:
            segment = Segment2D(
                self.basePoint[0], self.basePoint[1], length, angle)

        self.armLength += segment.length

        self.segments.append(segment)

    def isReachable(self, targetX, targetY):
        """  
            Check if a point in space is reachable by the end-effector.

            Params:
                targetX -> the target x coördinate to check.

                targetY -> the target y coördinate to check.

            Returns:
                Boolean.  
        """

        if np.linalg.norm(self.basePoint - np.array([targetX, targetY])) < self.armLength:
            return True
        return False

    def inMarginOfError(self, targetX, targetY):
        """  
            Check if the distance of a point in space and the end-effector is smaller than the margin of error.

            Params:
                targetX -> the target x coördinate to check.

                targetY -> the target y coördinate to check.

                targetZ -> the target z coördinate to check. 

            Returns:
                Boolean.  
        """

        if np.linalg.norm(self.segments[-1].point - np.array([targetX, targetY])) < self.marginOfError:
            return True
        return False

    def iterate(self, targetX, targetY):
        """ 
            Do one iteration of the fabrik algorithm. Used in the compute function. 
            Use in simulations or other systems who require motion that converges over time.  

            Params:
                targetX -> the target x coördinate to move to.

                targetY -> the target y coördinate to move to.
        """

        target = np.array([targetX, targetY])

        # Forward.
        for i in range(len(self.segments) - 1, 0, -1):

            # Op het uiteinde moeten we eerst het eindpunt gebruiken om de formule te kunnen toepassen.

            # Kijk of de waarde van i gelijk is aan de index van de laatse vector aan de arm.
            if i == len(self.segments) - 1:
                # Ga nog een index lager naar de een na laatse vector in de list. Gebruik dan de formule met de eindvector en vermenigvuldig met de lengte van de vector met de laatste index.

                # Vervang oude vector met nieuwe vector.
                self.segments[i-1].point = (unitVector(
                    self.segments[i-1].point - target) * self.segments[i].length) + target

            else:
                self.segments[i-1].point = (unitVector(self.segments[i-1].point -
                                                       self.segments[i].point) * self.segments[i].length) + self.segments[i].point

         # Backward.
        for i in range(len(self.segments)):
            if i == 0:
                self.segments[i].point = (unitVector(
                    self.segments[i].point - self.basePoint) * self.segments[i].length) + self.basePoint

            elif i == len(self.segments) - 1:
                self.segments[i].point = (unitVector(
                    self.segments[i-1].point - target) * self.segments[i].length * -1) + self.segments[i-1].point

            else:
                self.segments[i].point = (unitVector(
                    self.segments[i].point - self.segments[i-1].point) * self.segments[i].length) + self.segments[i-1].point

    def compute(self, targetX, targetY):
        """  
            Iterate the fabrik algoritm until the distance from the end-effector to the target is within the margin of error.

            Params:
                targetX -> the target x coördinate to move to.

                targetY -> the target x coördinate to move to.

        """

        self.history.append(copy.deepcopy(self.segments))
        if self.isReachable(targetX, targetY):
            while not self.inMarginOfError(targetX, targetY):
                self.iterate(targetX, targetY)
                self.history.append(copy.deepcopy(self.segments))
        else:
            print('Target not reachable.')

    def plot(self, segments=None, save=False, name="graph", xMin=-300, xMax=300, yMin=-300, yMax=300):
        """  
            Plot the chain.

            Params:
                save -> choose to save the plot to a file.

                name -> give the plot a name.

                xMin -> the left bound of the plot.

                xMax -> the right bound of the plot.

                yMin -> the low bouwnd of the plot.

                yMax -> the hight bound of the plot.
        """

        if segments == None:
            segments = self.segments

        # Plot chain.
        for i in range(len(segments)):
            # Plot the coördinate of a segment point.
            plt.plot([segments[i].point[0]], [
                     segments[i].point[1]], 'ro')

            # Display coördinates of the point.
            plt.text(segments[i].point[0], segments[i].point[1] + 1, '(x:{}, y:{})'.format(
                int(segments[i].point[0]), int(segments[i].point[1])))

        # Plot begin point
        plt.plot([self.basePoint[0]], [self.basePoint[1]], 'bo')
        plt.text(self.basePoint[0], self.basePoint[1], 'Base')

        plt.axis([xMin, xMax, yMin, yMax])
        plt.grid(True)

        if save == True:
            plt.savefig('{}.png'.format(name))

        plt.show(block=True)

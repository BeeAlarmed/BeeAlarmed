"""! @brief This class is used to keep track of detected bee characteristics. """
##
# @file Statistics.py
#
# @brief This class is used to keep track of detected bee characteristics
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Config import *
from collections import deque

class Statistics(object):
    """! The 'Statistics' class keeps track of all the monitoring results.
    """

    def __init__(self):
        """! Initializes the statistics object
        """
        self._beesIn = 0
        self._beesOut = 0
        self._beesInOverall = 0
        self._beesOutOverall = 0
        self._wespenCount = 0
        self._wespenCountOverall = 0
        self._varroaCount = 0
        self._varroaCountOverall = 0
        self._pollenCount = 0
        self._pollenCountOverall = 0
        self._coolingCount = 0
        self._coolingCountOverall = 0
        self._processedFames = 0
        self._processedFamesOverlall = 0

    def frameProcessed(self):
        """! Increases the frame processed counter
        """
        self._processedFames += 1
        self._processedFamesOverlall += 1

    def addBeeIn(self):
        """! Increases the bee-in counter
        """
        self._beesIn += 1
        self._beesInOverall += 1

    def addBeeOut(self):
        """! Increases the bee-out counter
        """
        self._beesOut +=1
        self._beesOutOverall +=1

    def getBeeCountOverall(self):
        """! Returns the overal counted bees (bees_in, bees_out)
        @return tuple (bees_in, bees_out)
        """
        return (self._beesInOverall, self._beesOutOverall)

    def getBeeCount(self):
        """! Returns the counted bees (bees_in, bees_out)
        """
        return (self._beesIn, self._beesOut)

    def addDetection(self, tag):
        """! Adds a detected bee charcteristic by tag
             @param tag any of (TAG_WESPE, TAG_VARROA, TAG_POLLEN, TAG_COOLING)
        """
        if TAG_WESPE == tag:
            self._wespenCount += 1
            self._wespenCountOverall += 1
        if TAG_VARROA == tag:
            self._varroaCount += 1
            self._varroaCountOverall += 1
        if TAG_POLLEN == tag:
            self._pollenCount += 1
            self._pollenCountOverall += 1
        if TAG_COOLING == tag:
            self._coolingCount += 1
            self._coolingCountOverall += 1

    def addClassificationResult(self, trackId, result):
        """! Adds a detected bee charcteristic by classification results
             @param trackId unused
             @param result  A set containing any combination of (TAG_WESPE, TAG_VARROA, TAG_POLLEN, TAG_COOLING=
        """
        if TAG_WESPE in result:
            self.addDetection(TAG_WESPE)
        if TAG_VARROA in result:
            self.addDetection(TAG_VARROA)
        if TAG_POLLEN in result:
            self.addDetection(TAG_POLLEN)
        if TAG_COOLING in result:
            self.addDetection(TAG_COOLING)

    def addClassificationResultByTag(self, trackId, tag):
        """! Adds a detected bee charcteristic by tag
             @param trackId unused
             @param tag  any of TAG_WESPE, TAG_VARROA, TAG_POLLEN, TAG_COOLING
        """
        self.addDetection(tag)

    def readStatistics(self):
        """! Return the current statistics for counted wasps, varroa, pollen,
             cooling, bees in, bees out and the amount of processed frames
             @return tuple
        """
        return (self._wespenCount,
                self._varroaCount,
                self._pollenCount,
                self._coolingCount,
                self._beesIn,
                self._beesOut,
                self._processedFames)

    def readOverallStatistics(self):
        """! Return the overall statistics for counted wasps, varroa, pollen,
             cooling, bees in, bees out and the amount of processed frames
             @return tuple
        """
        return (self._wespenCountOverall,
                self._varroaCountOverall,
                self._pollenCountOverall,
                self._coolingCountOverall,
                self._beesInOverall,
                self._beesOutOverall,
                self._processedFamesOverall)

    def resetStatistics(self):
        """! Resets the current statistics
        """
        self._wespenCount = 0
        self._varroaCount = 0
        self._pollenCount = 0
        self._coolingCount = 0
        self._beesIn = 0
        self._beesOut = 0
        self._processedFames = 0


__dh = None
def getStatistics():
    """! Returns the statistics object
    #TODO: use pattern to realize singleton
    @return The statistics instance
    """
    global __dh
    if __dh == None:
        __dh = Statistics()

    return __dh


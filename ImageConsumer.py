"""! @brief This module contains the 'ImageConsumer', which processes the video frames. """
##
# @file ImageConsumer.py
#
# @brief This module contains the 'ImageConsumer', which processes the video frames.
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
import cv2
import time
import logging
import queue
import multiprocessing
from Statistics import getStatistics
from ImageProvider import ImageProvider
from BeeDetection import detect_bees
from BeeTracking import BeeTracker, BeeTrack
from Utils import get_config, get_args
from BeeProcess import BeeProcess
if get_config("NN_ENABLE"):
    from BeeClassification import BeeClassification

from multiprocessing import Queue

logger = logging.getLogger(__name__)

class ImageConsumer(BeeProcess):
    """! The 'ImageConsumer' processes the frames which are provided
    by the 'ImageProvider'. It performs the bee detection, bee tracking and
    forwards findings to the 'ImageExtractor' to feed them to the neural network.
    """
    def __init__(self):
        """! Intitilizes the 'ImageConsumer'
        """
        super().__init__()
        self._extractQueue = Queue()
        self._classifierResultQueue = None
        self._imageQueue = None
        self._visualQueue = None
        self.set_process_param("e_q", self._extractQueue)
        self.set_process_param("c_q", self._classifierResultQueue)
        self.set_process_param("i_q", self._imageQueue)
        self.set_process_param("v_q", self._visualQueue)

    def getPositionQueue(self):
        """! Returns the queue object where detected bee positions will be put
        @return A queue object
        """
        return self._extractQueue

    def setImageQueue(self, queue):
        """! Set the queue object where the image consumer can find new frames
        @param queue    The queue object to read new frames from
        """
        self._imageQueue = queue
        self.set_process_param("i_q", self._imageQueue)
    
    def setVisualQueue(self, queue):
        """! Set the queue object where the image consumer can find new frames
        @param queue    The queue object to read new frames from
        """
        self._visualQueue = queue
        self.set_process_param("v_q", self._visualQueue)

    def setClassifierResultQueue(self, queue):
        """! Set the queue obejct where the 'ImageConsumer' can read classification results
        @param  queue   The queue that provides the classification results from the neural network
        """
        self._classifierResultQueue = queue
        self.set_process_param("c_q", self._classifierResultQueue)

    @staticmethod
    def run(c_q, i_q, e_q, v_q, parent, stopped, done):
        """! The main thread that runs the 'ImageConsumer'
        """
        _process_time = time.time()
        _process_cnt = 0
        _lastProcessFPS = 0
        _start_t = time.time()
        writer = None

        # Create a Bee Tracker
        tracker = BeeTracker(50, 20)

        # Create statistics object
        statistics = getStatistics()

        if type(i_q) == type(None):
            raise("No image queue provided!")

        while stopped.value == 0:

            _process_cnt += 1

            # When the neural network is enabled, then read results from the classifcation queue
            # and forward them the the corresponding track and statistics
            if get_config("NN_ENABLE"):

                # Populate classification results
                while not c_q.empty():

                    # Transfer results to the track
                    trackId, result = c_q.get()
                    track = tracker.getTrackById(trackId)
                    if type(track) != type(None):
                        track.imageClassificationComplete(result)
                    else:
                        statistics.addClassificationResult(trackId, result)

            # Process every incoming image
            if not i_q.empty() and stopped.value == 0:

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(get): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Get frame set
                fs = i_q.get()
                if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_150x300":
                    img_1080, img_540, img_180 = fs
                elif get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
                    img_540, img_180 = fs
                
                if _process_cnt % 100 == 0:
                    logger.debug("Process time(track): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Detect bees on smallest frame
                detected_bees, detected_bee_groups = detect_bees(img_180, 3)
                
                # Update tracker with detected bees
                if get_config("ENABLE_TRACKING"):
                    tracker.update(detected_bees, detected_bee_groups)

                # Extract detected bee images from the video, to use it our neural network
                # Scale is 2 because detection was made on img_540 but cutting is on img_1080
                if get_config("ENABLE_IMAGE_EXTRACTION"):
                    data = tracker.getLastBeePositions(get_config("EXTRACT_FAME_STEP"))
                    if len(data) and type(e_q) != type(None):
                        if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_150x300":
                            e_q.put((data, img_1080, 2, _process_cnt))
                        elif get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
                            e_q.put((data, img_540, 1, _process_cnt))
                        else:
                            raise("Unknown setting for EXT_RES_75x150, expected EXT_RES_150x300 or EXT_RES_75x150")

                # Draw the results if enabled
                if get_config("VISUALIZATION_ENABLED"):
                    if _process_cnt % get_config("VISUALIZATION_FRAME_SKIP") == 0:
                        try:
                            data = (img_540, detected_bees, detected_bee_groups, tracker, _lastProcessFPS) 
                            v_q.put(data, block=False)
                        except queue.Full:
                            print("frame skip !!")
                

                # Print log entry about process time each 100 frames
                if _process_cnt % 100 == 0:
                    _pt = time.time() - _process_time
                    _lastProcessFPS = 100 / _pt
                    logger.debug("Process time all: %0.3fms" % (_pt * 10.0))
                    _process_time = time.time()

                # Update statistics
                _dh = getStatistics()
                _dh.frameProcessed()

            else:
                time.sleep(0.01)

            # Limit FPS by delaying manually
            _end_t = time.time() - _start_t
            limit_time = 1 / get_config("LIMIT_FPS_TO")
            if _end_t < limit_time:
                time.sleep(limit_time - _end_t)
            _start_t = time.time()

        logger.info("Image Consumer stopped")

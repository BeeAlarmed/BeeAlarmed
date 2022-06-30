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
from Statistics import getStatistics
from threading import Thread
from ImageProvider import ImageProvider
from BeeDetection import detect_bees
from BeeTracking import BeeTracker, BeeTrack
from Utils import get_config, get_args
if get_config("NN_ENABLE"):
    from BeeClassification import BeeClassification

from multiprocessing import Queue

logger = logging.getLogger(__name__)

class ImageConsumer(Thread):
    """! The 'ImageConsumer' processes the frames which are provided
    by the 'ImageProvider'. It performs the bee detection, bee tracking and
    forwards findings to the 'ImageExtractor' to feed them to the neural network.
    """
    def __init__(self):
        """! Intitilizes the 'ImageConsumer'
        """
        self.stopped = False
        self._done = False
        self._extractQueue = Queue()
        self._classifierResultQueue = None
        self._imageQueue = None
        self._visualQueue = None
        Thread.__init__(self)

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
    
    def setVisualQueue(self, queue):
        """! Set the queue object where the image consumer can find new frames
        @param queue    The queue object to read new frames from
        """
        self._visualQueue = queue

    def setClassifierResultQueue(self, queue):
        """! Set the queue obejct where the 'ImageConsumer' can read classification results
        @param  queue   The queue that provides the classification results from the neural network
        """
        self._classifierResultQueue = queue

    def run(self: Thread) -> None:
        """! The main thread that runs the 'ImageConsumer'
        """
        _process_time = 0
        _process_cnt = 0
        writer = None

        # Create a Bee Tracker
        tracker = BeeTracker(50, 20)

        # Create statistics object
        statistics = getStatistics()

        if type(self._imageQueue) == type(None):
            raise("No image queue provided!")

        while not self.stopped:

            _start_t = time.time()

            # When the neural network is enabled, then read results from the classifcation queue
            # and forward them the the corresponding track and statistics
            if get_config("NN_ENABLE"):

                # Populate classification results
                while not self._classifierResultQueue.empty():

                    # Transfer results to the track
                    trackId, result, image = self._classifierResultQueue.get()
                    track = tracker.getTrackById(trackId)
                    if type(track) != type(None):
                        track.imageClassificationComplete(result)
                    else:
                        statistics.addClassificationResult(trackId, result)

            # Process every incoming image
            if not self._imageQueue.empty():
                _process_cnt += 1

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(get): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Get frame set
                fs = self._imageQueue.get()
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
                    if len(data) and type(self._extractQueue) != type(None):
                        if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_150x300":
                            self._extractQueue.put((data, img_1080, 2, _process_cnt))
                        elif get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
                            self._extractQueue.put((data, img_540, 1, _process_cnt))
                        else:
                            raise("Unknown setting for EXT_RES_75x150, expected EXT_RES_150x300 or EXT_RES_75x150")

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(previsual): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                try:
                    data = (img_540, img_1080, detected_bees, detected_bee_groups, tracker) 
                    self._visualQueue.put(data, block=False)
                except queue.Full:
                    print("frame skip !!")
                
                if _process_cnt % 100 == 0:
                    logger.debug("Process time(visual): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Print log entry about process time each 100 frames
                _process_time += time.time() - _start_t
                if _process_cnt % 100 == 0:
                    logger.debug("Process time all: %0.3fms" % (_process_time * 10.0))
                    _process_time = 0


                # Limit FPS by delaying manually
                _end_t = time.time() - _start_t
                limit_time = 1 / get_config("LIMIT_FPS_TO")
                if _end_t < limit_time:
                    time.sleep(limit_time - _end_t)

                # Update statistics
                _dh = getStatistics()
                _dh.frameProcessed()

            else:
                time.sleep(0.1)

        self._done = True
        logger.info("Image Consumer stopped")

    def isDone(self: Thread) -> bool:
        """! Returns whether the thread has stopped or not
        """
        return self._done

    def stop(self: Thread) -> None:
        """! Forces the thread to stop
        """
        self.stopped = True


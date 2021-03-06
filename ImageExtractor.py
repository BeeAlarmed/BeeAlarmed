"""! @brief This module contains the 'ImageExtractor' """
##
# @file ImageExtractor.py
#
# @brief Process that extracts single bee-images from a given
#        frame and position information.

# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Utils import cutEllipseFromImage
from Config import *
import time
import logging
import cv2
import signal
import multiprocessing
import datetime
from os.path import join, exists
from os import makedirs

logger = logging.getLogger(__name__)

class ImageExtractor(object):
    """! The 'ImageExtractor' class provides a process that extracts
          bee-images from a givem video frame. It uses a queue for
          incoming requests, see 'setInQueue' and a one
          queue to provides results, see 'setResultQueue'.

          To request can be inserted in the incoming queue, by providing
          a tuple with the following contents:

            (data, image, scale)

          - 'data' contains the result of 'getLastBeePositions' from the 'BeeTracker'.
          - 'image' represents the frame to extract the bee images from.
          - 'scale' is used adapt to different frame sizes.
    """

    def __init__(self):
        """! Initializes the image extractor
        """
        self.stopped = multiprocessing.Value('i', 0)
        self.done = multiprocessing.Value('i', 0)
        self._resultQueue = None
        self._inQueue = None
        self._process = None

    def start(self):
        """! Starts the image extraction process
        """
        if type(self._inQueue) == type(None):
            raise("Please provide a classifier queue!")

        # Start the process
        self._process = multiprocessing.Process(target=self.extractor, \
                args=(self._inQueue, self._resultQueue, self.stopped, self.done))
        self._process.start()

    def setResultQueue(self, queue):
        """! Sets the result queue of the image extractor
        @param queue  Sets the queue, where the process pushes its result
        """
        self._resultQueue = queue

    def setInQueue(self, queue):
        """! Sets the input queue of the image extractor
        @param queue  Sets the queue, where the process reads its input
        """
        self._inQueue = queue

    def stop(self):
        """! Forces the process to stop
        """
        self.stopped.value = 1
        try:
            while(not self._inQueue.empty()):
                self._inQueue.get()
            while(not self._resultQueue.empty()):
                self._resultQueue.get()
        except:
            pass

    def join(self):
        """! Terminate the process and joins it. Should be called after 'stop'.
        """
        self._process.terminate()
        self._process.join()

    @staticmethod
    def extractor(in_q, out_q, stopped, done):
        """! Static method, starts the process of the image extractor
        """

        # Ignore interrupt signals
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        _process_time = 0
        _process_cnt = 0

        # Prepare save path
        if SAVE_EXTRACTED_IMAGES and not exists(SAVE_EXTRACTED_IMAGES_PATH):
            makedirs(SAVE_EXTRACTED_IMAGES_PATH)

        while stopped.value == 0:
            if not in_q.empty():

                _start_t = time.time()
                _process_cnt += 1

                # Read one entry from the process queue
                data, image, scale = in_q.get()

                # Extract the bees from the image
                for item in data:
                    trackId, lastPosition = item

                    # Extract the bee image and sharpness value of the image
                    img, sharpness = cutEllipseFromImage(lastPosition, image, 0, scale)

                    # Check result, in some cases the result may be None
                    #  e.g. when the bee is close to the image border
                    if type(img) != type(None):

                        # Filter by minimum sharpness
                        if sharpness > EXTRACT_MIN_SHARPNESS:

                            # Forward the image to the classification process (if its running)
                            if NN_ENABLE:
                                if out_q.full():
                                    logger.debug("Classifier Queue full")
                                    # Remove oldest entry to add a new one
                                    out_q.get()
                                out_q.put((trackId, img))

                            # Save the image in case its requested
                            if SAVE_EXTRACTED_IMAGES:
                                cv2.imwrite(SAVE_EXTRACTED_IMAGES_PATH + "/%i-%s.jpeg" % (_process_cnt, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), img)

                _process_time += time.time() - _start_t

                # Print log entry about process time each 100 frames
                if _process_cnt % 100 == 0:
                    logger.debug("Process time: %0.3fms" % (_process_time * 10.0))
                    _process_time = 0

            else:
                time.sleep(0.01)

        # The process stopped
        logger.info("Image extractor stopped")

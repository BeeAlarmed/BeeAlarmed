"""! @brief This module contains the 'ImageExtractor' """
##
# @file ImageExtractor.py
#
# @brief Process that extracts single bee-images from a given
#        frame and position information.

# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Utils import cutEllipseFromImage, get_config
import time
import logging
import cv2
import queue
import multiprocessing
import datetime
from os.path import join, exists
from os import makedirs
from BeeProcess import BeeProcess

logger = logging.getLogger(__name__)

class ImageExtractor(BeeProcess):
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
        super().__init__()
        self._resultQueue = None
        self._inQueue = None

    def start(self):
        """! Starts the image extraction process
        """
        if type(self._inQueue) == type(None):
            raise("Please provide a classifier queue!")

        # Start the process
        super().start()

    def setResultQueue(self, queue):
        """! Sets the result queue of the image extractor
        @param queue  Sets the queue, where the process pushes its result
        """
        self._resultQueue = queue
        self.set_process_param("out_q", self._resultQueue)

    def setInQueue(self, queue):
        """! Sets the input queue of the image extractor
        @param queue  Sets the queue, where the process reads its input
        """
        self._inQueue = queue
        self.set_process_param("in_q", self._inQueue )

    @staticmethod
    def run(in_q, out_q, parent, stopped, done):

        """! Static method, starts the process of the image extractor
        """
        _process_time = 0
        _process_cnt = 0

        # Prepare save path
        e_path = get_config("SAVE_EXTRACTED_IMAGES_PATH")
        if get_config("SAVE_EXTRACTED_IMAGES") and not exists(e_path):
            makedirs(e_path)

        while stopped.value == 0:
            if not in_q.empty():

                _start_t = time.time()
                _process_cnt += 1

                # Read one entry from the process queue
                data, image, scale, frame_id = in_q.get()

                # Extract the bees from the image
                for item in data:
                    trackId, lastPosition = item

                    # Extract the bee image and sharpness value of the image
                    img, sharpness = cutEllipseFromImage(lastPosition, image, 0, scale)

                    # Check result, in some cases the result may be None
                    #  e.g. when the bee is close to the image border
                    if type(img) != type(None):

                        # Filter by minimum sharpness
                        if sharpness > get_config("EXTRACT_MIN_SHARPNESS"):

                            # Forward the image to the classification process (if its running)
                            if get_config("NN_ENABLE"):
                                try:
                                    out_q.put((trackId, img, frame_id), block=False)
                                except queue.Full:
                                    pass
                                    
                            # Save the image in case its requested
                            if get_config("SAVE_EXTRACTED_IMAGES"):
                                cv2.imwrite(e_path + "/%i-%s.jpeg" % (_process_cnt, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")), img)

                _process_time += time.time() - _start_t

                # Print log entry about process time each 100 frames
                if _process_cnt % 100 == 0:
                    logger.debug("Process time: %0.3fms" % (_process_time * 10.0))
                    _process_time = 0

            else:
                time.sleep(0.01)

        # The process stopped
        logger.info("Image extractor stopped")

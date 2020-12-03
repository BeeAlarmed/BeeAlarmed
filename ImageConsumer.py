"""! @brief This module contains the 'ImageConsumer', which processes the video frames. """
##
# @file ImageConsumer.py
#
# @brief This module contains the 'ImageConsumer', which processes the video frames.
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Statistics import getStatistics
from Config import *
from threading import Thread
from ImageProvider import ImageProvider
from BeeDetection import detect_bees
from BeeTracking import BeeTracker, BeeTrack
if NN_ENABLE:
    from BeeClassification import BeeClassification

from multiprocessing import Queue
import cv2
import time
import logging

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
            if NN_ENABLE:
                if _process_cnt % 100 == 0:
                    logger.debug("Process time(q): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Populate classification results
                while not self._classifierResultQueue.empty():
                    if _process_cnt % 100 == 0:
                        logger.debug("Process time(nn): %0.3fms" % ((time.time() - _start_t) * 1000.0))

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
                if NN_EXTRACT_RESOLUTION == EXT_RES_150x300:
                    img_1080, img_540, img_180 = fs
                elif NN_EXTRACT_RESOLUTION == EXT_RES_75x150:
                    img_540, img_180 = fs

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(detec): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Detect bees on smallest frame
                detected_bees, detected_bee_groups = detect_bees(img_180, 3)

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(track): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # # Update tracker with detected bees
                if ENABLE_TRACKING:
                    tracker.update(detected_bees, detected_bee_groups)

                # Extract detected bee images from the video, to use it our neural network
                # Scale is 2 because detection was made on img_540 but cutting is on img_1080
                if ENABLE_IMAGE_EXTRACTION:
                    data = tracker.getLastBeePositions(EXTRACT_FAME_STEP)
                    if len(data) and type(self._extractQueue) != type(None):
                        if NN_EXTRACT_RESOLUTION == EXT_RES_150x300:
                            self._extractQueue.put((data, img_1080, 2))
                        elif NN_EXTRACT_RESOLUTION == EXT_RES_75x150:
                            self._extractQueue.put((data, img_540, 1))
                        else:
                            raise("Unknown setting for EXT_RES_75x150, expected EXT_RES_150x300 or EXT_RES_75x150")

                if _process_cnt % 100 == 0:
                    logger.debug("Process time(print): %0.3fms" % ((time.time() - _start_t) * 1000.0))

                # Draw preview if wanted
                if not args.noPreview:

                    draw_on = img_540.copy()
                    if DRAW_DETECTED_ELLIPSES:
                        for item in detected_bees:
                            cv2.ellipse(draw_on, item, (0, 0, 255), 2)
                    if DRAW_DETECTED_GROUPS:
                        for item in detected_bee_groups:
                            cv2.ellipse(draw_on, item, (255, 0, 0), 2)

                    if DRAW_TRACKING_RESULTS:
                        tracker.drawTracks(draw_on)

                    skipKey = 1 if FRAME_AUTO_PROCESS else 0

                    cv2.imshow("frame", draw_on)
                    if cv2.waitKey(skipKey) & 0xFF == ord('q'):
                        break

                    # Save as Video
                    if SAVE_AS_VIDEO:
                        if type(writer) == type(None):
                            h, w, c = draw_on.shape
                            writer = cv2.VideoWriter(SAVE_AS_VIDEO_PATH, cv2.VideoWriter_fourcc(*'MJPG'), 18, (w, h))
                        writer.write(draw_on)

                # Print log entry about process time each 100 frames
                _process_time += time.time() - _start_t
                if _process_cnt % 100 == 0:
                    logger.debug("Process time: %0.3fms" % (_process_time * 10.0))
                    _process_time = 0


                # Limit FPS by delaying manually
                _end_t = time.time() - _start_t
                limit_time = 1 / LIMIT_FPS_TO
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


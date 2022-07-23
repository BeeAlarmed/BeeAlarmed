"""! @brief This module contains the 'Visual' module to visualise the results """
##
# @file Visual.py
#
# @brief Uses opencv to draw to resuls

# @section authors Author(s)
# - Created by Fabian Hickert on december 2021
#
import time
import logging
import cv2
import signal
import multiprocessing
import datetime
import queue
from Utils import get_config, get_args
from BeeTracking import BeeTracker, BeeTrack
from multiprocessing import Queue

logger = logging.getLogger(__name__)

class Visual(object):
    """! The 'Visual' module uses a separate process to visualize the programs results.
         It uses a in-queue to receive the current image and the tracking results
    """

    def __init__(self):
        """! Initializes the visualiser
        """
        self.stopped = multiprocessing.Value('i', 0)
        self.done = multiprocessing.Value('i', 0)
        self._inQueue = multiprocessing.Queue(maxsize=20)
        self._process = None

    def start(self):
        """! Starts the process
        """
        # Start the process
        self._process = multiprocessing.Process(target=self.visualise, \
                args=(self._inQueue, self.stopped, self.done))
        self._process.start()

    def getInQueue(self):
        """! Sets the input queue to receive the current image and the tracking results
             (img_540, detected_bees, detected_bee_groups, tracker)
        """
        return self._inQueue 

    def stop(self):
        """! Forces the process to stop
        """
        self.stopped.value = 1
        try:
            while(not self._inQueue.empty()):
                self._inQueue.get()
        except:
            pass

    def join(self):
        """! Terminate the process and joins it. Should be called after 'stop'.
        """
        self._process.terminate()
        self._process.join()

    @staticmethod
    def visualise(in_q, stopped, done):
        """! Static method, starts the process of the image extractor
        """

        # Ignore interrupt signals
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        _process_time = 0
        _process_time_n100 = 0
        _process_cnt = 0
        _process_time_n100 = time.time()
        _lastFPS = 0

        while stopped.value == 0:
            if not in_q.empty():

                _start_t = time.time()
               
                # Log FPS
                if _process_cnt != 0 and _process_cnt % 100 == 0:
                    fps = (100/ (time.time() - _process_time_n100))
                    _lastFPS = fps
                    _process_time_n100 = time.time()
                    logger.info(f"FPS visual: {fps:.2f} FPS")

                _process_cnt += 1

                # Read one entry from the process queue
                img_540, detected_bees, detected_bee_groups, tracker, processFPS = in_q.get()

                if get_config("SHOW_VISUALIZATION_DETAILS"):
                    cv2.putText(img_540,"Process FPS: %.2f" % (processFPS,), 
                        (img_540.shape[1]-200,20),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                    cv2.putText(img_540,"Visual FPS: %.2f" % (_lastFPS), 
                        (img_540.shape[1]-200,40),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)
                    cv2.putText(img_540,"Frame Skip: %i" % (get_config("VISUALIZATION_FRAME_SKIP"),), 
                        (img_540.shape[1]-200,60),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 1)

                if get_config("DRAW_DETECTED_ELLIPSES"):
                    for item in detected_bees:
                        cv2.ellipse(img_540, item, (0, 0, 255), 2)
                if get_config("DRAW_DETECTED_GROUPS"):
                    for item in detected_bee_groups:
                        cv2.ellipse(img_540, item, (255, 0, 0), 2)

                if get_config("DRAW_TRACKING_RESULTS"):
                    tracker.drawTracks(img_540)

                # Draw preview if wanted
                if not get_args().noPreview:

                    skipKey = 1 if get_config("FRAME_AUTO_PROCESS") else 0

                    cv2.imshow("frame", img_540)
                    if cv2.waitKey(skipKey) & 0xFF == ord('q'):
                        break

                # Save as Video
                if get_config("SAVE_AS_VIDEO"):
                    if type(writer) == type(None):
                        h, w, c = img_540.shape

                        #TODO: Set real Framerate from video input or from video stream
                        writer = cv2.VideoWriter(get_config("SAVE_AS_VIDEO_PATH"), \
                                cv2.VideoWriter_fourcc(*'MJPG'), 18, (w, h))
                    writer.write(img_540)
                

                _process_time += time.time() - _start_t

                # Print log entry about process time each 100 frames
                if _process_cnt % 100 == 0:
                    logger.debug("Process time: %0.3fms" % (_process_time * 10.0))
                    _process_time = 0

            else:
                time.sleep(0.01)

        # The process stopped
        logger.info("Image extractor stopped")

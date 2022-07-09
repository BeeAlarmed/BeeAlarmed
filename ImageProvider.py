"""! @brief This module contain the image provider process. """
##
# @file ImageProvider.py
#
# @brief This modul contains the 'ImageProvider' process which
#         extracts single images from a video file or camera
#         input and provides it for further processing
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Utils import get_config, get_frame_config
from pathlib import Path
from queue import Queue
import cv2
import time
import logging
import multiprocessing
import signal

logger = logging.getLogger(__name__)

class ImageProvider(object):

    """! The 'ImageProvider' class provides access to the camera or video
          input using a queue. It runs in a dedicated process and feeds
          the extracted images into a queue, that can then be used by other
          tasks.
    """
    def __init__(self, video_source=None, video_file=None):
        """! Initializes the image provider process and queue
        """
        self.frame_config = None
        self._videoStream = None
        self._stopped = multiprocessing.Value('i', 0)
        self._started = multiprocessing.Value('i', 0)
        self._process = None


        # Validate the frame_config
        max_w = max_h = 0
        frame_config = get_frame_config()
        if not len(frame_config):
            raise BaseException("At least one frame config has to be provided!")

        # Ensure that each item of the frame config has the same size or less as the previous one
        for num, item in enumerate(frame_config):
            if type(item[0]) != int:
                raise BaseException("Expected item 1 of frame_config %i to be integer" % (num+1,))
            if type(item[1]) != int:
                raise BaseException("Expected item 2 of frame_config to be integer" % (num+1,))
            if item[2] not in (cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED):
                raise BaseException("Expected item 3 of frame_config to be one of cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED")

            if max_w < item[0]:
                max_w = item[0]
            if max_h < item[1]:
                max_h = item[1]

        # Ensure that at least one source is defined
        if video_source is None and video_file is None:
            raise BaseException("Either a video file or a video source id is required")

        # Prepare for reading from video file
        self.frame_config = frame_config
        if video_file is not None:
            self._queue = multiprocessing.Queue(maxsize=get_config("FRAME_SET_BUFFER_LENGTH_VIDEO"))
            vFile = Path(video_file)
            if not vFile.is_file():
                raise BaseException("The given file '%s' doesn't seem to be valid!" % (video_file,))
        else:
            self._queue = multiprocessing.Queue(maxsize=get_config("FRAME_SET_BUFFER_LENGTH_CAMERA"))

        self._process = multiprocessing.Process(target=self._imgProcess,
                args=(self._queue, frame_config, video_source, video_file, self._stopped, self._started))
        self._process.start()

    def getQueue(self):
        """! Returns the queue-object where the extracted frames will be put.
        @return Returns the queue object
        """
        return self._queue

    def isStarted(self):
        """! Returns whether the image processing started or not
        @return Returns True if the process was started
        """
        return self._started.value

    def isDone(self):
        """! Returns whether the image processing still running or ended
        @return Returns True if the process has stopped
        """
        return self._stopped.value

    def stop(self) -> None:
        """! Stops the image provides process
        """
        self._stopped.value = 1
        while not self._queue.empty():
            self._queue.get()

    def join(self):
        """! Terminates the process and joins it. Should be called after 'stop'.
        """
        self._process.terminate()
        self._process.join()

    @staticmethod
    def _imgProcess(q_out, config, video_source, video_file, stopped, started):

        # Ignore interrupts
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        # Open video stream
        if video_source == None:
            logger.info("Starting from video file input: %s" % (video_file,))
            # use HW acceleration for video file
            if get_config("USE_GSTREAM"):
                _videoStream = cv2.VideoCapture('filesrc location={}\
                                        ! queue ! h264parse ! omxh264dec ! nvvidconv \
                                        ! video/x-raw,format=BGRx,width=960,height=544 ! queue ! videoconvert ! queue \
                                        ! video/x-raw, format=BGR ! appsink'.format(video_file),
                                        cv2.CAP_GSTREAMER)
            else:
                _videoStream = cv2.VideoCapture(video_file)
        else:
            logger.info("Starting from camera input")
            _videoStream = cv2.VideoCapture(video_source)
            w, h, f = get_config("CAMERA_INPUT_RESOLUTION")
            if f != None:
                fourcc = cv2.VideoWriter_fourcc(*f)
                _videoStream.set(cv2.CAP_PROP_FOURCC, fourcc)
            if w != None:
                _videoStream.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
            if h != None:
                _videoStream.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))

        _process_time = 0
        _process_cnt = 0
        _skipped_cnt = 0
        while stopped.value == 0:

            # Check if the queue is full
            if q_out.full():

                # If the queue is full, then report it
                if _skipped_cnt % 100 == 0:
                    logger.debug("Buffer reached %i" % (q_out.qsize(),))
                time.sleep(get_config("FRAME_SET_FULL_PAUSE_TIME"))
                _skipped_cnt += 1
            else:

                # There is still space in the queue, get a frame and process it
                _start_t = time.time()
                (_ret, _frame) = _videoStream.read()

                if started.value == 0:
                    started.value = 1

                if _ret:

                    # Get the original shape
                    h, w, c = _frame.shape

                    # Convert the frame according to the given configuration.
                    # The image will be resized if necessary and converted into gray-scale
                    #  if needed.
                    fs = tuple()
                    for item in config:
                        width, height = _frame.shape[0:2]
                        if width != item[0] or height != item[1]:
                            _frame = cv2.resize(_frame, (item[1], item[0]))
                        if item[2] == cv2.IMREAD_GRAYSCALE:
                            tmp = cv2.cvtColor(_frame, cv2.COLOR_BGR2GRAY)
                            fs += (tmp,)
                        else:
                            fs += (_frame,)

                    # put the result in the outgoing queue
                    q_out.put(fs)

                    # Calculate the time needed to process the frame and print it
                    _process_time += time.time() - _start_t
                    _process_cnt += 1
                    if _process_cnt % 100 == 0:
                        logger.debug('FPS: %i (%i, %i)\t\t buffer size: %i' % (100/_process_time, w, h ,q_out.qsize()))
                        _process_time = 0
                else:
                    logger.error("No frame received!")
                    stopped.value = 1

        # End of process reached
        logger.info("Image provider stopped")

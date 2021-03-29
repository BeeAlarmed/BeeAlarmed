"""! @brief This module provides access to the neural network."""
##
# @file BeeClassification.py
#
# @brief Process that runs the neural network fir bee image classification
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Utils import get_config
from os import listdir, makedirs
from os.path import isfile, join, exists
from datetime import datetime
import cv2
import time
import multiprocessing
import logging

logger = logging.getLogger(__name__)

class BeeClassification(object):
    """! The 'BeeClassification' class provides access to the neural network
          which runs in a seperate process. It provides two queue-objects,
          one to queue to incoming images that have to be processed by the
          neural network and a second one, where the results are put.
    """

    def __init__(self):

        """! Initializes the neural network and the queues
        """
        # used to stop the process
        self._stopped = multiprocessing.Value('i', 0)

        # reports when the porcess with the neural network is ready
        self._ready = multiprocessing.Value('i', 0)
        self._done = False

        # The queue for the incoming images
        self._q_in = multiprocessing.Queue(maxsize=20)

        ## The queue where the results are reported
        self._q_out = multiprocessing.Queue()

        # Start the process and wait for it to run
        self._process = multiprocessing.Process(target=self._neuralN, args=(self._q_in, self._q_out, self._ready, self._stopped))
        self._process.start()
        while self._ready.value == 0:
            time.sleep(1)
            logger.debug("Waiting for neural network, this may take up to two minutes")
        logger.debug("Classification terminated")

    def getQueue(self):
        """! Returns the queue-object for the icoming queue
        @return  Returns the incoming queue object
        """
        return self._q_in

    def getResultQueue(self):
        """! Returns the queue-object which holds the classification results
        @return  Returns the result queue object
        """
        return self._q_out

    def stop(self) -> None:
        """! Tell the classification process and thus the neural network to stop.
             The process will quit and you need to call join afterwards.
        """
        self._stopped.value = True
        while not self._q_in.empty():
            self._q_in.get()
        while not self._q_out.empty():
            self._q_out.get()

    def join(self):
        """! Terminate the process and joins it. Should be called after 'stop'.
        """
        self._process.terminate()
        self._process.join()

    @staticmethod
    def _neuralN(q_in, q_out, ready, stopped):
        """! Static method, starts a new process that runs the neural network
        """

        # Include tensorflow within the process
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.models import Sequential
        from tensorflow.keras import layers
        import signal

        # Ignore interrupts
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        _process_time = 0
        _process_cnt = 0

        # Enable growth of GPU usage
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)

        # Load the model
        try:
            _model = tf.keras.models.load_model(get_config("NN_MODEL_FOLDER"))
        except Exception as e:
            ready.value = True
            logger.error("Failed to load Model: %s" % (e,))
            return

        # Detect desired image size for classification
        img_height = 300
        img_width = 150
        if get_config("NN_EXTRACT_RESOLUTION") == "EXT_RES_75x150":
            img_height = 150
            img_width = 75

        # Initialize the network by using it
        # This ensures everything is preloaded when needed
        if True:

            # Load all images from the "Images" folder and feed them to the neural network
            # This ensures that the network is fully running when we start other processes
            test_images = ["Images/"+f for f in listdir("Images") if isfile(join("Images", f))]
            imgs = []
            for item in test_images:
                img = tf.io.read_file(item)
                img = tf.image.decode_jpeg(img, channels=3)
                img = tf.image.resize(img, [img_height, img_width])
                imgs.append(img)

            # Perform prediction
            _model.predict_step(tf.convert_to_tensor(imgs))

        # Mark process as ready
        ready.value = True

        # Create folders to store images with positive results
        if get_config("SAVE_DETECTION_IMAGES"):
            for lbl in ["varroa", "pollen", "wasps", "cooling"]:
                s_path = get_config("SAVE_DETECTION_PATH")
                if not exists(join(s_path, lbl)):
                    makedirs(join(s_path, lbl))

        classify_thres = get_config("CLASSIFICATION_THRESHOLDS")
        while stopped.value == 0:

            # While the image classification queue is not empty
            # feed the images to the network and push the result
            # back in the outgoing queue
            if not q_in.empty():
                _start_t = time.time()
                _process_cnt += 1

                images = []
                tracks = []

                # Load the images from the in-queue and prepare them for the use in the network
                failed = False
                while not q_in.empty() and len(images) < 20 and stopped.value == 0:
                    item = q_in.get()
                    t, img = item

                    # Change color from BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if img.shape != (img_height, img_width, 3):
                        img = tf.image.resize(img, [img_height, img_width])
                    images.append(img)
                    tracks.append(t)

                # Quit process if requested
                if stopped.value != 0:
                    return

                # Feed collected images to the network
                if len(tracks):
                    results = _model.predict_on_batch(tf.convert_to_tensor(images))

                    # precess results
                    for num, track in enumerate(tracks):

                        # Create dict with results
                        entry = set([])
                        for lbl_id, lbl in enumerate(["varroa", "pollen", "wasps", "cooling"]):
                            if results[lbl_id][num][0] > classify_thres[lbl]:
                                entry.add(lbl)

                                # Save the corresponding image on disc
                                if get_config("SAVE_DETECTION_IMAGES") and lbl in get_config("SAVE_DETECTION_TYPES"):
                                    img = images[num]
                                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                    cv2.imwrite(get_config("SAVE_DETECTION_PATH") + "/%s/%i-%s.jpeg" % (lbl, _process_cnt, \
                                            datetime.now().strftime("%Y%m%d-%H%M%S")), img)

                        # Push results back
                        q_out.put((tracks[num], entry, images[num]))

                _end_t = time.time() - _start_t
                logger.debug("Process time: %0.3fms - Queued: %i, processed %i" % (_end_t * 1000.0, q_in.qsize(), len(images)))
                _process_time += _end_t
            else:
                time.sleep(0.01)
        logger.info("Classifcation stopped")

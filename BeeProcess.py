"""! @brief This file contains the 'BeeProcess' """
##
# @file BeeProcess.py
#
# @brief Process class which all processes inherit from

# @section authors Author(s)
# - Created by Fabian Hickert on december 2022
#
import time
import signal
import logging
import multiprocessing
logger = logging.getLogger(__name__)

class BeeProcess(object):
    def __init__(self):
        """! Initializes the defaults
        """
        self._stopped = multiprocessing.Value('i', 0)
        self._done = multiprocessing.Value('i', 0)
        self._process = None
        self._process_params = {}
        self._parentclass = self.__class__
        self._started = False

    def set_process_param(self, name, queue):
        self._process_params[name] = queue

    def isDone(self):
        return self._done.value
    
    def isStarted(self):
        return self._started

    def stop(self):
        """! Forces the process to stop
        """

        # Wait for process to stop

        self._stopped.value = 1
        for i in range(100):
            if self._done.value == 1:
                break
            time.sleep(0.01)
        if self._done.value == 0:
            logger.warn("Terminating process after waiting 1s for gracefull shutdown!")
            self._process.terminate()

        for qn,q in self._process_params.items():
            if q is not None:
                try:
                    while not q.empty():
                        q.get()
                except:
                    pass

    def join(self):
        if self._stopped.value == 0 and self._done.value == 0 and self._started:
            self._process.join()

    @staticmethod
    def run(*args):
        print("run")
        time.sleep(1)

    @staticmethod
    def _run(args):

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        parent = args["parent"]
        stopped = args["stopped"]
        done = args["done"]
        try:
            parent.run(**args)
        except KeyboardInterrupt as ki:
            logger.debug(">> Received KeyboardInterrupt")

        stopped.value = 1
        done.value = 1
            
    def start(self):
        """! Starts the image extraction process
        """
        # Start the process
        args = self._process_params.copy()
        args["parent"] = self._parentclass
        args["stopped"] = self._stopped
        args["done"] = self._done

        self._process = multiprocessing.Process(target=self._run, \
                args=[args])
        self._process.start()
        self._started = True



#!/usr/bin/env python3
from ImageProvider import ImageProvider
from ImageConsumer import ImageConsumer
from ImageExtractor import ImageExtractor
from LoRaWANThread import LoRaWANThread
from Visual import Visual
from Utils import get_args, get_config
import logging
import time
import sys

# Only load neural network if needed. the overhead is quite large
if get_config("NN_ENABLE"):
    from BeeClassification import BeeClassification

logging.basicConfig(level=logging.DEBUG, format='%(process)d %(asctime)s - %(name)s - %(levelname)s - \t%(message)s')
logger = logging.getLogger(__name__)

def main():

    # Check input format: camera or video file
    args = get_args()
    if args.video:
        logger.info("Starting on video file '%s'" % (args.video))
        imgProvider = ImageProvider(video_file=args.video)
    else:
        logger.info("Starting on camera input")
        imgProvider = ImageProvider(video_source=0)

    while(not (imgProvider.isStarted() or imgProvider.isDone())):
        time.sleep(1)

    if imgProvider.isDone():
        logger.error("Aborted, ImageProvider did not start. Please see log for errors!")
        return

    # Enable bee classification process only when its enabled
    imgClassifier = None
    if get_config("NN_ENABLE"):
        imgClassifier = BeeClassification()

    # Create processes and connect message queues between them
    lorawan = None
    if get_config("RN2483A_LORA_ENABLE"):
        lorawan = LoRaWANThread()
    imgExtractor = ImageExtractor()
    imgConsumer = ImageConsumer()
    visualiser = Visual()
    imgConsumer.setImageQueue(imgProvider.getQueue())
    imgConsumer.setVisualQueue(visualiser.getInQueue())
    if get_config("NN_ENABLE"):
        imgExtractor.setResultQueue(imgClassifier.getQueue())
        imgConsumer.setClassifierResultQueue(imgClassifier.getResultQueue())
    imgExtractor.setInQueue(imgConsumer.getPositionQueue())

    try:

        # Start the processes
        imgConsumer.start()
        imgExtractor.start()
        visualiser.start()
        if lorawan is not None:
            lorawan.start()

        # Quit program if end of video-file is reached or
        # the camera got disconnected
        #imgConsumer.join()
        while True:
            time.sleep(0.01)
            if imgConsumer.isDone() or imgProvider.isDone():
                raise SystemExit(0)

    except (KeyboardInterrupt, SystemExit):

        # Tear down all running process to ensure that we don't get any zombies
        if lorawan is not None:
            lorawan.stop()
        imgProvider.stop()
        imgExtractor.stop()
        visualiser.stop()
        imgConsumer.stop()
        if imgClassifier:
            imgClassifier.stop()
            imgClassifier.join()
        imgExtractor.join()
        imgProvider.join()
        visualiser.join()

if __name__ == '__main__':
    main()
    logger.info('\n! -- BeeAlarmed stopped!\n')

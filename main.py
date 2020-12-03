#!/usr/bin/env python3
from ImageProvider import ImageProvider
from ImageConsumer import ImageConsumer
from ImageExtractor import ImageExtractor
from LoRaWANThread import LoRaWANThread
from Config import *
import logging
import time
import sys

# Only load neural network if needed. the overhead is quite large
if NN_ENABLE:
    from BeeClassification import BeeClassification

logger = logging.getLogger(__name__)

def main():

    # Check input format: camera or video file
    if args.video:
        logger.info("Starting on video file '%s'" % (args.video))
        imgProvider = ImageProvider(video_file=args.video)
    else:
        logger.info("Starting on camera input")
        imgProvider = ImageProvider(video_source=0)

    # Enable bee classification process only when its enabled
    imgClassifier = None
    if NN_ENABLE:
        imgClassifier = BeeClassification()

    # Create processes and connect message queues between them
    lorawan = LoRaWANThread()
    imgExtractor = ImageExtractor()
    imgConsumer = ImageConsumer()
    imgConsumer.setImageQueue(imgProvider.getQueue())
    if NN_ENABLE:
        imgExtractor.setResultQueue(imgClassifier.getQueue())
        imgConsumer.setClassifierResultQueue(imgClassifier.getResultQueue())
    imgExtractor.setInQueue(imgConsumer.getPositionQueue())

    try:

        # Start the processes
        imgConsumer.start()
        imgExtractor.start()
        lorawan.start()

        # Quit program if end of video-file is reached or
        # the camera got disconnected
        while True:
            time.sleep(0.01)
            if imgConsumer.isDone() or imgProvider.isDone():
                raise SystemExit(0)

    except (KeyboardInterrupt, SystemExit):

        # Tear down all running process to ensure that we don't get any zombies
        lorawan.stop()
        imgProvider.stop()
        imgExtractor.stop()
        imgConsumer.stop()
        if imgClassifier:
            imgClassifier.stop()
            imgClassifier.join()
        imgExtractor.join()
        imgProvider.join()

if __name__ == '__main__':
    main()
    logger.info('\n! -- BeeAlarmed stopped!\n')

"""! @brief This module provides access to the LoRaWAN transceiver. """
##
# @file LoRaWANThread.py
#
# @brief This module provides access to the LoRaWAN transceiver.
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from Config import *
from threading import Thread
from Statistics import getStatistics
import time
import logging
from serial import Serial
import struct

logger = logging.getLogger(__name__)

class LoRaWANThread(Thread):
    """! The LoRaWAN object utilizes the RB2483A transeiver from Microship
         to transfer the monitoring results to the server.
         When initiated, this object runs in a separate thread.
    """
    def __init__(self):
        """! Initializes the class
        """
        self.stopped = False
        self._done = False
        self._ser = None
        Thread.__init__(self)

    def _sendCmd(self: Thread, cmd: str) -> str:
        """! Used to send a command to the transceiver
        @param cmd  The command to be excuted
        @return     Returns the resulting response
        """
        logger.debug("Sending command   : %s" % (cmd))
        self._ser.write((cmd + '\r\n').encode("UTF-8"))
        tmp = self._ser.readline().decode("UTF-8").strip()
        logger.debug("Received          : %s" % (str(tmp)))
        return tmp

    def _read(self: Thread) -> str:
        """! Used to read from the serial interface of the transeiver
        @return     Returns the resulting response
        """
        tmp = self._ser.readline().decode("UTF-8").strip()
        logger.debug("Received (_read)  : %s" % (str(tmp)))
        return tmp

    def initialize(self: Thread):
        """! Initializes and the LoRaWAN transceiver
        Configures european channels and perform ABP connection
        """

        # this only works with a connected serial interface
        if self._ser:
            try:
                self._ser.close()
            except:
                pass

        # Put the transeiver into a defined state
        self._ser = Serial(RN2483A_USB_PORT, 57600, timeout=60)
        self._sendCmd("sys reset")
        self._sendCmd("sys factoryRESET")
        self._sendCmd("radio set freq 868000000")

        # Get channels
        channel_config = LORAWAN_CHANNEL_CONFIG

        # Calculate Duty Cycle limitation
        # The 868 band has a 1% cycle, rn2483 has a duty-cycle per channel
        # this means the duty cycle per channel has to be shorter than 1%
        # Instead of 99% off, the channels have a (100% - (1 / channel-count)) dty-cylce
        if LORAWAN_DISABLE_DUTY_CYCLE_CHECKS:
            dty = 9
        else:
            dty = int((100 - (1 / len(channel_config))) * 10)

        # Introduce the channel setup to the mac layer
        for ch in channel_config:
            self._sendCmd("mac set ch freq %i %i" % (ch[0], ch[1]))
            self._sendCmd("mac set ch dcycle %i %i" % (ch[0], dty))
            self._sendCmd("mac set ch drrange %i %i %i" % (ch[0], ch[2], ch[3]))
            self._sendCmd("mac set ch status %i on" % (ch[0],))

        # Set connection details
        self._sendCmd("mac set devaddr %s" % (LORAWAN_DEVADDR,))
        self._sendCmd("mac set nwkskey %s" % (LORAWAN_NET_SESSION_KEY,))
        self._sendCmd("mac set appskey %s" % (LORAWAN_APP_SESSION_KEY,))
        self._sendCmd("mac set sync 34")

        # Save settings
        self._sendCmd("mac save")

        # Initiate the join process
        self._sendCmd("mac join abp")
        self._read()

    def run(self: Thread) -> None:
        """! Start the LoRaWAN thread.
        Sends the monitoring results every five minutes
        """

        # Ensure the transcevier is initialized
        try:
            self.initialize()
        except Exception as e:
            logger.error("Initialization failed: " + str(e))
            return

        fail_cnt = 0

        # Send every five minutes
        while not self.stopped:

            # Get current statistics
            _dh = getStatistics()
            (_wespenCount,
                 _varroaCount,
                 _pollenCount,
                 _coolingCount,
                 _beesIn,
                 _beesOut,
                 _frames) = _dh.readStatistics()

            # Reset statistics
            _dh.resetStatistics()

            # Prepare data
            data = tuple([_varroaCount, _pollenCount, _coolingCount, _wespenCount, _beesIn, _beesOut])
            data_bin = struct.pack("hhhhhh", *data)

            # Conver monitoring results in transferrable string
            data_bin_str = ""
            for item in data_bin:
                data_bin_str += "%02X" % (item,)
            logger.debug("Binary data: " + str(data_bin))
            logger.debug("String data: " + data_bin_str)

            # Send the LoRaWAN Telegram
            ret = self._sendCmd("mac tx uncnf 1 %s" % (data_bin_str,))
            if ret == "ok":
                ret = self._read()
                if ret == "mac_tx_ok":
                    logger.info("Sending successful with: %s" % (ret,))
                else:
                    logger.error("Sending failed with: %s" % (ret,))

            elif ret in ["not_joined", "silent", "frame_counter_err_rejoin_needed", "mac_paused"]:
                fail_cnt += 1
                logger.error("Sending failed with: %s" % (ret,))
                self.initialize()
            else:
                fail_cnt += 1
                logger.error("Sending failed with: %s" % (ret,))

            # Wait for five minutes, before sending the next results
            _start_t = time.time()
            while not self.stopped and (_start_t + (60 * 1) > time.time()):
                time.sleep(0.01)

        # Close the serial connection
        if self._ser != None:
            self._ser.close()

        # Thread stopped
        self._done = True
        logger.info("LoRaWAN stopped")

    def isDone(self: Thread) -> bool:
        """! Return whether the Thread has stopped or is still running
        @return True if it is stopped, else False
        """
        return self._done

    def stop(self: Thread) -> None:
        """! Stopps the LoRaWAN Thread and joins it
        """
        self.stopped = True
        self.join()


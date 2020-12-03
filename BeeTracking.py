"""! @brief This module contains classes related to the tracking of bees. """
##
# @file BeeTracking.py
#
# @brief This modul contains the 'BeeTrack' and 'BeeTracker' classes.
#        The 'BeeTrack' keep the track information for each bee. To
#        do so, it implements a kalman-filter.
#        The 'BeEtracker' combine all bee-tracks and provides access
#        to them including updates of tracks and drawing the current
#        tracks
#
# @section authors Author(s)
# - Created by Fabian Hickert on december 2020
#
from filterpy.common import kinematic_kf
from collections import deque
import numpy as np
import math
import cv2
import logging
import random

from Config import *
from Statistics import getStatistics
from Utils import loadWomanNames, variance_of_laplacian, pointInEllipse

logger = logging.getLogger(__name__)


class BeeTrack(object):

    """! The 'BeeTrack' object tracks a single bees movement using a kalman filter.
    """
    def __init__(self, trackId):
        super(BeeTrack, self).__init__()

        ##! Name that gets shown on the screen for that bee
        self._name = ""

        ## Last detected bee position
        self._lastDetect = None

        ## The tracks ID
        self.trackId = trackId

        # Initialize the klaman filter
        self.dt = 1
        self.KF = kinematic_kf(dim=2, order=2, dt=self.dt, dim_z=1, order_by_dim=True)
        self.KF.R *= 2
        self.KF.Q = np.array(
                         [[self.dt**4/4,     self.dt**3/2,   self.dt**4/2,    0,0,0 ],
                          [self.dt**3/2,     self.dt**2,     self.dt**4,      0,0,0 ],
                          [self.dt**3/1,     self.dt**1,     self.dt**1/2,    0,0,0 ],
                          [0,0,0, self.dt**4/4,     self.dt**3/2,   self.dt**4/2 ],
                          [0,0,0, self.dt**3/2,     self.dt**2,     self.dt**4   ],
                          [0,0,0, self.dt**3/1,     self.dt**1,     self.dt**1/2 ]
                     ])

        # Keep track of the
        self.trace = deque(maxlen=MAX_BEE_TRACE_LENGTH)

        # Amount of missed detetions
        self.skipped_frames = 0

        # Amount of frames processed with this track
        self.processed_frames = 0

        # A set if determined track/bee characteristics
        self.tags = set()
        self.reportedTags = set()

        # The first detection which created this track
        self.firstPosition = None

        # Whether the track is underneath of a group of bees
        self.inGroup = False

        self.__tagCnts = {}

    def setTrackName(self, name):
        """! Sets the name printed next to the bee in previews
        @param name     A string representing the bees name
        """
        self._name = name

    def addTag(self, tag):
        """! Add a tag the to track. Tag could be on of TAG_WESPE, TAG_VARROA, TAG_COOLING, TAG_POLLEN
        @param tag      The tag to add
        """
        if tag not in self.__tagCnts:
            self.__tagCnts[tag] = 0
        self.__tagCnts[tag] += 1

        # Bees cooling the hive stay at the same position for a long time
        # and will pass the classification network multiple times.
        # To harden the detection, wait for at least 5 detection
        if tag == TAG_COOLING and self.__tagCnts[tag] < 5:
            return

        # Report to statistics
        if tag not in self.reportedTags:
            _dh = getStatistics()
            _dh.addClassificationResultByTag(self.trackId, tag)

        # Add the tag
        self.tags |= set((tag,))
        self.reportedTags |= set((tag,))

    def imageClassificationComplete(self, result):
        """! Merge classification results into this track
        @param results  A tuple, any of (TAG_WESPE, TAG_VARROA, TAG_COOLING, TAG_POLLEN)
        """
        values = [TAG_WESPE, TAG_VARROA, TAG_COOLING, TAG_POLLEN]
        for item in values:
            if item in result:
                self.addTag(item)

    def setPosition(self, position):
        """! Forces the position, which is reprenseted by the kalman filter, to the given position
        @param  position    List continaing [x,y] coordinates
        """
        self.KF.x[0] = position[0]
        self.KF.x[3] = position[1]

        # Add the position to the trace
        if len(self.trace) == 0:
            self.firstPosition = position
        self.trace.append(position)

    def predict(self):
        """! Perform the kalman prediction
        """
        self.KF.predict()
        self.lastPredict = self.KF.x
        return self.KF.x

    def correct(self, position):
        """! Perform the kalman correction
        @param  position    The actual position of the bee, to correct to
        """
        self.trace.append(position)
        self.KF.update(position[0:2])


class BeeTracker(object):
    """! The 'BeeTracker' manages all 'BeeTrack' instances.
    """

    def __init__(self, dist_threshold, max_frame_skipped, frame_size=(960, 540)):
        """! Initializes the 'BeeTracker'
        """
        super(BeeTracker, self).__init__()
        self.dist_threshold = dist_threshold
        self.max_frame_skipped = max_frame_skipped
        self.trackId = 0
        self.tracks = []
        self.names = loadWomanNames()
        self._frame_height = frame_size[1]
        self._frame_width = frame_size[0]

    def getTrackById(self, trackId):
        """! Returns the track object for the given ID (If it exists)
        @param  trackId     The track id to return a track for
        @return The 'BeeTrack' object, if it exists else None
        """
        for item in self.tracks:
            if item.trackId == trackId:
                return item
        return None

    def drawTracks(self, frame):
        """! Draw the current tracker status on the given frame.
        Draw tracks, names, ids, groups, ... depending on configuration
        @param  frame   The frame to draw on
        @return The resulting frame
        """

        # Draw tracks and detections
        for j in range(len(self.tracks)):

            # Only Draw tracks that have more than one waypoints
            if (len(self.tracks[j].trace) > 1):

                # Select a track color
                t_c = track_colors[self.tracks[j].trackId % len(track_colors)]

                # Draw marker that shows tracks underneath groups
                if DRAW_GROUP_MARKER and self.tracks[j].inGroup:
                    x = int(self.tracks[j].trace[-1][0])
                    y = int(self.tracks[j].trace[-1][1])
                    tl = (x-30,y-30)
                    br = (x+30,y+30)
                    cv2.rectangle(frame,tl,br,(0,0,0),10)

                # Draw rectangle over last position
                if DRAW_RECTANGLE_OVER_LAST_POSTION:
                    x = int(self.tracks[j].trace[-1][0])
                    y = int(self.tracks[j].trace[-1][1])
                    tl = (x-10,y-10)
                    br = (x+10,y+10)
                    cv2.rectangle(frame,tl,br,t_c,1)

                # Draw trace
                if DRAW_TRACK_TRACE:
                    for k in range(len(self.tracks[j].trace)):
                        x = int(self.tracks[j].trace[k][0])
                        y = int(self.tracks[j].trace[k][1])

                        if k > 0:
                            x2 = int(self.tracks[j].trace[k-1][0])
                            y2 = int(self.tracks[j].trace[k-1][1])
                            cv2.line(frame,(x,y), (x2,y2), t_c, 4)
                            cv2.line(frame,(x,y), (x2,y2), (0,0,0), 1)

                # Draw prediction
                if DRAW_TRACK_PREDICTION:
                    x = int(self.tracks[j].lastPredict[0])
                    y = int(self.tracks[j].lastPredict[3])
                    cv2.circle(frame,(x,y), self.dist_threshold, (0,0,255), 1)

                # Draw velocity, acceleration
                if DRAW_ACCELERATION or DRAW_VELOCITY:
                    l_p = self.tracks[j].lastPredict

                    l_px = int(l_p[0])
                    v_px = int(l_p[1])*10 + l_px
                    a_px = int(l_p[2])*10 + l_px
                    l_py = int(l_p[3])
                    v_py = int(l_p[4])*10 + l_py
                    a_py = int(l_p[5])*10 + l_py

                    if DRAW_VELOCITY:
                        cv2.line(frame, (l_px, l_py), (v_px, v_py), (255,255,255), 4)
                        cv2.line(frame, (l_px, l_py), (v_px, v_py), t_c, 2)

                    if DRAW_ACCELERATION:
                        cv2.line(frame, (l_px, l_py), (a_px, a_py), (255,255,255), 8)
                        cv2.line(frame, (l_px, l_py), (a_px, a_py), t_c, 6)

                x = int(self.tracks[j].trace[-1][0])
                y = int(self.tracks[j].trace[-1][1])
                if TAG_VARROA in self.tracks[j].tags:
                    cv2.circle(frame, (x-10, y-50), 9, (0, 0, 255), -1)
                    cv2.circle(frame, (x-10, y-50), 10, (0, 0, 0), 2)
                if TAG_POLLEN in self.tracks[j].tags:
                    cv2.circle(frame, (x-30, y-50), 9, (255, 0, 0), -1)
                    cv2.circle(frame, (x-30, y-50), 10, (0, 0, 0), 2)
                if TAG_COOLING in self.tracks[j].tags:
                    cv2.circle(frame, (x+10, y-50), 9, (0, 255, 0), -1)
                    cv2.circle(frame, (x+10, y-50), 10, (0, 0, 0), 2)
                if TAG_WESPE in self.tracks[j].tags:
                    cv2.circle(frame, (x+30, y-50), 9,(0, 0, 0), -1)
                    cv2.circle(frame, (x+30, y-50), 10, (0, 0, 0), 2)

                # Add Track Id
                if DRAW_TRACK_ID:
                    cv2.putText(frame, str(self.tracks[j].trackId) + " " + self.tracks[j]._name, (x,y-30),
                            cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255))
        # Draw count of bees
        if DRAW_IN_OUT_STATS:
            _dh = getStatistics()
            bees_in, bees_out = _dh.getBeeCountOverall()
            cv2.putText(frame,"In: %i, Out: %i" % (bees_in, bees_out), (50,50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 5)

        return frame

    def getLastBeePositions(self, frame_step):
        """! Returns a list of all tracks last positions
        @param frame_step   Only return those positions for every 'frame_step' frames processed
        @return A list of all detection that matched 'frame_step'
        """
        data = []
        for j in range(len(self.tracks)):
            track = self.tracks[j]
            if len(track.trace) and track.skipped_frames == 0 and track.processed_frames % frame_step == 0:
                data.append((self.tracks[j].trackId, self.tracks[j].trace[-1]))
        return data

    def isOutOfPane(self, pos):
        """! Returns whether the given position is near the top/bottom of the frame
        @param  pos     The position as list [x,y]
        @return True if near or out of frame else False
        """
        return pos[1] < 5 or pos[1] > (self._frame_height - 5)

    def _delTrack(self, trackId, count=False):
        """! Deletes a the given trackId from the 'BeeTracker' and checks whether this track
        corresponds to a bee that entered or left the hive.
        @param trackId    The trackId of the track to delete
        @param count      Whether to count the bee or not
        """
        track = self.tracks[trackId]
        if count:
            _dh = getStatistics()

            # Y-Position of first detection
            f_y = track.firstPosition[1]

            # Y-Position of last detection
            l_y = lastPosition = track.trace[-1][1]

            # Half of pane height
            pH = int(self._frame_height / 2)

            # Moved in
            if f_y > pH and l_y <= pH:
                _dh.addBeeIn()

            # Moved out
            if f_y < pH and l_y >= pH:
                _dh.addBeeOut()

        del(self.tracks[trackId])

    def update(self, detections: list, groups: list):
        """! Update all the tracks with the given list of detections.
        """
        # Convert ellipses to numpy array
        tmp  = np.zeros((len(detections), 5))
        for i, item in enumerate(detections):
            tmp[i] = np.concatenate((item[0], item[1], [item[2]]), axis=0)
        detections = tmp

        # Helper to mark matches
        def matched(item):
            t = item[1]
            d = item[2]
            used_tracks.append(t)
            used_detections.append(d)
            self.tracks[t]._lastDetect = detections[d]
            self.tracks[t].correct(detections[d])
            self.tracks[t].skipped_frames = 0
            self.tracks[t].processed_frames += 1

        # Calculate the distance on each track to the detections
        dist_list = []
        for num_t, item_t in enumerate(self.tracks):

            # Check whether this track is under a group of bees
            item_t.inGroup = False
            for g in groups:
                item_t.inGroup |= pointInEllipse(item_t.trace[-1], g)

            # Prepare the tracks
            # Assume that each trach has missed a detection/frame
            #  (will be correcty during matching)
            item_t.skipped_frames += 1

            # If the bee is inside of a group, then recude the kalman gain
            # to slow it down.
            if item_t.inGroup:
                item_t.KF.x[1] = item_t.KF.x[1] * 0.5
                item_t.KF.x[2] = item_t.KF.x[2] * 0.5
                item_t.KF.x[4] = item_t.KF.x[4] * 0.5
                item_t.KF.x[5] = item_t.KF.x[5] * 0.5
                item_t.skipped_frames -= 1

            pred = item_t.predict()
            for num_d, item_d in enumerate(detections):

                # Instead of only using the track prediction, try also to
                #  match with tracks last position
                if item_t.inGroup:
                    lastValidPosition = item_t._lastDetect
                    p_diff = (np.array([lastValidPosition[0],
                            lastValidPosition[1]]).reshape(-1,2) -
                            np.array(item_d[0:2]).reshape(-1,2))[0]
                    p_dist = math.sqrt(p_diff[0]*p_diff[0] + p_diff[1]*p_diff[1])
                    dist_list.append((p_dist, num_t, num_d))

                # Get the tracks prediction and calculate its distance to each detection
                p_diff = (np.array([pred[0], pred[3]]).reshape(-1,2) -
                        np.array(item_d[0:2]).reshape(-1,2))[0]

                p_dist = math.sqrt(p_diff[0]*p_diff[0] + p_diff[1]*p_diff[1])

                # Store each (distance, track_id, detection_id) for further processing
                dist_list.append((p_dist, num_t, num_d))

        # Sort distance list be least distance first
        dist_list = sorted(dist_list, key=lambda entry: entry[0])

        # Try to the best match for each track
        used_tracks = []
        used_detections = []

        for item in dist_list:
            dist, num_t, num_d = item

            # Skip those entry that refer to already assigned tracks or detections
            if num_t in used_tracks or num_d in used_detections:
                continue

            # Get the track
            track = self.tracks[num_t]

            # Get all detections for this track
            per_track = list(filter(lambda x: x[1] == num_t, dist_list))

            # Filter out used detections
            per_track = list(filter(lambda x: x[2] not in used_detections, per_track))

            # Filter by distance, distance is twice as large for new tracks
            distance = self.dist_threshold

            # This is a new track, it cannot make predictions about where the bee will go
            # Double the detection range
            if track.processed_frames > track.skipped_frames:
                distance = self.dist_threshold *2

            # This track recently missed some detections, but had a solid track before.
            # Double the detection range
            if track.processed_frames > track.skipped_frames:
                distance = self.dist_threshold *2

            # This Track is inside of a bee-group, so it cannot be detected
            # As we can only assume were the bee is going, double the distance
            if track.inGroup:
                distance = self.dist_threshold *2

            by_dist = list(filter(lambda x: x[0] < self.dist_threshold, per_track))

            # Only one finding, that is our match
            if len(by_dist) == 1:
                matched(by_dist[0])
            elif len(by_dist) > 1:
                print("Multiple findings")
                print(by_dist)

                # Simply use first one right now
                matched(by_dist[0])
                #for item in by_dist:
                #
                #    print(item, self.tracks[item[1]]._name)

        # Delete tracks that didn't match any of the last detections
        IN = 0
        OUT = 0

        for num_t in reversed(range(len(self.tracks))):
            item = self.tracks[num_t]

            # Remove tracks that were used just once
            if num_t not in used_tracks and \
                    item.skipped_frames > 0 and item.processed_frames == 0:
                self._delTrack(num_t, count=False)

            # Remove tracks that have more losses than hits
            elif num_t not in used_tracks and \
                    item.skipped_frames > item.processed_frames:
                self._delTrack(num_t, count=False)

            # Remove tracks that exceeded max frame skip
            elif self.tracks[num_t].skipped_frames > self.max_frame_skipped:
                self._delTrack(num_t, count=False)

            # Remove tracks that hit the entry or exit of the hive and have a frame skip
            elif self.isOutOfPane(self.tracks[num_t].trace[-1]):
                self._delTrack(num_t, count=True)

            # Remove tracks that hit the entry or exit of the hive and have a frame skip
            elif self.isOutOfPane([self.tracks[num_t].lastPredict[0], \
                    self.tracks[num_t].lastPredict[3]]):
                self._delTrack(num_t, count=True)

        # Create tracks for unmatched detections
        unmatched_detections = list(filter(lambda x: x not in used_detections, range(len(detections))))
        for item in unmatched_detections:

            # Only create new BeeTrack for bees that are on the pane
            if True:
                track = BeeTrack(self.trackId)
                track.setTrackName(random.choice(self.names))
                track._lastDetect = detections[item]
                self.tracks.append(track)
                track.setPosition(detections[item])
                self.trackId += 1

        return (IN, OUT)

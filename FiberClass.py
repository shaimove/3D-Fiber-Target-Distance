# FiberClass.py
import cv2
import numpy as np

class Fiber():
    """A tracked Fiber with a state including an ID, tracking
    window, histogram, and Kalman filter.
    """
    def __init__(self, ID, frame, track_window, point):
        # initialize ID, window track and criteria to end detection
        self.id = ID
        self.track_window = track_window
        self.term_crit = (cv2.TERM_CRITERIA_COUNT , 10, 1)
        
        # Mark the fiber
        x, y, w, h = track_window
        roi = frame[y:y+h, x:x+w]
        
        # Initialize the histogram
        roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255,cv2.NORM_MINMAX)
        
        # Initialize the Kalman filter
        self.kalman = cv2.KalmanFilter(4, 2) # (x_pos,y_pos,x_vec.y_vec), (x_pos,y_pos)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03
        
        # intial point supplied by the user
        cx,cy = point
        self.kalman.statePre = np.array([[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[cx], [cy], [0], [0]], np.float32)
        
    def update(self, frame):
        # get back projection of the histoogram
        back_proj = cv2.calcBackProject([frame], [0], self.roi_hist, [0, 180], 1)
        
        # meanshift tracker
        ret, self.track_window = cv2.meanShift(back_proj, self.track_window, self.term_crit)
        
        # get results
        x, y, w, h = self.track_window
        center = np.array([x+w/2, y+h/2], np.float32)
        
        # kalman filter
        prediction = self.kalman.predict()
        estimate = self.kalman.correct(center)
        center_offset = estimate[:,0][:2] - center
        self.track_window = (x + int(center_offset[0]),y + int(center_offset[1]), w, h)
        x, y, w, h = self.track_window
        
        # Draw the predicted center position as a circle
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])),4, (255, 0, 0), -1)
        
        # Draw the corrected tracking window as a rectangle.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)
        
        # Draw the ID above the rectangle
        cv2.putText(frame, 'ID: %d' % self.id, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),1, cv2.LINE_AA)
        


        
        
        
        
        
        
        
        

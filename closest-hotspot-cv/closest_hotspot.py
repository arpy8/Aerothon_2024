import cv2
import numpy as np


def find_closest_hotspot(curr_hotspot, rest_hotspots):    
    if len(rest_hotspots) == 0:
        return None
    
    rest_hotspots = np.array(rest_hotspots)
    distances = np.sqrt(np.sum((rest_hotspots - curr_hotspot) ** 2, axis=1))
    closest_hotspot = rest_hotspots[np.argmin(distances)]
    path.append(closest_hotspot)
    
    _ = find_closest_hotspot(closest_hotspot, np.delete(rest_hotspots, np.argmin(distances), axis=0))

def quadrant(x, y, screen_center_x, screen_center_y):
    if x > screen_center_x and y < screen_center_y:
        return 1
    elif x < screen_center_x and y < screen_center_y:
        return 2
    elif x < screen_center_x and y > screen_center_y:
        return 3
    elif x > screen_center_x and y > screen_center_y:
        return 4
    else:
        return 0


cap = cv2.VideoCapture(3)

COLOR_LOWER = [161, 155, 84]
COLOR_UPPER = [180, 255, 255]

while True:
    path = []

    _, frame = cap.read()
    
    screen_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    screen_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    screen_center_x = int(screen_width / 2)
    screen_center_y = int(screen_height / 2)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_lower = np.array(COLOR_LOWER, np.uint8)
    color_upper = np.array(COLOR_UPPER, np.uint8)
    mask = cv2.inRange(hsv, color_lower, color_upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    closest_distance = float('inf')
    closest_hotspot = None
    closest_quadrant = None

    hotspots = []

    for cnt in contours:
        for point in cnt:
            x, y = point[0]

        x, y, w, h = cv2.boundingRect(cnt)
        medium_x = int((x + x + w) / 2)
        medium_y = int((y + y + h) / 2)

        hotspots.append((medium_x, medium_y))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "x: " + str(medium_x) + " y: " + str(medium_y), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.line(frame, (medium_x, medium_y), (screen_center_x, screen_center_y), (0, 0, 0), 1)
        
        distance = np.sqrt((medium_x - screen_center_x) ** 2 + (medium_y - screen_center_y) ** 2)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_hotspot = (medium_x, medium_y)
            closest_quadrant = quadrant(medium_x, medium_y, screen_center_x, screen_center_y)

    _ = find_closest_hotspot(closest_hotspot, hotspots)

    for i in range(len(path) - 1):
        cv2.putText(frame, f"{i+1}", path[i]-20, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.line(frame, path[i], path[i+1], (0, 0, 255), 2)
    
    if closest_hotspot:
        cv2.putText(frame, f"Closest Hotspot: {closest_hotspot}", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, f"Quadrant: {closest_quadrant}", (60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.line(frame, (screen_center_x, screen_center_y), closest_hotspot, (255, 0, 0), 2)

    cv2.putText(frame, f"Total Hotspots: {len(contours)}", (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # cv2.imshow("mask", mask)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
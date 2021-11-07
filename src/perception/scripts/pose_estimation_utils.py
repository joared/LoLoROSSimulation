import cv2 as cv
import numpy as np

def plotAxis(img, translationVector, rotationVector, camera, points):
    points = points[:, :3].copy()
    #print(points)
    rad = 0.043
    projPoints, jacobian = cv.projectPoints(points, 
                                            rotationVector, 
                                            translationVector, 
                                            camera.cameraMatrix, 
                                            camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])

    zDir, jacobian = cv.projectPoints(np.array([(0.0, 0.0, rad)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    yDir, jacobian = cv.projectPoints(np.array([(0.0, rad, 0.0)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    xDir, jacobian = cv.projectPoints(np.array([(rad, 0.0, 0.0)]), 
                                      rotationVector, 
                                      translationVector, 
                                      camera.cameraMatrix, 
                                      camera.distCoeffs)

    center, jacobian = cv.projectPoints(np.array([(0.0, 0.0, 0.0)]), 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)

    center = center[0][0][0] / camera.pixelWidth, center[0][0][1] / camera.pixelHeight   
    for d, c in zip((xDir, yDir, zDir), ((0,0,255), (0,255,0), (255,0,0))):
        cx = center[0]
        cy = center[1]
        point1 = (int(round(cx)), int(round(cy)))
        point2 = (int(round(d[0][0][0] / camera.pixelWidth)), int(round(d[0][0][1] / camera.pixelHeight)))
        cv.line(img, point1, point2, c, 5)

    for p in projPoints:
        # to pixel coordinates
        x = int( p[0] / camera.pixelWidth )
        y = int( p[1] / camera.pixelHeight )
        radius = 2
        cv.circle(img, (x,y), radius, (0, 0, 255), 3)

def plotPoints(img, translationVector, rotationVector, camera, points, color):
    projPoints, jacobian = cv.projectPoints(points, 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])

    # to pixel coordinates
    for p in projPoints:
        x = int( p[0] / camera.pixelWidth )
        y = int( p[1] / camera.pixelHeight )
        radius = 3
        cv.circle(img, (x,y), radius, color, 3)

def projectPoints2(translationVector, rotationVector, camera, points):
    projPoints, jacobian = cv.projectPoints(points, 
                                        rotationVector, 
                                        translationVector, 
                                        camera.cameraMatrix, 
                                        camera.distCoeffs)
    projPoints = np.array([p[0] for p in projPoints])
    return projPoints
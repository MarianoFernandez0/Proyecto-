# SpermTrackingProject
date of initial commit 4 April 2018

Main project code files:
- VideoSegmentation_rev_11L.m
- VideoSpermTracker_rev_26L60sec.m
- analyzeTrackRecord_rev_7L.m

Code for Automatic Tracking and Motility Analysis of Human Sperm in Time-Lapse Images

by Leonardo F. Urbano, Puneet Masson, Matthew Vermilyea, and Moshe Kam

IEEE TRANSACTIONS ON MEDICAL IMAGING, VOL.36, NO.3, MARCH 2017

Order of execution:
1. VideoSegmentation_rev_11L.m
2. VideoSpermTracker_rev_26L60sec.m

Selecting video:
1. save the directory of the video as string into variable videoFile (line 19 in VideoSegmentation_rev_11L.m)
2. save the directory of the data file output from VideoSegmentation_rev_11L.m and the video into dataFile and videoFile, respectively (line 20,21 in VideoSpermTracker_rev_26L60sec.m)  

Warning: the video file must be located within the path set in MATLAB

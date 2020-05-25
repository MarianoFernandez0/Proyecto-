from evaluation import evaluation
import tifffile
import os
from oct2py import octave
import configparser
import pandas as pd

current_path = os.getcwd()
octave.addpath(current_path+'/SpermTrackingProject')


def tracking_urbano(config_params):
    '''
    Perform detectoin and tracking with the .m urbano implementation.
    The parameters must be specified in the config_params file.
    '''

    # Read params
    config_path = config_params

    config = configparser.ConfigParser()
    config.read(config_path)
    config.sections()

    dataFile = config["Input"]["DATAFILE_PATH"]
    reformat_dataFile = int(config["Input"]["reformat_dataFile"])

    videoFile_mp4 = config["Input"]["VIDEOFILE_MP4_PATH"]
    videoFile_tiff = config["Input"]["VIDEOFILE_TIFF_PATH"]
    numFrames = int(config["Input"]["numFrames"])
    fps = int(config["Input"]["fps"])
    px2um = float(config["Input"]["px2um"])
    ROIx = int(config["Input"]["ROIx"])
    ROIy = int(config["Input"]["ROIy"])

    csv_tracks = config["Output"]["CSV_TRACKS_PATH"]

    mttAlgorithm = int(config["Algoritmn params"]["mttAlgorithm"])
    PG = float(config["Algoritmn params"]["PG"])
    PD = float(config["Algoritmn params"]["PD"])
    gv = float(config["Algoritmn params"]["gv"])

    plotResults = int(config["Do"]["plotResults"])
    saveMovie = int(config["Do"]["saveMovie"])
    snapShot = int(config["Do"]["snapShot"])
    plotTrackResults = int(config["Do"]["plotTrackResults"])
    analyzeMotility = int(config["Do"]["analyzeMotility"])

    # Load tiff sequence
    tiff = tifffile.TiffFile(videoFile_tiff)
    # Perform detection step
    detected = evaluation(tiff)
    detected.to_csv(dataFile)

    # Perform tracking step
    octave.Tracker(dataFile, videoFile_mp4, csv_tracks, reformat_dataFile, numFrames, fps, px2um, ROIx, ROIy,
                   mttAlgorithm, PG, PD, gv, plotResults, saveMovie, snapShot, plotTrackResults, analyzeMotility,
                   nout=0)
    octave.clear_all(nout=0)
########################################################
#  START
########################################################

config_params = 'params.txt'

tracking_urbano(config_params)
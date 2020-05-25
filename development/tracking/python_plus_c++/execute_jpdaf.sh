#! /bin/bash
echo "Executing jpdaf script from bash..."

jpdaf_params="$PWD/jpdaf_tracking/config/params.txt"
jpdaf_detections="$PWD/sequences_for_jpdaf/detections.csv"
jpdaf_in="$PWD/sequences_for_jpdaf/video"

echo "----------------------------"

echo "----------------------------"
echo "Calling C++ script"
aux=$PWD/jpdaf_tracking/bin/jpdaf_tracker
$aux $jpdaf_params $jpdaf_detections $jpdaf_in

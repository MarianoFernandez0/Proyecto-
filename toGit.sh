#!/bin/sh
#
echo "---------- Capturing date and time ---------------------"
datetime=$(date '+%d/%m/%Y %H:%M:%S');

echo "---------- Moving to the folder: Timag  --------------"
cd /home/leo/Desktop/SPERM_ANALISIS_TOOL

echo "-------------- Git pulling -----------------------------"
git pull origin master

echo "--------------- Adding all -----------------------------"
git add .

echo "-----Write a mesage if necessary, otherwise press enter----"
read user_message

echo "-------------- Making commit ---------------------------"
git commit -m "Updated: $datetime. $user_message"  

echo "------------ Pushing to server -------------------------"
git push -u origin master

echo "------------ Done MoFo ---------------------------------"

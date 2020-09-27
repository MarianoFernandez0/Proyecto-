# build container. In the directory with the Dokcerfile
docker build -t tracking_tool .

# make sure that in de working directoy is a data folder with
# the config input and output
mkdir -p data && mkdir -p data/config && mkdir -p data/input && mkdir -p data/output

# run container that shares a folder with host
# OBS: second mount (-v) is just for testing. Is to be able to edit the project without 
# having to build the container every time
docker run -it \
--name tracking_inference \
-v "$(pwd)"/data:/data \
-v "$(pwd)"/docker_project:/docker_project \
tracking_tool

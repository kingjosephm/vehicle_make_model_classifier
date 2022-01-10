# Useful remote GPU server info

### Create new docker volume using data on GPU cluster

    docker volume create --name FLIR_data_matched \
      --opt type=none \
      --opt device=/home/kingj/FLIR_matched_gray_thermal \
      --opt o=bind

### Create new empty docker volume

    docker volume create --name MERGEN_FLIR_output

### Restart a previously instantiated, but stopped container in interactive mode

    docker start -i <container_ID or container_name>

### Reattach terminal to running container

    docker attach <container_ID or container_name>

### Detach terminal but keep container running

    CTRL-p CTRL-q  # key sequence

### Copy data within container to host machine

    docker cp <container_name>:<source_path> <destination_path>

`source_path` pertains to the path within the container. Note - this must be run from outside a running or stopped container. It is not possible to move data out of a removed (deleted) container.

### Other Docker commands
    $ docker ps  # lists running docker containers
    $ docker ps -a  # view all containers, both running and stopped
    $ docker container rm <container_name>  # deletes a container *CAUTION!*
    $ docker image ls  # view all docker images on host machine
    $ docker stats <container_name> --no-stream  # no-stream option presents just the current stats

### Create a Docker image and upload to DockerHub for future use
    
    $ docker login  # requires account, will prompt for username and password
    $ docker commit [OPTIONS] <container_ID or container_name> <dockerhub_path>:<tag>
    $ docker push <dockerhub_path>:<tag>

# Linux commands
### Move individual local data onto FTP server 
    $ scp file1 file2 <credentials>:<remote_dir> # Do this before ssh remote in

### Move directory and all contents from local drive onto FTP
    $ scp -r <dir> <credentials>:<remote_dir>

### Move data from FTP server to local drive
    $ scp username@remote:<remote_path> <local_path> # must be done outside remote session

### Unzip files
    $ unzip myzip.zip
    $ tar -xf train.tar.xz # unzips tar file

### Read text file (e.g. log.txt) in container

    $ cat <file>

### Exit environment (including Docker interactive, which stops container)
    $ exit

### To view available GPUs and their usage
    $ nvidia-smi

### To get IDs of all GPUs
    $ nvidia-smi --list-gpus
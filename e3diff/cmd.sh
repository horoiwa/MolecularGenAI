
CMD=$1

if [ $CMD = "build" ]; then
    echo "hello"
else if [ $CMD = "run" ]; then
    docker run --rm --gpus all -it tfgpu /bin/bash
else
    echo "Invalid cmd $CMD"
fi

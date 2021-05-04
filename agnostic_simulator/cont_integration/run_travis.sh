#!/bin/bash

docker build . -t agnostic_simulator_test_image
docker run -it -v $TRAVIS_BUILD_DIR:/root/agnostic_simulator --name agnostic_simulator_test_container agnostic_simulator_test_image ./agnostic_simulator/cont_integration/run_test.sh

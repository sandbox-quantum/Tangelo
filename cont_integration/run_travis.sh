#!/bin/bash

docker build . -t qsdk_test_image
docker run -it -v $TRAVIS_BUILD_DIR:/root/qsdk --name qsdk_test_container qsdk_test_image ./qsdk/cont_integration/run_test.sh

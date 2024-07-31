New-Item -ItemType Directory -Path ./build

cd ./build

cmake -G"NMake Makefiles" -DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/opencv_contrib-4.x/modules ../opencv/opencv-4.x -DCMAKE_INSTALL_PREFIX="$myRepo/install/$RepoSource"

cmake --build .  --target install --config release
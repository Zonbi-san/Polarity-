wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
Expand-Archive -Path ./opencv.zip
Expand-Archive -Path ./opencv_contrib.zip
#downloading opencv and extracting it

New-Item -ItemType Directory -Path ./build
set-location ./build

$CMAKE_OPTIONS = '-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON'
$CMAKE_GENERATOR_OPTIONS = '-G "Ninja"' #switch to whatever generator you use but the ones with long names is not working for some stupid reason
$P_REPO_LOCATION = 'C:\Users\ourpc\CLionProjects\Polarity-' #change ya path to whereever polarity root folder is

Write-host "$CMAKE_OPTIONS"
Write-host "$CMAKE_GENERATOR_OPTIONS"
Write-host "$REPO_LOCATION"
cmake "$CMAKE_GENERATOR_OPTIONS" "$CMAKE_OPTIONS" -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/opencv_contrib-4.x/modules -DCMAKE_INSTALL_PREFIX="$P_REPO_LOCATION/opencv" "../opencv/opencv-4.x"
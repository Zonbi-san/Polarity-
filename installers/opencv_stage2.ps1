#wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
#wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
wget -O MinGW64.zip https://github.com/brechtsanders/winlibs_mingw/releases/download/14.1.0posix-18.1.8-12.0.0-ucrt-r3/winlibs-x86_64-posix-seh-gcc-14.1.0-llvm-18.1.8-mingw-w64ucrt-12.0.0-r3.zip
#New-Item -ItemType Directory -Path ./opencv
#New-Item -ItemType Directory -Path ./opencv_contrib
Expand-Archive -Path ./opencv.zip
Expand-Archive -Path ./opencv_contrib.zip
Expand-Archive -Path ./MinGW64.zip
#downloading opencv and extracting it

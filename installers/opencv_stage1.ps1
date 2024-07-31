# Invoke-WebRequest -Uri https://github.com/microsoft/winget-cli/releases/download/v1.3.2691/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle -OutFile .\MicrosoftDesktopAppInstaller_8wekyb3d8bbwe.msixbundle
 winget install Microsoft.VisualStudio.2022.BuildTools Kitware.CMake GNU.Wget2 7zip.7zip 
# uncomment the first line if you dont have winget installed

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
expand ./opencv_contrib.zip ./opencv_contrib
expand ./opencv.zip ./opencv
# downloading opencv and extracting it
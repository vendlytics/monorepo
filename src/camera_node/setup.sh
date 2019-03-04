# Setup script for first thing after Ubuntu 16.04 is installed on Up board.
# from: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md

# 1. register server's public key
sudo apt-key adv --keyserver keys.gnupg.net --recv-key C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C8B3A55A6F3EFCDE

# 2. Add server to list of repositories
sudo add-apt-repository "deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main" -u

# 3. install libraries
sudo apt-get -y install librealsense2-dkms
sudo apt-get -y install librealsense2-utils

# optional: install developer and debug packages
# sudo apt-get install librealsense2-dev
# sudo apt-get install librealsense2-dbg

# verify kernel is updated with (should include `realsense` string)
modinfo uvcvideo | grep "version:"


###################
# Update firmware #
###################
echo 'deb http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo http://realsense-hw-public.s3.amazonaws.com/Debian/apt-repo xenial main' | sudo tee /etc/apt/sources.list.d/realsensepublic.list
sudo apt-key adv --keyserver http://keys.gnupg.net/ keys.gnupg.net --recv-key 6F3EFCDE
sudo apt-get update
sudo apt-get install intel-realsense-dfu*

# need to get the latest firmware from: 
# https://downloadcenter.intel.com/download/27522/Latest-Firmwarefor-Intel-RealSense-D400-Product-Family?v=t

lsusb

# find the bus and device number of the device, and specify in:
# intel-realsense-dfu –b 002 –d 003 –f –i ~/Downloads/Signed_Image_UVC_5_10_3_0.bin

# confirm that you've downloaded the right firmware with
intel-realsense-dfu -p

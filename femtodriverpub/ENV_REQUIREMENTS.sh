#!/bin/bash

# You can put arbitrary shell commands in here.
# Docker will also make use of these commands, so these must work in docker.
# Think hard before introducing these kind of dependencies.

function mac_install {
    # install function for mac
    # Check if already installed and install if not
    brew list ${1} || brew install ${1}
}

function ubuntu_install {
    # install function for mac
    # Check if already installed and install if not
    dpkg -s ${1} || apt-get install -y ${1}
}

# install packages. Unfortunately there is different method for each OS
if [[ ${OSTYPE} == "linux"* ]]; then
    DISTRO=$(awk -F= '/^ID=/{print $2}' /etc/os-release)

    if [[ ${DISTRO} == *"ubuntu"* ]]; then :
        # ==================== Ubuntu dependencies here ====================
        #ubuntu_install hiredis hiredis-devel
        # ==================================================================

    elif [[ ${DISTRO} == *"centos"* ]] ; then :
        # ==================== CentOS dependencies here ====================
        #yum install -y hiredis hiredis-devel
        # ==================================================================
    fi

elif [[ ${OSTYPE} == "darwin"* ]]; then :
    # ==================== mac dependencies here =======================
    #mac_install hiredis hiredis-devel
    # ==================================================================
fi

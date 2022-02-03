#!/usr/bin/env bash
HERE=$(dirname "$0")
VERSION=${1:-"stable"}
REPO=${2:-"https://github.com/keras-team/autokeras.git"}
PKG=${3:-"autokeras"}
if [[ "$VERSION" == "latest" ]]; then
    VERSION="master"
fi

vers=$(lsb_release -d | awk '{print $2}')

# creating local venv
. ${HERE}/../shared/setup.sh ${HERE} true
if [[ -x "$(command -v apt-get)" ]]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    SUDO mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    SUDO apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    SUDO add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    SUDO apt-get update
    SUDO apt-get -y install cuda
fi
PIP install --no-cache-dir -r ${HERE}/requirements.txt

if [[ "$VERSION" == "stable" ]]; then
    PIP install --no-cache-dir -U ${PKG}
elif [[ "$VERSION" =~ ^[0-9] ]]; then
    PIP install --no-cache-dir -U ${PKG}==${VERSION}
else
    TARGET_DIR="${HERE}/lib/${PKG}"
    rm -Rf ${TARGET_DIR}
    git clone --depth 1 --single-branch --branch ${VERSION} --recurse-submodules ${REPO} ${TARGET_DIR}
    PIP install -U -e ${TARGET_DIR}
fi

PY -c "from autokeras import __version__; print(__version__)" >> "${HERE}/.setup/installed"

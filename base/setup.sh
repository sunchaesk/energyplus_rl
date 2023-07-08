#!/bin/sh

EPLUS_TAR_LINK=https://github.com/NREL/EnergyPlus/releases/download/v23.1.0/EnergyPlus-23.1.0-87ed9199d4-Linux-CentOS7.9.2009-x86_64.tar.gz
EPLUS_TAR_FILE=EnergyPlus-*.tar.gz
EPLUS_DIR_OG=EnergyPlus-*/
EPLUS_DIR=Eplus

wget $EPLUS_TAR_LINK

tar -xvzf EnergyPlus-*.tar.gz

mv $EPLUS_DIR_OG $EPLUS_DIR

./$EPLUS_DIR/energyplus --help

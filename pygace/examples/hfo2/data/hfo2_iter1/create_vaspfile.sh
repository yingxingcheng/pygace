#!/bin/bash

for x in `ls -d [0-9]*`;do cd $x; runstruct_vasp -nr ; cd ../ ;done

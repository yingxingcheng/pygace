#!/bin/bash

for x in `ls -d [0-9]*`;do 
    ls $x/energy > /dev/null 2>&1
    [ ! $? -eq 0 ] && echo $x
done

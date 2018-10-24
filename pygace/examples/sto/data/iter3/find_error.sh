#!/bin/bash

for i in `ls -d [0-9]*`; do 
    ls $i/error > /dev/null 2>&1
    [ $? -eq 0 ] && echo $i
done 

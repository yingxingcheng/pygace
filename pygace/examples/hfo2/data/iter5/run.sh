#!/bin/bash

rm maps_is_running run.log
rm gs_connect.out gs.out gs_str.out
rm run.log
mmaps -d > run.log 2>&1 &

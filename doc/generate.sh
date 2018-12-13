#!/bin/bash

cd source
rm pygace.rst pygace.examples.rst pygace.examples.hfo2.rst pygace.examples.sto.rst modules.rst 
cd ../
sphinx-apidoc -o source ../pygace

make clean
make html

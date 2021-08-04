#!/bin/bash

cd source
files=(
pygace.rst
pygace.demo.rst
pygace.demo.hfo2.rst
pygace.demo.sto.rst
modules.rst
)

for f in $@[files];do
    if [ -f $f ]; then
        rm $f
    fi
done

# rm pygace.rst pygace.examples.rst pygace.examples.hfo2.rst pygace.examples.sto.rst modules.rst
cd ../
sphinx-apidoc -o source ../pygace

make clean
make html
make latex

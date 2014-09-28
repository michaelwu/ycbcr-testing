#!/bin/bash

CXXFLAGS="-O2 `libpng15-config --cflags`"
LDFLAGS="`libpng15-config --ldflags`"

g++ -g -Wall convert.cpp $CXXFLAGS $LDFLAGS -msse2 -o convert &&
g++ -Wall convert.cpp $CXXFLAGS -msse2 -S -o convert.s &&
g++ -g -Wall convert.cpp $CXXFLAGS $LDFLAGS -mavx -o convertavx &&
g++ -Wall convert.cpp $CXXFLAGS -mavx -S -o convertavx.s

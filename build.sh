#!/bin/bash

g++ -O2 -Wall convert.cpp `libpng15-config --cflags --ldflags` -msse2 -o convert &&
g++ -O2 -Wall convert.cpp `libpng15-config --cflags` -msse2 -S -o convert.s &&
g++ -O2 -Wall convert.cpp `libpng15-config --cflags --ldflags` -mavx -o convertavx &&
g++ -O2 -Wall convert.cpp `libpng15-config --cflags` -mavx -S -o convertavx.s

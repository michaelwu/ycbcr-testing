#!/bin/bash

g++ -O2 -Wall convert.cpp -msse2 -o convert &&
g++ -O2 -Wall convert.cpp -msse2 -S -o convert.s &&
g++ -O2 -Wall convert.cpp -mavx -o convertavx &&
g++ -O2 -Wall convert.cpp -mavx -S -o convertavx.s

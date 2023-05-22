#!/bin/bash
gcc test.c libbcalcdds.so -o test.so -Wl,-rpath=$(pwd)

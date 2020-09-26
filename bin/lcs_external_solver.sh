#!/bin/bash
# Call Marko's LCS implementation for the provided instance or tmp.lcs.

fname=${1:-tmp.lcs}
LCS -ifile $fname -guidance 1 -beta 10 | grep 'value'

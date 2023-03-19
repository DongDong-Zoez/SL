#!/bin/sh

echo "Finding all .py files in the current directory"
echo Find files: *.py
for file in *.py
do
    echo "Executing $file saveing result to assets" 
    python3 $file
done

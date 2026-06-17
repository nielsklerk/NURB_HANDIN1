#!/bin/bash

# If you get a permission denied error for run.sh itself, run this line in the terminal:
chmod +x ./run.sh

# If you get errors about weird (return) characters, and/or you edited run.sh on Windows, run this command:
dos2unix ./run.sh &>/dev/null

# Make sure you do NOT run in a virtual environment (e.g. conda, uv),
# or your results may be different than when we run your code.
# On the STRW computers, you may need to run "module purge" if you load any modules at startup.

pythonversion="$( python3 --version | cut -d' ' -f2 )"
if [ "${pythonversion}" != "3.9.25" ] ; then
    echo "WARNING: Python version is different from the default vdesk one (${pythonversion} vs 3.9.25), this may or may not cause differences/errors."
fi

matplotlibversion="$( python3 -m pip list | grep "matplotlib " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${matplotlibversion}" != "3.9.0" ] ; then
    echo "WARNING: Matplotlib version is different from the default vdesk one (${matplotlibversion} vs 3.9.0), this may or may not cause differences/errors."
fi

numpyversion="$( python3 -m pip list | grep "numpy " | tr -s ' ' | cut -d' ' -f2 )"
if [ "${numpyversion}" != "1.26.4" ] ; then
    echo "WARNING: Numpy version is different from the default vdesk one (${numpyversion} vs 1.26.4), this may or may not cause differences/errors."
fi

# Check if black formatter is installed
if ! python3 -m black --version &>/dev/null ; then
    echo "Black formatter not found. Installing..."
    python3 -m pip install black
fi

# Format all python files
echo "Uniformly formatting Python code..."
python3 -m black .

echo "Clearing/creating the plotting directory..."
if [ ! -d "Plots" ]; then
  mkdir Plots
fi
rm -rf Plots/*

echo "Downloading data files..."
if [ ! -d "Data" ]; then
  mkdir Data
fi
cd Data
# Download the satellite galaxy data files if they don't exist yet. Make sure to exclude the .txt files from your handin, as they are large and we have them already
if [ ! -f "satgals_m11.txt" ]; then
    wget -q https://home.strw.leidenuniv.nl/~daalen/Handin_files/galaxy_data.txt
fi
# Move back to the main directory
cd ..


echo "Running Python script for Exercise 1: Simulating the solar system..."
python3 simulating_solar_system.py

echo "Running Python script for Exercise 2: Calculating potentials..."
python3 potentials.py

echo "Running Python script for Exercise 3: Spiral and elliptical galaxies..."
python3 spiral_elliptical_galaxies.py

echo "Compiling LaTeX..."
pdflatex -interaction=batchmode NUR_B_handin.tex
pdflatex -interaction=batchmode NUR_B_handin.tex &>/dev/null 

echo "run.sh completed! Don't forget to hand in a clean version of this directory."
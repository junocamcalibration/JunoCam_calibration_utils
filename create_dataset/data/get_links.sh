#!/bin/zsh

# this file gets the links from cycle 22 through 29 and downloads the fits and README files
# into the required folder structure for create_img_pairs.py
# make sure to run this code inside the data/ folder

mkdir -p HST_fits

for cycle in 22 23 24 25 26 27 28 29 30
do
	for substring in "readme.txt" ".fits"
	do
		url="https://archive.stsci.edu/missions/hlsp/opal/cycle"${cycle}"/jupiter/"
		outfile="cycle"${cycle}"links"
		wget -q -nH -nd $url -O - | grep $substring | cut -f6 -d\" > $outfile
		mkdir -p HST_fits"/"cycle${cycle}
		while read map
		do 
			mapurl=$url$map
			wget $mapurl -O HST_fits"/"cycle${cycle}"/"${map}
		done < $outfile
	done
done

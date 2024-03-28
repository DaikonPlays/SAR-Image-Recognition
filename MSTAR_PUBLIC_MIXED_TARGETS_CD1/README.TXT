
MSTAR/IU (Public) Mixed Targets 

The targets are: 2S1, BRDM-2, BTR-60, D7, T62, ZIL131, ZSU-23/4, and SLICY

The data is organized in directories that can be 
accessed using an Internet Web browser.  The START.HTM file may 
be opened with a Web browser (using 'open file') 
The binary data files may be read using 
appropriate software.

DIRECTORY STRUCTURE

The highest level
is Depression Angle, then Collection, Scene, and Target.

example:
	/15_DEG/
		COL2/
		     SCENE1/
			    D7/
			    T62/
	/45_DEG/
	        COL2/
		     SCENE1/
			    2S1/
			    BRDM-2/
			    ZSU_23-4/
		     SCENE2/
		 	    ZSU_23-4/
		     SCENE3/
			    BRDM-2/


FILE FORMAT

Each file is constructed with a prepended, variable-length, 
Phoenix formatted (ASCII) header which contains detailed ground 
truth and sensor information for the specific chip.  Following 
the Phoenix header is the data block.  The data block is written 
in Sun floating point format and is divided into two blocks, a 
magnitude block followed by a phase block.  Byte swapping may be 
required for certain host platforms.  Tools for reading and 
manipulating the header information may be found at 
https://www.sdms.afrl.af.mil

Also for each chip there is a corresponding JPEG image. 
NOTE: JPG files are for illustration purposes ONLY,
during the conversion from raw format to jpeg(.JPG) format,
the resolution was degraded.   In each target directory,
there is a mosaic of all the chips in that directory.
Because of screen size, there are four of these mosaics per target.  
Each mosaic is a quadrant of the total aspect, ie: 0 degrees to 89 degrees,
90 degrees to 179 degrees, etc.  For illustration purposes, a photograph of
each target has been included.  These photos are in the /HTMPICS/
directory and have the extension *LG.JPG.  The photo is viewable at the top 
of each page.


For more information, please send an email message to:

sdms_help@mbvlab.wpafb.af.mil

or Check out the following URL:

https://www.sdms.afrl.af.mil/

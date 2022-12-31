## SIED parameters SEVIRI
import os
import datetime

global_params= {
		# The data regarding a specific datetime. Can be replaced by
		# the name of the file directly
		# Data type is specified for MODIS and SEVIRI data (gmi is
		# also loaded with data_type SEVIRI.
		# It was because a specific reader was made for each of these 2
		# sensors in order to have the same input format whatever the
		# sensor was. It may not work if the fields in your .nc
	        # data file have different names than expected in the readers.
		'data_type' : 'REGULAR_OBS',
                # variable name in netcdf file
		'variable': 'sea_surface_temperature',
		# fronts are split in smaller fronts whose length will not
		# overcome max_length_of_fronts.
		'max_length_of_fronts': 1000,
		# there will be "number_of_flags" flags to sort the fronts from
		# the weakest ones (flag max) to the strongest ones (flag 1).
		# There is approximately the same number of fronts in each flag
		'number_of_flags':6,
		# The time_span of front data file for display in SEAScope
		'minutes_delta': 30,
		# spatial resolution of the sensor in km
		'sensor_resolution': 5,
		# coordinates box: [min_lon, max_lon, min_lat, max_lat]
		#'box' : [-10, 2, 32, 42],
		'box' : [17, 33, -40, -32],
        'out_pattern': 'seviri_sst',
		#'box' : [-25, -15, 15, 25],
		#'box' : [-24, -16, 18, 26]
		#'box': [24,40,-40,-24]
		}

# Compare with Cayula&Cornillon parameters :
	# < : param is smaller than the one set by Cayula&Cornillon
	# = : param is equal to the one set by Cayula&Cornillon
	# > : param is bigger than the one set by Cayula&Cornillon

# PART 1 is histogram analysis. IE the map is scanned along ROW and COLUMN axes
# and a window is considered for some (row,column) coordinates. The following
# parameters are used to select the characteristics of the windows (size,
# stride), but also some thresholds to qualify a front window.

params1= {# List of the length of the side for each window (km) (=)
          'hist_window_km': [80, 50, 30],
          # if a window size is smaller than 6 pixels theoretically, then
          # histogramWindowSize is set at minSizeOfWindow.
	      'min_len_window': 6,
	      # The step from one window to another when scanning a data matrix
	      # is this parameter multiplied by the window size in pixels.
          'hist_step_window_ratio': 1/8,
          # Threshold of non masked pixels in a window, if below the window is
          # not considered anymore. (<) (between 0 and 1)
          'min_perc_valid_pts': 0.3,
          # Threshold for theta, the criterion function used by Cayula and
          # Cornillon to qualify a window containing a front. The criterion
          # function increases when two populations are distinguishable in the
          # histogram of the values in the window (>) (between 0 and 1)
          'theta_min': [0.74,0.72,0.71],
          # Threshold: min temperature difference between 2 populations
          # (converted to int16). Correspond to the data sensitivity
          'data_sensitivity': 6,
          # Threshold: minimal proportion of one population in a window (>)
          # (between 0 and 1)
          'min_perc_pop': 0.3,
          # Single population cohesion threshold. Increases when pixels that
          # belong to the same population are close to each other
          'min_single_cohesion': [0.88,0.82,0.78],
          # Global population cohesion threshold. Increases when increases
          # when both populations are separated
          'min_global_cohesion': [0.92,0.88,0.86],
          # Merging scale methods MAX, MEAN,  or RMS:
          # The algorithm scans the map for each scale and then makes a map of
          # front probability based on every scale. Several possibility
          # to merge the scales:
          # MAX: For each pixel consider the max value of each scale (recommended)
          # MEAN: For each pixel: proba = (sum of all theta encountered in a
          # window for each scale over number of times this pixel appeared in a
          # window)
          # MEAN: For each pixel: proba is the sum on scales of all theta over
          # number of times this pixel appeared in a window.
          # RMS For each pixel: proba is the rms on scales of all theta over
          # number of times this pixel appeared in a window.
          'merging_scales_method': 'MAX',
          # The max difference between the most extreme temperature and its
          # nearest neighbor in a window. If not the most extreme value is
          # deleted.
          'delete_extrem': 50,
	      # Min size of a 8-neighboring front group of pixels in a window,
	      # divided by windowSize. The front group is deleted if too small.
          'min_size_of_edge': 0,
          # Min size of a 4-neighboring population group, the group is given to
          # the other population if too small.
          'min_size_of_group': 0.15
		  }

# PART 2 and 3 are related to contour followingT. Part 2 considers the fronts
# probability calculated in part 1 to compute a multidirectional operator
# (dimensions (length of map, width of map, number of directions)) and highlight
# ridges in the map of probability, which are fronts... Part 3 is pure
# contour following, considers a start pixel and follows the ridges thanks to
# the multidirectional operator. Part 2 and 3 have the same parameters file,
# because one of them (ndir) is used in both part and then should be the same
params23_fusion_proba = {# Length of rays (in pixels). Number of pixels
                         # considered along a direction to compute the operator
                         'r':[4],
                         # Step of angular discretization (number of direction)
                         'ndir':[16],
			 # Min length of fronts (in pixels) in part 3
			 'min_length_of_contour':[9],
			 # Max angle in the front broken line
			 'max_angle_with_last_direction':[90],
			 # If the front ends on itself, the length of the loop
			 # created is checked. if less than
			 # min_perimeter_circle (in pixels) then the loop is
			 # removed
			 'min_perimeter_circle':[30],
                         # In part2, the multidirectional operator is computed
                         # as a linear combination of 3 attributes:
                         # contrast, homogeneity, and curvature.
                         # The linear weight of contrast in the operator
			 'alpha':1,
			 # The linear weight of homogeneity in the operator
			 'beta':0.8,
			 # The linear weight of curvature in the operator 
			 'gamma':1
					   }

# PART 4 is post processing, ie the goal is to remove fronts that should not
# have been detected. The 3 parameters correspond to 3 thresholds for 3 rules.
# A rule is respected if the value related to it is larger than the threshold.
# If a front does not respect any of the 3 rules, then there is no reason to
# keep it -> it is deleted.

params4_fusion_proba = {# First rule: the length of the front in pixels
                        'length_threshold':[20],
                        # Second rule: the mean of front probability along the
                        # front (strength)
			'Strength_threshold':[0.2],
			# Third rule: the distance between the extremities of
			# the front (in order to filter small loops)
			'extremity_threshold':[10]}

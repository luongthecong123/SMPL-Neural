This folder fix the bug in STAR
STAR is a better version of SMPL, its shape blend shapes are trained on a larger dataset than that of SMPL

Run main.py

use STAR to create synthetic data from AMASS mocap data for validation subset:

	- Given AMASS subset SSM's pose coefficient for the movement punch and kick, use the first 24 element in pose array (see Amass_preprocess)
	- Take the first scan of each subject in DFaust
	- Find the shape and pose coefficient from STAR that minimize scan and STAR's output difference:
		+ in class STAR: num_betas = 10 for fastest optimization, 
				 num_betas = 300 for lowest difference between scan and STAR-generated shape
		+ note that STAR also returns translation


	- After found the shape coefficient for each subject:
		+ Now we have pose coefficient from AMASS mocap and shape coefficient, trans from optimization above
		+ Use those to create valdiation data for each subject on motion "punch and kick".


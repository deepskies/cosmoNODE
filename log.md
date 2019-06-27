

Time-of documentation:
6/26/19

	11:04 trained using anandijain/sip/gym-sip/regression.py neural network on training_set.csv of LSST and tested on the test_set_sample.py
		
		+ the model is just linear 5 -> 20 -> 3 -> 8 -> 4 -> 1 (definitely not optimal, but wanted to just train on anything)
		+ the data was scaled -1 to 1 and the p-value for correct flux prediction was 0.1
		+ the inputs were all of the columns besides the flux label.
		+ this is most likey not proper, as flux_err for an input might not be kosher
		+ the accuracy on the training set was [correct guesses: 9475 / total guesses: 11107] ~ 0.8531 %
		+ (3:26 PM update) - this accuracy ^ is most likely false as the object_id was given as one of the inputs to the network, while I only trained on one epoch, this definitely will cause overfitting

	11:47 AM using matplotlib to interpret timeseries data
		 
		 + plotting each object with flux/time and colored according to band

	12:00 PM 

		+ journal meeting

	12:50
		
		+ minerva tour

	3:15 PM 
		
		+ class written for taking in data and merging.

	4:15 
		
		+ getting acquainted w tf2 for cross validating NODEs (#TODO)

	5:00

		+ weak tf model on merged semi-working, bugged loss

		+ testing on colab, something about my CPU was throwing errors/warnings



6/27/19

	8:30 
		+ got files to upload to colab (the model is not working, reason: input shape is wrong. the model can't learn from a single flux value)
		+ starting torch custom dataloader for NODE

	9:15 

		+ loader framework done, fitting shape so that the input tensor is the timeseries data and not an individual row

	9:30 
	
		+ each object can have a different number of datapoints, which means the nn needs to take in dynamic shapes 
		

	10:00 

		+ read thru Deep Skies guide, filled out timecard

		


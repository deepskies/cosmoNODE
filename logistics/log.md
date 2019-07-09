Time-of documentation:
6/26/19

	11:04 AM trained using anandijain/sip/gym-sip/regression.py neural network on training_set.csv of LSST and tested on the test_set_sample.py

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

	3:15

		+ class written for taking in data and merging.

	4:15

		+ getting acquainted w tf2 for cross validating NODEs (#TODO)

	5:00

		+ weak tf model on merged semi-working, bugged loss

		+ testing on colab, something about my CPU was throwing errors/warnings



6/27/19

	8:30 AM
		+ got files to upload to colab (the model is not working, reason: input shape is wrong. the model can't learn from a single flux value)
		+ starting torch custom dataloader for NODE

	9:15

		+ loader framework done, fitting shape so that the input tensor is the timeseries data and not an individual row

	9:30

		+ each object can have a different number of datapoints, which means the nn needs to take in dynamic shapes


	10:00

		+ read thru Deep Skies guide, filled out timecard

	10:20

		+ transfered repo to /deepskies/

		+ looking into dynamic shape methods:
			1. pad input tensors to all the same shape (easy and less cool)
				+ using tf.pad and finding the max num of rows for all objs


			2. dynamic

	11:00

		+ finished custom dataloader for pytorch

		+ writing test net for torch (building up to NODE)

	11:43

		+ fixed bugs with torch Dataset/Dataloader, padding finished

	12:50 PM

		+ simple mnist torch classifier modified to use LSST data

	2:00

		+ lunch

		+ shape debugging

	4:00

		+ working thru exponentially growing loss/nan outputs

		+ added data scaling using sklearn.preprocessing.min_max

	5:00

		+ general discussions on RL strategies and noise w Yunchong and Callista

6/28/19

	+ 8:00 AM

		+ setting up zoom, making a few edits on DS guide, overleaf

		+ added resources to guide for RL

		+ working on filling out the outline and goals of project as stated in the guide

			- updated google doc to fit template

	+ 10:00

		+ cosmoNODE now meets the minimum requirements for a deepskies project (i think)


7/1/19

	+ 9:47 AM

		+ reading through torchdiffeq to understand how the mnist model works,
			- because im still not positive how the one dimentional ODE is used for 2 dimensional inputs, like images

		+ i am realizing that i need to know more about ODEs

		+ found a keras implementation of ODENET

	+ 11:00  

		+ com meeting

	+ 12:20 PM

		+ starting to grasp how the ode is working on images,

		+ (n, 1, 28, 28) -> (n, 64, 26, 26) -> (n, 64, 13, 13)

7/2/19

	+ 10:15 AM

		+ ditching 2D non timeseries example for mnist to actually make headway on timeseries classification

		+ graph specific bands working

		+ linear model in torch trained
			- found that, since the value count distribution of targets is unequal, the model learns to just pick the most common class


	+ 3:00 - 5:00 PM
		RL and NODE meetings. decision to focus on 1D time for light curve prediction (NO CLASSIFICATION YET)

7/3/19

	+ 9:00 AM
		+ built torch dataloader for the task below

		+ reading through ode_demo and repurposing cosmoNODE/ode_demo.py to handle light curve data

	+ 2:40 PM

		+ ode_flux almost ready for testing, last leg of debugging and tensor shape fitting

		+ have a meeting at 3, wouldn't be able to finish, cleaned repository

	+ 3:00 - 4:30 PM

		+ RL meeting, back to ODE

		+ got the tensor datatypes worked out, but there is a shape issue



--
Starting new format

7/5/19
	+ Day goals:
		- I want to get ode_demo.py to run on one object (completed ~1:00 PM)

		- make more progress on latex document

		- clean up and improve google doc

	+ Log:
		- 10:00 AM:
			- dtype errors are so frustrating!!
			- net = net.float() is very useful
			- got odeint(func, y0, t0) to run!!

		- 11:00 AM:
			- up to backprop, got a gradient problem
			- fixed, ODE_demo now runs, however, loss is always zero
			- it is getting the pred_y dead on so it must have the answer
				- we understand why: by t0.size == [1] we basically are only trying to
				solve y for that point, we need to give it other values to calculate y for

		- 12:00 PM
			- batching almost working

			- t not strictly increasing error, added sort_values in loaders

			- got rid of sequence paddding, need to fix index errors in get_batch

			- index error fix written (pretty hacky)

		- 1:00 PM
			- ready for more rigorous testing and implementing multiple objects and
			separated passbands

			- JK (still getting loss of zero or shape errors)

		- 2:30 PM
			- okay i think i kinda see what they're doing w batching, for now,
			im only going to do one y0 at a time.


7/8/19
	+ Day goals:
		+ run the NODE with multiple objects and non 1D Y data (maybe flux, flux_err,
			and some others)

		+ shuffle data

		+ google doc and latex doc

		+ look into taking output of node and classifying

		+ colabs (basic colabs done at ~2:00 PM)

	+ Log:

		+ (8:30 - 11:00 AM) Updating jupyter notebook

		+ (11:00 AM - 12:30 PM) presentations

		+ 12:30 finished jupyter, now adding visualization for colab, and inf train

		+ 2:15 Two colabs made with some graphs. Loss decreases over time w NODE,
			but asymptotically and the loss[0] - loss[end] is pretty small.

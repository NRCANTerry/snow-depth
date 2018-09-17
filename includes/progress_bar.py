import sys

# function to create progress bar
def progress(count, total, status = ''):
	# bar parameters
	bar_length = 60
	filled_length = int(round(bar_length * count / float(total)))

	# create output
	percents = round(100.0 * count / float(total), 1)
	bar = '=' * filled_length + '-' * (bar_length - filled_length)

	# write to command line
	sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
	sys.stdout.flush()
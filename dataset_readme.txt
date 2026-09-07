partinfo_survey.csv
-------------------
Contains exactly one row per participant.
Columns:
	id: participant ID
	age: participant age
	gender: participant gender
	days_to_get_familiar: number of days the participant needed to get familiar with the N-Back test, as reported in the exit survey
	confirm/contradict: whether the participant felt the Oura ring mostly confirmed (1) or contradicted (0) their own feeling about their sleep, as reported in the exit survey

selfreport_nback_data.csv
-------------------------
Contains 1546 rows, where every row is one day for one participant where the participant went through the experiment application (trial = self-assessment + N-Back)
Columns:
	id: participant ID
	date: trial date
	time: trial time
	n_correct: number of correct answers out of 20 in the N-Back test
	timings: array with 20 entries, where array[i] gives the time it took to answer the i-th N-Back question
	correct_array: array with 20 entries, where array[i] is 1 if the i-th N-Back question was correct, else 0. Sums to 'n_correct'
	letters_shown: array with 23 entries that shows which letters were shown during the N-Back test. Missing for some days/participants because capturing this was patched in later during the experiment, but never used
	selfassessment_value: participant's self-assessment of their sleep for that day. {2: much better, 1: better, 0: no change, -1: worse, -2: much worse}
	nback_score: N-Back score calculated for that day (see publication)
	nback_available: whether N-Back data is available for that day and participant. Useful when merging with sleep data tables
	nback_valid: whether the N-Back data for that day and participant is valid. The trials before a participant got familiar to the test are invalid. One participant did N-Back with N=2 for some time; until this was corrected their data is invalid.
	selfassessment_available: whether sleep self-assessment data is available for that day and participant. Useful when merging with sleep data tables

oura_sleep.csv
--------------
Oura Sleep data for every participant.
Columns:
	id: participant ID
	rest: see https://cloud.ouraring.com/v2/docs

oura_daily_sleep.csv
--------------
Oura daily sleep summaries for every participant.
Columns:
	id: participant ID
	rest: see https://cloud.ouraring.com/v2/docs
# TODO --> #1
    # start program at half time of game -- *stat_collection.py can invoke this program
    # collect half time spread from vegas
    # copy data file to a tmp directory

    # TODO
        # watch for team ordering

    # TODO next: --> #2
        # the *stat_collection.py program should compute the final spread after the game is over...
        #     ... and then open the file saved in this program, and append the final spread
        # the final output file should be:
        #     vegas_spread,actual_spread

    #TODO different program: --> #3
    # start a prediction network for a file saved in tmp directory above
    # run prediction network for 500 training iterations, and then 100 prediction iterations
    #     prediction network should output final spread to file
    # open file with vegas spread
    # append prediction spread to vegas spread
    # save to file
    # final output file will be vegas_spread,actual_spread,prediction_spread
        # NOTE: spreads will be inverted in their sign -- +5 for my program means -5 for vegas

    # TODO next:
        # write a script that calls program 3 for all games

    # TODO next:
        # write a program that goes through all the files saved from this program, and
        #     ... and counts the number of winning predictions


# DESIGN ---------------------------

# stat_collection.py --> record_vegas.py
#     arguments:
#         team names
#         date value
#     design:
#         open vegas website
#         find 2nd half spread using similar strings
#             if teams not found, then vegas_spread = 1000
#         copy data file into data_tmp/
#         open data file
#         record current scores
#         record to file named team_1_team_2_vegas_date.csv
#             should look like: vegas_spread,team_1_halftime_score,team_2_halftime_score
#
# stat_collection.py completes
#     design:
#         open file team_1_team_2_vegas_date.csv
#         append final scores
#         record to same file
#             should look like: vegas_spread,team_1_halftime_score,team_2_halftime_score,team_1_final_score,team_2_final_score
#
#
# predict_2nd_half.py
#     design:
#         read in all files from tmp/
#         if file.read()[0] == 1000: do nothing
#         else:
#             data file will be the same as the file from tmp/ without the _vegas -- must be the
#             open data file
#             train on data for 500 iterations
#             predict on 150 iterations
#             compute final prediction spread
#             append to tmp/file
#                 should look like: vegas_spread,team_1_halftime_score,team_2_halftime_score,team_1_final_score,team_2_final_score,prediction_spread

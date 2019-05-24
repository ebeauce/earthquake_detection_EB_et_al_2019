# earthquake_detection_EB_et_al_2019
Codes used in the earthquake detection and location method presented in Beauce et al. 2019. A real data example is also provided.<br/>

# Prerequirements:

Template matching is done using the software Fast Matched Filter (FMF, Beaucé et al 2017, doi: [10.1785/0220170181](https://doi.org/10.1785/0220170181)), which is available at [https://github.com/beridel/fast_matched_filter](https://github.com/beridel/fast_matched_filter). cf. the [documentation](https://ebeauce.github.io/FMF_documentation/) to properly install FMF.<br/>

To compile both FMF and the codes provided here, you will need a C compiler and the cuda compiler nvcc if you wish to use the GPU codes. For example, gcc 4.8.5 (C compiler) with the tools from cuda 7.5 are known to work well.

# Procedure to follow:

- Download the repository.<br/>
- Go to the folder automatic_detection and compile the libraries using the Makefile by running the command line:

$ make

- Go back to the main folder and start running the scripts from the folder scripts. You should either use ipython (which is my personal preference) or add your current working directory to your python path. The first code to run is 00_download_data.py (for example with the command line ipython ./scripts/00_download_data.py), which downloads the data, the moveout grid and the classifier from my dropbox.<br/>
- Run 0_make_architecture.py, which creates different folders where the intermediate outputs will be stored.<br/>
- You are now ready to start the detection workflow. Run 1_calculate_CNR.py, which calculates the composite network response and extract a bunch of candidate template events and store them to the output folder.<br/>
- Run 2_extract_features.py, which calculates the features presented in Beauce et al. 2019 for all the events detected previously.<br/>
- Run 3_classify_candidate_template_events.py, which uses our classifier to select only the good candidate template events for subsequent matched-filter searches.<br/>
- Run 4_output_template_objects.py, which properly formates the database of template events.<br/>
- Run 5_matched_filter_search.py, which detect new events by using the template events in a matched-filter search.<br/>
- Run 6_automatic_relocation.py, which relocates the template events using the stack of all the newly detected events (more precisely, using the SVDWF). New templates will be stored in output/template_db_2/.<br/>

You can run scripts 5 and 6 iteratively if you change the paths written at the beginning of each script. template_db_1 and matched_filter_1 are for the first matched-filter search, and you can run a second matched-filter search by modifying those to template_db_2 and matched_filter_2. Between each matched-filter search, do not forget to run the script update_database_index.py.<br/>

Plotting functions are provided in 1_calculate_CNR.py, so that running:

$ ipython -i scripts/1_calculate_CNR.py

will allow you to enter a ipython terminal from where you can run the plotting functions to see what kind of events were detected.<br/>

This Github repository is still work-in-progress, and it should get easier to use in the future. If you have any question, don't hesitate to contact me at: ebeauce@mit.edu<br/>

Enjoy!

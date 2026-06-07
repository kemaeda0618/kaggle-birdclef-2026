AnuraSet 
-------------------------------
Date: 2023-06-15
-------------------------------

Description
-------------------------------
This dataset has 93378 samples of 3s audio which corresponds to 27 hours of human annotations.
The labels are 42 species of neotropical anurans. In the metadata file, each specie has 
a column where is 1 if any portion (independete of lenght and quality) are in the sample, and 0 else.
The dataset was built using soundscape recordings from passive acoustic monitoring in Brazil. The sites are
INCT17 (42.5%), INCT20955(33%) INCT41 (13.5%) and INCT4 (11%) 

The recordings were preprocessed by resampling the audio to 22050Hz and formatted to 16-bit depth. 
The recordings are trimmed to a fixed window length, but no filtering was applied.
The frequency limits are: (1, 10000). 
The construction used a sliding windows approach of 3s moving 1s (Overlapping of 2/3).
-------------------------------

Dictionary of data 
-------------------------------
-sample_name: the unique identifier of each sample that corresponds to a unique audio file in the audio folder and follows the structure {site}_{date}_{hour}_{start second}_{final second}.wav. The next 5 columns were constructed based on this column. 
-fname: raw audio filename extracted from a site and used by annotators to create weak labels. 
-min_t: second where the annotation starts in a fixed window length. 
-max_t: second where the annotation ends in a fixed window length. 
-site: identifier of the recording site. 
-date: datetime of the recording.
-subset: training or test subset. 
-species_number: total number of species in each sample. The sum of the next 42 columns per row.
{species}×42 Binary columns of each species where 1 if some portion of the call is in the sample, 0 else. The 42 species column names are the codes shown in Table 2
-------------------------------

Contact: jcanas@humboldt.org.co

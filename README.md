# Emotion-Recognition-from-Speech
A machine learning application for emotion recognition from speech.

Language: Python 2.7

# Authors

Mario Ruggieri

E-mail: mario.ruggieri@uniparthenope.it

# Dependencies

- [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis) for short time features extraction
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) for preprocessing, classification and validation

# Datasets

- [Berlin Database of Emotional Speech](http://emodb.bilderbar.info/start.html) [1]
- [DaFeX Dataset](https://i3.fbk.eu/resources/dafex-database-kinetic-facial-expressions) [2-3]

Download Berlin DB from the link.
Request DaFeX dataset following the link instructions. The code will generate automatically .wav files

# Usage

Long option | Option | Description
----------- | ------ | -----------
--dataset | -d | dataset type
--dataset_path | -p | dataset path
--load_data | -l | load dataset data and info and save them into a .p file
--extract_features | -e | extract features from data and save them into a .p file
--speaker_indipendence | -s | cross validation is made using different actors for train and test sets
--plot_eigenspectrum | -i | show eigenspectrum for each training set

Example:

    python emorecognition.py -d 'berlin' -p [berlin db path] -e -l

The first time you run the application, -l and -e options are mandatory because you need to extract data and features. Every time you change the feature extraction method and/or the dataset data you need to specify -e and/or -l to update your .p files.

# License

Please read LICENSE file.

# References

- [1] Burkhardt F., Paeschke A., Rolfes M., Sendlmeier W. and Weiss B., A Database of German Emotional Speech, Proceedings Interspeech 2005, Lissabon, Portugal
- [2] Battocchi, A.; Pianesi, F.; Goren-Bar, D.. A First Evaluation Study of a Database of Kinetic Facial Expressions (DaFEx). Proceedings of the 7th International Conference on Multimodal Interfaces ICMI 2005, October 04-06, 2005, Trento (Italy), pp. 214- 221. ACM Press New York, NY, USA.
- [3] Battocchi, A.; Pianesi, F.; Goren-Bar, D.. The Properties of DaFEx, a Database of Kinetic Facial Expressions. In Jianhua Tao, Tieniu Tan, Rosalind W. Picard (Eds.): Affective Computing and Intelligent Interaction, First International Conference, ACII 2005, Beijing, China, October 22-24, 2005, Proceedings. Lecture Notes in Computer Science 3784 Springer 2005, pp. 558-565.

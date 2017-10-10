from pyAudioAnalysis import audioBasicIO
import os
import subprocess as sp
import itertools

class Dataset:

	# Dataset object is composed of:
	# data 
	# targets
	# train and test sets for cross validation
	# classes dictionary to map classes to numbers

	def __init__(self,path,type,decode):
		self.type = type
		if type == "berlin":
			self.classes = {0:'W', 1:'L', 2:'E', 3:'A', 4:'F', 5:'T', 6:'N'}
			self.get_berlin_dataset(path)
		elif type == "dafex":
			self.classes = {0:'ang', 1:'dis', 2:'fea', 3:'hap', 4:'neu', 5:'sad', 6:'sur'}
			self.get_dafex_dataset(path,decode)

	def get_berlin_dataset(self,path):
		males = ['03','10','11','12','15']
		females = ['08','09','13','14','16']
		classes = {v: k for k, v in self.classes.iteritems()}
		self.targets = []; self.data = []; self.train_sets = []; self.test_sets = []; get_data = True
		for speak_test in itertools.product(males,females):#test_couples:
			i = 0; train = []; test = [];
			for audio in os.listdir(path):
				audio_path = os.path.join(path,audio)
				[Fs,x] = audioBasicIO.readAudioFile(audio_path)
				if get_data:
					self.data.append((x,Fs))
					self.targets.append(classes[audio[5]])
				if audio[:2] in speak_test:
					test.append(i)
				else:
					train.append(i)
				i = i + 1
			self.train_sets.append(train)
			self.test_sets.append(test)
			get_data = False

	def get_dafex_dataset(self,path,decode=False):
		males = ['4','5','7','8']
		females = ['1','2','3','6']
		no_audio = [3,6]
		classes = {v: k for k, v in self.classes.iteritems()}
		self.targets = []; self.data = []; self.train_sets = []; self.test_sets = []; get_data = True
		for speak_test in itertools.product(males,females):
			i = 0; train = []; test = []
			for actor_dir in os.listdir(path):
				if actor_dir[-1].isdigit():	#avoid invisible files
					actor_path = os.path.join(path,actor_dir)
					for block in os.listdir(actor_path):
						if block[-1].isdigit() and int(block[-1]) not in no_audio:	#avoid only video blocks and invisible files
							block_path = os.path.join(actor_path,block)
							for f_video in os.listdir(block_path):
								if f_video.endswith('avi'):	#avoid invisible files
									ss = f_video.split()
									f_video = f_video.replace(" ","\ ")		#for shell command
									video_path = os.path.join(block_path,f_video)
									audio_path = video_path.replace('.avi','.wav')	#output
									if decode and get_data: 
										sp.call("ffmpeg -i " + video_path + " -ab 160k -ac 1 -ar 22050 -vn " + audio_path, shell=True)
									[Fs,x] = audioBasicIO.readAudioFile(audio_path.replace("\ "," "))
									if get_data:
										self.targets.append(classes[ss[6]])	#getting targets
										self.data.append((x,Fs))	#getting signals + Fs
									if actor_dir[-1] in speak_test:
										test.append(i)
									else:
										train.append(i)
									i = i + 1
			self.train_sets.append(train)
			self.test_sets.append(test)
			get_data = False

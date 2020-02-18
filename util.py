with open("training.1600000.processed.noemoticon.csv", "r", encoding='ISO-8859-1') as all_samples:
	with open("training.csv", "w") as training:
		with open("test.csv", "w") as test:
			idx = 0
			for line in all_samples:
				idx += 1
				idx = idx % 200
				if idx in [1, 2]:
					training.write(line)
				elif idx == 3:
					test.write(line)

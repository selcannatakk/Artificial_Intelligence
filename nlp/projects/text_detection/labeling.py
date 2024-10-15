import pandas as pd


data = pd.read_csv("../raw_data/data.csv")

def labeling(dataset, labels):

  # tüm texler küçük harf yapılıyor
  for i in range(len(labels)-1):
      labels[i] = labels[i].lower()

  # print(labels.values().lower())
  labels = {word: index  for index, word in enumerate(labels.values())}

  for text in range(len(dataset)-1):
      dataset[text].lower()

      # liste yapılıyor
      text_dataset = dataset[text].split()

      # arayacağımız texti sözlük formatına getiriyoruz
      words_dataset = {word: index for index, word in enumerate(text_dataset)}

      # searched_word_labels -> sözlük -> 1 boyutu az
      # words_dict -> sözlük 2
      common_keys = []
      for key in labels:
          if key in words_dataset:
              common_keys.append(key)

      row = 0
      print(words_dataset)
      print(labels['say'])
      for word in words_dataset:
        row+=1
        for key in common_keys:
          if word == key:
            print(key)
            data.at[row, 'Label'] = word


  print(data.head())
  print("finish")


dataset = data['Message']
labels = {
  0: 'say',
  1: 'ok',
  2: 'salak'}

labeling(dataset, labels)

data.to_csv('../processed_data/labeling_data.csv', index=False)


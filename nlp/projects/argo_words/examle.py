# text = "Python ile cümle içerisinde kelime arama salak"
# searched_word = "salak"
#
# text = text.lower()
# searched_word = searched_word.lower()
#
# text_to_words = text.split()
# print(text_to_words)
#
#
# words_dict = {index: word for index, word in enumerate(text_to_words)}
#
# print(words_dict)
# print(words_dict[2])
# for index in words_dict:
#     if words_dict[index] == searched_word:
#         print(f"argo kelime:{searched_word}")
'''
import pandas as pd


text = "Python ile cümle içerisinde kelime arama salak"
# text = input("text giriniz")
data = "./data/spam.csv"

dataset = pd.read_csv(data)
dataset = dataset['Message']
print(dataset.head(10))


labels = {
    0: 'aptal',
    1: 'Mal',
    3: 'salak'}

# tüm texler küçük harf yapılıyor
# for i in range(len(labels)-1):
#     labels[i] = labels[i].lower()

labels = {word: index for index, word in enumerate(labels.values())}
for text in range(len(dataset)-1):
    select_column[text].lower()
    # liste yapılıyor
    text_dataset = dataset[text].split()
    # 'arayacağımız texti sözlük formatına getiriyoruz
    words_dataset = {word: index for index, word in enumerate(text_dataset)}
    # searched_word_labels -> sözlük -> 1 boyutu az
    #     words_dict -> sözlük 2
    print(words_dataset)
    print(labels)
    print("-------------------------------------------")
    common_keys = []
    for key in labels:
        if key in words_dataset:
            common_keys.append(key)
            print(common_keys)
            print(words_dataset[key])
            print(labels[key])

    # key ler text formatında
    for word in words_dataset:
        for key in common_keys:
            if word == key:
                pass

    for label in labels:
        print(label)
        # for key in common_keys:
        # dataset['Label'] = dataset['Category'].apply(lambda label: key if label == key else 0)

# with open('labels.txt', 'r') as file:
#     labels = file.read()
#     labels = labels.upper()
#
# labels = {word: index for index, word in enumerate(labels)}
# print(labels)
labels = {
    0: 'aptal',
    1: 'Mal',
    3: 'salak'}

tüm texler küçük harf yapılıyor
# for i in range(len(labels)-1):
#     labels[i] = labels[i].lower()
'''


labels = {word: index for index, word in enumerate(labels.values())}
print(labels)

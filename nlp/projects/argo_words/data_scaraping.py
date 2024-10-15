import pandas as pd
import snscrape.modules.twitter as sntwit
import tweepy


# max_tweet = 50
# tweets =[]
# query = 'from: Selcan_ATAK_'
#
# for t, tweet in enumerate(sntwit.TwitterSearchScraper(query, top=True).get_items()):
#     if t > max_tweet:
#         break
#     else:
#         tweets.append([tweet.date,tweet.id,tweet.content,tweet.user,tweet.username])
#
#
# '''
# Create Data Frame
# # '''
# # df = pd.DataFrame(tweets, columns = ["date","id","content","user","username"])
# # print(df.head(3))
# # df.to_csv('tweet_data.csv')


# Twitter API anahtarları ve erişim belgeleri
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Tweepy ile kimlik doğrulaması
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Çekilecek tweet'leri filtrelemek için arama sorgusu
search_query = 'Python'
tweets = api.search_tweets(q=search_query, count=10)

for tweet in tweets:
    print(tweet.text)



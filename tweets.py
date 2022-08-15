import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "Malaysia Industry 4.0 until:2022-06-01 since:2015-01-01"
tweets = []
limit = 5000

for tweet in sntwitter.TwitterSearchScraper(query).get_items():

    #print(vars(tweet))
    #break

    if len(tweets) == limit:
        break
    else:
        tweets.append([tweet.date, tweet.user.username, tweet.content])

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])

df.to_csv('Malaysia Tweets.csv')
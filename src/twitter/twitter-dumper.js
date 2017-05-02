import { writeFileSync } from 'fs'
import Twitter from 'twitter'

import normaliseTweet from './normalise-tweet'
import { twitterConfig } from '../constants'

const tweets = new Twitter(twitterConfig)

const makeQuery = screenName => ({
  screen_name: screenName,
  count: 200,
  trim_user: true,
  exclude_replies: false,
  include_rts: false
})

const retrieveTweetChunk = (baseQuery, olderThan = undefined) => (
  new Promise((resolve, reject) => {
    tweets.get(
      'statuses/user_timeline',
      Object.assign({}, baseQuery, { max_id: olderThan }),
      (err, tweets) => {
        if (err) {
          return reject(err)
        }

        resolve(tweets)
      }
    )
  })
)

const retrieveAllTweets = screenName => {
  console.log(`Retrieving all tweets for ${screenName}...`)

  const query = makeQuery(screenName)

  const recursion = (num = 0, olderThan) => retrieveTweetChunk(query, olderThan)
    .then((tweets = []) => {
      const totalTweets = num + tweets.length
      console.log(screenName, 'Retrieved', totalTweets, 'tweets...')

      const normalisedTweets = tweets
        .map(normaliseTweet)
        .filter(Boolean)
        // .map(text => `@${screenName}: ${text}`)

      if (tweets.length === 0) {
        return normalisedTweets
      }

      const { id: lastId } = tweets[tweets.length - 1]
      if (lastId === olderThan) {
        return tweets.slice(0, -1)
      }

      return recursion(totalTweets, lastId)
        .then(olderTweets => normalisedTweets.concat(olderTweets))
    })

  return recursion()
    .then(tweets => {
      writeFileSync(`./data/tweets-${screenName}.json`, JSON.stringify(tweets))
      return tweets
    })
}

const screenNames = [
  'ken_wheeler',
  'dan_abramov',
  'mxstbr',
  'kentcdodds',
  'rauchg',
  'thejameskyle'
]

Promise.all(
  screenNames.map(retrieveAllTweets)
).catch(err => {
  console.error('An unexpected error occured:', err)
})

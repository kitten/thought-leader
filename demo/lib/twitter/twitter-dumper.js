const { writeFileSync } = require('fs')
const Twitter = require('twitter')

const normaliseTweet = require('./normalise-tweet')
const { twitterConfig } = require('../constants')

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

const keywords = [
  'browser',
  'js',
  'javascript',
  'react',
  'angular',
  'ember',
  'dev',
  'html',
  'css',
  'web',
  'facebook',
  'uber',
  'google',
  'tech',
  'programm',
  'api',
  'app',
  'npm',
  'package',
  'library',
  'native',
  'ecosystem',
  'platform',
  'component',
  'build',
  'tool',
  'design',
  'code',
  'optimiz',
  'client',
  'compiler',
  'babel',
  'webpack'
]

const containsKeyword = text => keywords.some(k => text.includes(k))

const retrieveAllTweets = screenName => {
  console.log(`Retrieving all tweets for ${screenName}...`)

  const query = makeQuery(screenName)

  const recursion = (num = 0, olderThan) => retrieveTweetChunk(query, olderThan)
    .then((tweets = []) => {
      const totalTweets = num + tweets.length
      console.log(screenName, 'Retrieved', totalTweets, 'tweets...')

      const normalisedTweets = tweets
        .filter(({
          lang,
          entities: { media, urls }
        }) => (
          lang === 'en' &&
          (!media || !media.length) &&
          (!urls || !urls.length)
        ))
        .map(({ text, entities: { user_mentions, hashtags }}) => {
          let tweet = text.toLowerCase()

          tweet = user_mentions
            .reduce((acc, { screen_name }) => (
              acc.replace('@' + screen_name.toLowerCase(), '')
            ), tweet)

          tweet = hashtags
            .reduce((acc, { text }) => (
              acc.replace('#' + text.toLowerCase(), '')
            ), tweet)

          return tweet
        })
        .map(normaliseTweet)
        .filter(tweet => (
          tweet &&
          tweet.length > 15 &&
          containsKeyword(tweet)
        ))

      if (tweets.length === 0) {
        return normalisedTweets
      }

      const lastId = tweets[tweets.length - 1].id
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
  'thejameskyle',
  'benlesh',
  'addyosmani',
  'paul_irish',
  '_developit',
  'sebmarkbage',
  'ryanflorence',
  '_jayphelps',
  '_ericelliott',
  'toddmotto',
  'sebmck',
  'sarah_edo',
  'cpojer',
  'acdlite',
  'leeb',
  'soprano',
  'vjeux',
  'trueadm',
  'mweststrate',
  'linclark',
  'nikgraf',
  'mattpodwysocki',
  'ladyleet',
  'mjackson',
  '_chenglou',
  'jordwalke',
  'andrestaltz',
  'jevakallio'
]

Promise.all(
  screenNames.map(retrieveAllTweets)
).catch(err => {
  console.error('An unexpected error occured:', err)
})

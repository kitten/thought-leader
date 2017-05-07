const path = require('path')
const dotenv = require('dotenv')

dotenv.config()

const twitterConfig = {
  consumer_key: process.env.CONSUMER_KEY,
  consumer_secret: process.env.CONSUMER_SECRET,
  access_token_key: process.env.ACCESS_TOKEN_KEY,
  access_token_secret: process.env.ACCESS_TOKEN_SECRET
}

const MODEL_PATH = path.join(process.cwd(), './data/net-state.json')

const LETTER_SIZE = 15
const HIDDEN_SIZES = [ 64, 64 ]

module.exports = {
  twitterConfig,
  MODEL_PATH,
  LETTER_SIZE,
  HIDDEN_SIZES
}

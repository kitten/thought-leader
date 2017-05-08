const unescape = require('unescape')
const unicodeNormalisation = require('normalize-strings')

const leadingDotRegex = /^\./g
const multiWhitespaceRegex = /[\s\n]+/g
const quoteRegex = /[“”]/g

const nonValidCharsRegex = /[^\x00-\x7F]/g

const normaliseTweet = text => unescape(
  unicodeNormalisation(text)
    .toLowerCase()
    .replace(leadingDotRegex, '') // remove leading dot
    .replace(quoteRegex, '"') // normalise quotes
    .replace(nonValidCharsRegex, '') // remove unknown chars
    .replace(multiWhitespaceRegex, ' ') // normalise multiple whitespaces
    .trim()
)

module.exports = normaliseTweet

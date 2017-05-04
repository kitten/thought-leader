const unescape = require('unescape')
const unicodeNormalisation require('normalize-strings')

const mentionRegex = /[@]+[A-Za-z0-9-_]+/g
const urlRegex = /[A-Za-z]+:\/\/[A-Za-z0-9-_]+\.[A-Za-z0-9-_:%&~\?\/.=]+/g
const leadingDotRegex = /^\./g
const multiWhitespaceRegex = /[\s\n]+/g
const quoteRegex = /[“”]/g

//const nonValidCharsRegex = /[^\x00-\x7F]/g
const nonValidCharsRegex = /[^\s\w-]/g

const normaliseTweet = ({ text }) => unescape(
  unicodeNormalisation(text)
    .toLowerCase()
    .replace(mentionRegex, '') // remove @-mentions
    .replace(urlRegex, '') // remove urls
    .replace(leadingDotRegex, '') // remove leading dot
    .replace(quoteRegex, '"') // normalise quotes
    .replace(nonValidCharsRegex, '') // remove unknown chars
    .replace(multiWhitespaceRegex, ' ') // normalise multiple whitespaces
    .trim()
)

module.exports = normaliseTweet

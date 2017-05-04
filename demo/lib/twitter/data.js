const path = require('path')
const glob = require('glob').sync

const dataDir = path.join(process.cwd(), './data')

const data = glob(
  path.join(dataDir, 'tweets-*.json')
)
  .map(file => require(file))
  .reduce((acc, tweets) => (
    acc.concat(tweets)
  ), [])

module.exports = data

const { Network } = require('../..')
const { MODEL_PATH } = require('./constants')

const net = Network.fromJSON(require(MODEL_PATH))

console.log('0.6:', net.predict(0.6))
console.log('0.7:', net.predict(0.7))
console.log('0.8:', net.predict(0.8))
console.log('0.9:', net.predict(0.9))
console.log('1.0:', net.predict(1))

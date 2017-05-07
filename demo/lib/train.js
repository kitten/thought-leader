const { Network } = require('../..')
const { writeFileSync } = require('fs')
const trainingData = require('./twitter/data')

const {
  LETTER_SIZE,
  HIDDEN_SIZES,
  MODEL_PATH
} = require('./constants')

console.log('Creating network...')

const net = process.env.RESTART === 'true' ?
  Network.fromJSON(require(MODEL_PATH)) :
  new Network(
    trainingData,
    LETTER_SIZE,
    HIDDEN_SIZES,
    'gru'
  )

console.log('Training network...')

const DECAY = 0.97
const EPOCH_SIZE = net.data.input.length
const MAX_TRAIN = EPOCH_SIZE * 52

let EPOCH = Math.floor(net.iterations / EPOCH_SIZE)
let RATE = EPOCH >= 10 ?
  0.002 * Math.pow(DECAY, EPOCH - 10) :
  0.002

for (let i = net.iterations; i < MAX_TRAIN; i++) {
  EPOCH = Math.floor(i / EPOCH_SIZE)

  if (i % EPOCH_SIZE === 0 && EPOCH >= 10) {
    RATE = RATE * DECAY
  }

  const [ ppl, cost ] = net.train(RATE)

  if (i % 100 === 0) {
    const pred = net.predict()

    console.log(
      'ppl:', ppl,
      '; cost:', cost,
      '; lr:', RATE,
      '; ix:', i,
      '; e:', EPOCH,
      '; p:', pred
    )

    writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON()))
  }
}

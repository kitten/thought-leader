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
    'lstm'
  )

console.log('Training network...')

const DECAY = 0.0005
const EPOCH_SIZE = net.data.input.length
const MAX_TRAIN = EPOCH_SIZE * 200

let learningRate = 0.01

for (let i = net.iterations; i < MAX_TRAIN; i++) {
  const epoch = Math.floor(i / EPOCH_SIZE)

  if (i > 0 && i % EPOCH_SIZE === 0) {
    learningRate = learningRate / (1 + DECAY * epoch)
  }

  const ppl = net.train(learningRate)

  if (i % 100 === 0) {
    const pred = net.predict()

    console.log(
      'ppl:', (ppl === ppl && ppl !== Infinity) ? ppl.toFixed(2) : ppl,
      '; lr:', learningRate,
      '; ix:', i,
      '; e:', epoch,
      '; p:', pred
    )

    writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON()))
  }
}

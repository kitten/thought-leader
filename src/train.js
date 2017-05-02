// @flow

import Network from './network'
import { writeFileSync } from 'fs'
import trainingData from './twitter/data'

import {
  LETTER_SIZE,
  HIDDEN_SIZES,
  MODEL_PATH
} from './constants'

console.log('Creating network...')

const net = process.env.RESTART === 'true' ?
  Network.fromJSON(require(MODEL_PATH)) :
  new Network(
    trainingData,
    LETTER_SIZE,
    HIDDEN_SIZES
  )

console.log('Training network...')

const EPOCH_SIZE = net.data.input.length

for (let i = 0; i < 1000 * EPOCH_SIZE; i++) {
  const ppl = net.train()

  if (i % 100 === 0) {
    const pred = net.predict()
    console.log('ppl:', ppl, '; ix:', i, '; p:', pred)

    writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON()))
  }
}

writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON()))
console.log('Done.')

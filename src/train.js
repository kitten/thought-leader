// @flow

import Network from './network'
import { writeFileSync } from 'fs'
import { data, charset } from './utils/data'
import { EPOCH_SIZE, MODEL_PATH } from './constants'

console.log('Entries:', data.length, '; Charset:', charset.length)

const net = new Network()

console.log('Training network... (Output perplexity)')
for (let i = 0; i < 1000 * EPOCH_SIZE; i++) {
  const ppl = net.train()

  if (i % 100 === 0) {
    const pred = net.predict()
    console.log('ppl:', ppl, '; ix:', i, '; p:', pred)

    // writeFileSync(MODEL_PATH, JSON.stringify(net.toJSON()))
  }
}

// @flow

import Network from './network'
import { MODEL_PATH } from './constants'

const net = Network.fromJSON(require(MODEL_PATH))
const prediction = net.predict(0.6)

console.log(prediction)

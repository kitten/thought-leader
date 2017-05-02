// @flow

import path from 'path'
import { sync as glob } from 'glob'

const dataDir = path.join(process.cwd(), './data')

const data: string[] = glob(
  path.join(dataDir, 'tweets-*.json')
)
  // $FlowFixMe
  .map((file: string) => require(file))
  .reduce((acc: string[], tweets: string[]): string[] => (
    acc.concat(tweets)
  ), [])

export default data

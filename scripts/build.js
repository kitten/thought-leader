const fs = require('fs')
const path = require('path')
const webpack = require('webpack')

const { prepackFileSync } = require('../vendor/prepack')
const config = require('../webpack.config')

const outputPath = path.join(process.cwd(), 'dist')
const fileName = config.output.filename

config.output.path = outputPath

const compiler = webpack(config)

compiler.run((err, stats) => {
  if (err) {
    throw err
  }

  const outputFile = path.join(
    outputPath,
    fileName
  )

  const optimised = prepackFileSync(outputFile)
    .code
    .replace('ThoughtLeader =', 'module.exports = ThoughtLeader =')

  fs.writeFileSync(path.join(
    outputPath,
    fileName.replace('.js', '.opt.js')
  ), optimised)

  console.log('Done.')
})

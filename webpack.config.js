const path = require('path')
const webpack = require('webpack')
const ClosureCompilerPlugin = require('webpack-closure-compiler')

module.exports = {
  entry: './src/index.js',
  output: {
    path: path.join(__dirname, 'dist'),
    filename: 'thought-leader.js',
    library: 'ThoughtLeader',
    libraryTarget: 'umd'
  },
  module: {
    rules: [
      {
        test: /\.jsx?/,
        include: [
          path.resolve(__dirname, 'src')
        ],
        loader: 'babel-loader'
      }
    ]
  },
  /*
  plugins: [
    new ClosureCompilerPlugin({
      jsCompiler: true,
      compiler: {
        language_in: 'ECMASCRIPT6',
        language_out: 'ECMASCRIPT5',
        compilation_level: 'SIMPLE'
      }
    })
  ]
  */
}

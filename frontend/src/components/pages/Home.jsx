import React, { useState } from 'react'
import ImageUpload from '../upload/ImageUpload'
import AnalysisResults from '../results/AnalysisResults'

const Home = () => {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result)
    setIsAnalyzing(false)
  }

  const handleAnalysisStart = () => {
    setIsAnalyzing(true)
    setAnalysisResult(null)
  }

  const handleReset = () => {
    setAnalysisResult(null)
    setIsAnalyzing(false)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Precision Agriculture Disease Analysis
          </h1>
          <p className="text-lg text-gray-600">
            Advanced AI-powered plant disease detection and analysis
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {!analysisResult && !isAnalyzing && (
            <ImageUpload 
              onAnalysisComplete={handleAnalysisComplete}
              onAnalysisStart={handleAnalysisStart}
            />
          )}

          {isAnalyzing && (
            <div className="bg-white rounded-lg shadow-lg p-8 text-center">
              <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-green-500 mx-auto mb-4"></div>
              <h3 className="text-xl font-semibold text-gray-800 mb-2">Analyzing Image</h3>
              <p className="text-gray-600">Our AI is examining your plant image for disease detection...</p>
            </div>
          )}

          {analysisResult && (
            <div>
              <div className="mb-6 text-center">
                <button
                  onClick={handleReset}
                  className="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-lg transition-colors"
                >
                  Analyze Another Image
                </button>
              </div>
              <AnalysisResults result={analysisResult} />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Home

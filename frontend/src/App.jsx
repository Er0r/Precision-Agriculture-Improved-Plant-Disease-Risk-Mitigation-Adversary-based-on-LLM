import React, { useState } from 'react'
import { Leaf } from 'lucide-react'
import ImageUpload from './components/ImageUpload'
import AnalysisResults from './components/AnalysisResults'
import LoadingOverlay from './components/LoadingOverlay'
import Toast from './components/Toast'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [toast, setToast] = useState(null)

  const showToast = (message, type = 'info') => {
    setToast({ message, type })
    setTimeout(() => setToast(null), 5000)
  }

  const handleAnalysisComplete = (result) => {
    setAnalysisResult(result)
    showToast('🎉 Analysis completed successfully!', 'success')
  }

  const handleAnalysisError = (error) => {
    showToast(`❌ Analysis failed: ${error}`, 'error')
  }

  const clearResults = () => {
    setAnalysisResult(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-gradient-to-r from-primary-600 to-primary-700 text-white shadow-xl">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center">
            <div className="flex items-center justify-center mb-4">
              <Leaf className="w-12 h-12 mr-4" />
              <h1 className="text-4xl font-bold">Smart Crop Disease Analyzer</h1>
            </div>
            <p className="text-xl text-primary-100">
              AI-Powered Disease Detection & Treatment Recommendations
            </p>
            <p className="text-primary-200 mt-2">
              Upload your crop images for instant expert analysis
            </p>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section */}
          <div className="space-y-6">
            <ImageUpload
              onAnalysisComplete={handleAnalysisComplete}
              onAnalysisError={handleAnalysisError}
              onLoadingChange={setIsLoading}
              onClearResults={clearResults}
            />
          </div>

          {/* Results Section */}
          <div className="space-y-6">
            <AnalysisResults 
              result={analysisResult}
              onClear={clearResults}
            />
          </div>
        </div>
      </main>

      {/* Loading Overlay */}
      {isLoading && <LoadingOverlay />}

      {/* Toast Notifications */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  )
}

export default App
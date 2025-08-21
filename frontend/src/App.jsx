import React, { useState } from 'react'
import { Leaf, BarChart3 } from 'lucide-react'
import ImageUpload from './components/upload/ImageUpload'
import AnalysisResults from './components/results/AnalysisResults'
import LoadingOverlay from './components/ui/LoadingOverlay'
import Toast from './components/ui/Toast'
import ClarityComparison from './components/results/ClarityComparison'
import Home from './components/pages/Home'
import { Routes, Route, Link, useLocation } from 'react-router-dom'

function App() {
  const [analysisResult, setAnalysisResult] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [toast, setToast] = useState(null)
  const location = useLocation()

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

  const isActive = (path) => location.pathname === path

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Navigation Header */}
      <nav className="bg-white shadow-lg border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Leaf className="w-8 h-8 text-primary-600 mr-3" />
              <h1 className="text-xl font-bold text-gray-900">Precision Agriculture Analytics</h1>
            </div>
            <div className="flex space-x-8">
              <Link
                to="/"
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive('/') 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <Leaf className="w-4 h-4 mr-2" />
                Disease Analysis
              </Link>
              <Link
                to="/clarity-analytics"
                className={`flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive('/clarity-analytics') 
                    ? 'bg-primary-100 text-primary-700' 
                    : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
                }`}
              >
                <BarChart3 className="w-4 h-4 mr-2" />
                Clarity Analysis
              </Link>
            </div>
          </div>
        </div>
      </nav>


      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={
            <>
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  AI-Powered Crop Disease Analysis
                </h2>
                <p className="text-xl text-gray-600 mb-2">
                  Upload crop images for instant disease detection and treatment recommendations
                </p>
                <p className="text-gray-500">
                  Advanced analytics compare our results with established research standards
                </p>
              </div>
              <Home
                onAnalysisComplete={handleAnalysisComplete}
                onAnalysisError={handleAnalysisError}
                onLoadingChange={setIsLoading}
                onClearResults={clearResults}
                analysisResult={analysisResult}
              />
            </>
          } />

          <Route path="/clarity-analytics" element={
            <div>
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-gray-900 mb-4">
                  Dataset Clarity Analysis
                </h2>
                <p className="text-xl text-gray-600 mb-2">
                  Domain-aware clarity evaluation and dataset performance overview
                </p>
              </div>
              <div className="bg-white border border-gray-200 rounded-xl p-6">
                <ClarityComparison result={null} showCalculationDetails={true} />
              </div>
            </div>
          } />
        </Routes>
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
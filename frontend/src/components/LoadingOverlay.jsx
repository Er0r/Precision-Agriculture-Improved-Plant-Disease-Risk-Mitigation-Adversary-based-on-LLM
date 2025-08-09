import React from 'react'
import { Loader2, Microscope } from 'lucide-react'

const LoadingOverlay = () => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 max-w-md mx-4 text-center shadow-2xl">
        <div className="mb-6">
          <div className="relative">
            <Microscope className="w-16 h-16 text-primary-600 mx-auto mb-4" />
            <Loader2 className="w-8 h-8 text-primary-500 animate-spin absolute -top-2 -right-2" />
          </div>
        </div>
        
        <h3 className="text-2xl font-bold text-gray-800 mb-2">
          Analyzing Your Crop
        </h3>
        
        <p className="text-gray-600 mb-6">
          Our AI is examining the image for disease detection...
        </p>
        
        <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
          <div className="h-full bg-gradient-to-r from-primary-500 to-primary-600 rounded-full animate-pulse"></div>
        </div>
        
        <p className="text-sm text-gray-500 mt-4">
          This may take a few moments
        </p>
      </div>
    </div>
  )
}

export default LoadingOverlay
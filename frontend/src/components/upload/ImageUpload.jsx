import React, { useState, useRef } from 'react'
import { Upload, X, Microscope, Leaf } from 'lucide-react'
import { uploadImage, analyzeImage } from '../../services/api'

const ImageUpload = ({ onAnalysisComplete, onAnalysisError, onLoadingChange, onClearResults }) => {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [cropType, setCropType] = useState('rice')
  const [fileId, setFileId] = useState(null)
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const handleFileSelect = (file) => {
    if (!file || !file.type.startsWith('image/')) {
      onAnalysisError('Please select a valid image file')
      return
    }

    if (file.size > 16 * 1024 * 1024) {
      onAnalysisError('File size must be less than 16MB')
      return
    }

    setSelectedFile(file)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    onClearResults()

    // Auto-upload the file
    handleUpload(file)
  }

  const handleUpload = async (file) => {
    try {
      const result = await uploadImage(file, cropType)
      if (result.success) {
        setFileId(result.file_id)
      } else {
        onAnalysisError(result.error || 'Upload failed')
      }
    } catch (error) {
      onAnalysisError('Upload failed: ' + error.message)
    }
  }

  const handleAnalyze = async () => {
    if (!fileId) {
      onAnalysisError('Please upload an image first')
      return
    }

    onLoadingChange(true)
    try {
      const result = await analyzeImage(fileId, cropType)
      if (result.success) {
        onAnalysisComplete(result.analysis)
      } else {
        onAnalysisError(result.error || 'Analysis failed')
      }
    } catch (error) {
      onAnalysisError('Analysis failed: ' + error.message)
    } finally {
      onLoadingChange(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreviewUrl(null)
    setFileId(null)
    onClearResults()
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragOver(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragOver(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  return (
    <div className="space-y-6">
      {/* Crop Type Selection */}
      <div className="card p-6">
        <label className="block text-sm font-semibold text-gray-700 mb-3">
          <Leaf className="w-5 h-5 inline mr-2 text-primary-600" />
          Select Crop Type
        </label>
        <select
          value={cropType}
          onChange={(e) => setCropType(e.target.value)}
          className="w-full p-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-primary-500 focus:border-transparent text-lg"
        >
          <option value="rice"> Rice</option>
          <option value="jute"> Jute</option>
        </select>
      </div>

      {/* Upload Area */}
      <div className="card p-6">
        <div
          className={`upload-area ${isDragOver ? 'dragover' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <Upload className="w-16 h-16 text-primary-500 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-800 mb-2">
            Drop your image here
          </h3>
          <p className="text-gray-600 mb-2">or click to browse files</p>
          <p className="text-sm text-gray-500">
            Supports: JPG, PNG, GIF, BMP (Max 16MB)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={(e) => handleFileSelect(e.target.files[0])}
            className="hidden"
          />
        </div>
      </div>

      {/* Image Preview */}
      {previewUrl && (
        <div className="card p-6 animate-slide-up">
          <div className="text-center">
            <img
              src={previewUrl}
              alt="Crop preview"
              className="max-w-full max-h-80 mx-auto rounded-xl shadow-lg"
            />
            <div className="mt-6 flex gap-4 justify-center">
              <button
                onClick={handleAnalyze}
                disabled={!fileId}
                className="btn-primary flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Microscope className="w-5 h-5" />
                Analyze Disease
              </button>
              <button
                onClick={handleClear}
                className="btn-secondary flex items-center gap-2"
              >
                <X className="w-5 h-5" />
                Clear
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ImageUpload

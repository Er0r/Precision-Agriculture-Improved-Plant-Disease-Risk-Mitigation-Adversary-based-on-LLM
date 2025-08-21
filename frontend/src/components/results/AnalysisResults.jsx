import React, { useEffect, useState } from 'react'
import { 
  CheckCircle, 
  AlertTriangle, 
  Pill, 
  Shield, 
  DollarSign, 
  Clock, 
  Eye,
  TrendingUp,
  Leaf
} from 'lucide-react'
import ClarityComparison from './ClarityComparison'

const AnalysisResults = ({ result, onClear }) => {
  const [confidenceWidth, setConfidenceWidth] = useState(0)

  useEffect(() => {
    if (result?.confidence) {
      // Animate confidence bar
      setTimeout(() => {
        setConfidenceWidth(result.confidence * 100)
      }, 100)
    }
  }, [result])

  if (!result) {
    return (
      <div className="card p-8">
        <div className="text-center text-gray-500">
          <Eye className="w-16 h-16 mx-auto mb-4 text-gray-300" />
          <h3 className="text-xl font-semibold mb-2">Ready for Analysis</h3>
          <p>Upload a crop image to get started with AI-powered disease detection</p>
        </div>
      </div>
    )
  }

  const isHealthy = !result.disease_detected
  const StatusIcon = isHealthy ? CheckCircle : AlertTriangle
  const statusColor = isHealthy ? 'text-green-600' : 'text-yellow-600'
  const statusBg = isHealthy ? 'bg-green-50' : 'bg-yellow-50'

  const getDangerLevelColor = (level) => {
    const levelLower = level?.toLowerCase() || ''
    if (levelLower.includes('low')) return 'bg-green-100 text-green-800'
    if (levelLower.includes('moderate')) return 'bg-yellow-100 text-yellow-800'
    if (levelLower.includes('high')) return 'bg-orange-100 text-orange-800'
    if (levelLower.includes('critical')) return 'bg-red-100 text-red-800'
    return 'bg-gray-100 text-gray-800'
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Analysis Header */}
      <div className="card overflow-hidden">
        <div className="bg-gradient-to-r from-primary-600 to-primary-700 text-white p-6">
          <h2 className="text-2xl font-bold flex items-center gap-3">
            <TrendingUp className="w-8 h-8" />
            Analysis Results
          </h2>
        </div>

        {/* Disease Detection Summary */}
        <div className="p-6">
          <div className={`${statusBg} rounded-2xl p-6 mb-6`}>
            <div className="flex items-center gap-4 mb-4">
              <StatusIcon className={`w-12 h-12 ${statusColor}`} />
              <div>
                <h3 className="text-2xl font-bold text-gray-800">
                  {isHealthy ? 'Healthy Crop Detected' : `Disease Detected: ${result.disease_name}`}
                </h3>
                <p className="text-gray-600">
                  {isHealthy ? 'No diseases detected in your crop' : `Severity: ${result.severity || 'Unknown'}`}
                </p>
              </div>
            </div>

            {result.bacterial_infection && (
              <div className="bg-orange-100 border border-orange-200 rounded-xl p-4 mt-4">
                <div className="flex items-center gap-2 text-orange-800">
                  <AlertTriangle className="w-5 h-5" />
                  <strong>Bacterial Infection Detected</strong> - Additional treatment may be required
                </div>
              </div>
            )}
          </div>

          {/* Confidence Display */}
          <div className="bg-white border border-gray-200 rounded-xl p-6 mb-6">
            <div className="flex justify-between items-center mb-3">
              <span className="font-semibold text-gray-700">Detection Confidence</span>
              <span className="bg-primary-100 text-primary-800 px-3 py-1 rounded-full text-sm font-semibold">
                {Math.round((result.confidence || 0) * 100)}%
              </span>
            </div>
            <div className="confidence-bar">
              <div 
                className="confidence-fill"
                style={{ width: `${confidenceWidth}%` }}
              />
            </div>
            <p className="text-sm text-gray-500 mt-2">
              {(result.confidence || 0) >= 0.8 ? 'High confidence detection' : 
               (result.confidence || 0) >= 0.6 ? 'Moderate confidence detection' : 
               'Low confidence - consider retaking image'}
            </p>
          </div>
        </div>
      </div>

      {!isHealthy && (
        <>
          {/* LLM Service Status */}
          {result.llm_error && (
            <div className="card p-6 bg-yellow-50 border border-yellow-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-yellow-600" />
                Treatment Service Notice
              </h3>
              <div className="bg-yellow-100 rounded-xl p-4">
                <p className="text-yellow-800 mb-2">
                  <strong>LLM Treatment Service:</strong> Currently unavailable
                </p>
                <p className="text-yellow-700 text-sm">
                  {result.llm_error === 'API key not configured' 
                    ? 'API key not configured. Contact system administrator for treatment recommendations.'
                    : `Service error: ${result.llm_error}. Please consult an agricultural expert for treatment advice.`}
                </p>
              </div>
            </div>
          )}

          {/* Risk Assessment */}
          {result.danger_level && (
            <div className="card p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <AlertTriangle className="w-6 h-6 text-orange-500" />
                Risk Assessment
              </h3>
              <div className={`inline-block px-4 py-2 rounded-full text-sm font-semibold ${getDangerLevelColor(result.danger_level)}`}>
                {result.danger_level}
              </div>
            </div>
          )}

          {/* Treatment Recommendations */}
          {result.recommendations && result.recommendations.length > 0 ? (
            <div className="card p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Pill className="w-6 h-6 text-blue-500" />
                Treatment Recommendations
              </h3>
              <div className="space-y-3">
                {result.recommendations.map((rec, index) => (
                  <div key={index} className="bg-gray-50 rounded-xl p-4 border-l-4 border-primary-500 hover:bg-gray-100 transition-colors">
                    <div className="flex items-start gap-3">
                      <Leaf className="w-5 h-5 text-primary-600 mt-0.5 flex-shrink-0" />
                      <p className="text-gray-800">{rec}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card p-6 bg-blue-50 border border-blue-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Pill className="w-6 h-6 text-blue-500" />
                Treatment Recommendations
              </h3>
              <div className="bg-blue-100 rounded-xl p-4">
                <p className="text-blue-800">
                  Treatment recommendations are not available at this time. 
                  Please consult with a local agricultural extension officer or plant pathologist 
                  for specific treatment advice for <strong>{result.disease_name}</strong>.
                </p>
              </div>
            </div>
          )}

          {/* Prevention Strategies */}
          {result.prevention_strategies && result.prevention_strategies.length > 0 ? (
            <div className="card p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-6 h-6 text-green-500" />
                Prevention Strategies
              </h3>
              <div className="space-y-3">
                {result.prevention_strategies.map((strategy, index) => (
                  <div key={index} className="bg-green-50 rounded-xl p-4 border-l-4 border-green-500 hover:bg-green-100 transition-colors">
                    <div className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                      <p className="text-gray-800">{strategy}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="card p-6 bg-green-50 border border-green-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Shield className="w-6 h-6 text-green-500" />
                Prevention Strategies
              </h3>
              <div className="bg-green-100 rounded-xl p-4">
                <p className="text-green-800">
                  Prevention strategies are not available at this time. 
                  Please consult agricultural resources for disease prevention methods 
                  specific to <strong>{result.disease_name}</strong> in your region.
                </p>
              </div>
            </div>
          )}

          {/* Economic Impact */}
          {result.economic_impact && 
           result.economic_impact !== 'Economic analysis provided by NIM LLM' && 
           result.economic_impact !== 'LLM service returned invalid response' && 
           result.economic_impact !== 'LLM service encountered an error' &&
           result.economic_impact !== 'Error occurred in LLM service' ? (
            <div className="card p-6 bg-gradient-to-br from-yellow-50 to-orange-50 border border-yellow-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <DollarSign className="w-6 h-6 text-yellow-600" />
                Economic Impact
              </h3>
              <p className="text-gray-800">{result.economic_impact}</p>
            </div>
          ) : result.economic_impact && result.economic_impact.includes('LLM') ? (
            <div className="card p-6 bg-yellow-50 border border-yellow-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <DollarSign className="w-6 h-6 text-yellow-600" />
                Economic Impact
              </h3>
              <p className="text-yellow-800">Economic impact analysis unavailable. Consult agricultural economist for assessment.</p>
            </div>
          ) : null}

          {/* Treatment Timeline */}
          {result.treatment_timeline && 
           result.treatment_timeline !== 'Treatment timeline provided by NIM LLM' &&
           result.treatment_timeline !== 'LLM service returned invalid response' &&
           result.treatment_timeline !== 'LLM service encountered an error' &&
           result.treatment_timeline !== 'Error occurred in LLM service' ? (
            <div className="card p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Clock className="w-6 h-6 text-purple-500" />
                Treatment Timeline
              </h3>
              <div className="bg-purple-50 rounded-xl p-4">
                <p className="text-gray-800 whitespace-pre-line">{result.treatment_timeline}</p>
              </div>
            </div>
          ) : result.treatment_timeline && result.treatment_timeline.includes('LLM') ? (
            <div className="card p-6 bg-purple-50 border border-purple-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Clock className="w-6 h-6 text-purple-500" />
                Treatment Timeline
              </h3>
              <div className="bg-purple-100 rounded-xl p-4">
                <p className="text-purple-800">Treatment timeline unavailable. Consult agricultural expert for timing guidance.</p>
              </div>
            </div>
          ) : null}

          {/* Monitoring Advice */}
          {result.monitoring_advice && 
           result.monitoring_advice !== 'Monitoring guidance provided by NIM LLM' &&
           result.monitoring_advice !== 'LLM service returned invalid response' &&
           result.monitoring_advice !== 'LLM service encountered an error' &&
           result.monitoring_advice !== 'Error occurred in LLM service' ? (
            <div className="card p-6">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-6 h-6 text-indigo-500" />
                Monitoring Guidelines
              </h3>
              <div className="bg-indigo-50 rounded-xl p-4">
                <p className="text-gray-800">{result.monitoring_advice}</p>
              </div>
            </div>
          ) : result.monitoring_advice && result.monitoring_advice.includes('LLM') ? (
            <div className="card p-6 bg-indigo-50 border border-indigo-200">
              <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                <Eye className="w-6 h-6 text-indigo-500" />
                Monitoring Guidelines
              </h3>
              <div className="bg-indigo-100 rounded-xl p-4">
                <p className="text-indigo-800">Monitoring guidelines unavailable. Consult agricultural expert for monitoring advice.</p>
              </div>
            </div>
          ) : null}
        </>
      )}

      {isHealthy && (
        <div className="card p-6">
          <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Leaf className="w-6 h-6 text-green-500" />
            Maintenance Recommendations
          </h3>
          <div className="space-y-3">
            {[
              'Continue current care practices',
              'Monitor regularly for early disease detection',
              'Maintain proper irrigation and nutrition',
              'Keep field clean and well-ventilated'
            ].map((rec, index) => (
              <div key={index} className="bg-green-50 rounded-xl p-4 border-l-4 border-green-500">
                <div className="flex items-start gap-3">
                  <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
                  <p className="text-gray-800">{rec}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Clarity Comparison Analytics */}
      <ClarityComparison result={result} />
    </div>
  )
}

export default AnalysisResults

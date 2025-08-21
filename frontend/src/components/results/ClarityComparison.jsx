import React, { useState, useEffect } from 'react'
import { TrendingUp, FileText, Database, AlertCircle, CheckCircle, BarChart3, Target, Zap, Calculator, Info, Trophy, Star } from 'lucide-react'

const ClarityComparison = ({ result, showCalculationDetails = false }) => {
  const [comparisonData, setComparisonData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showMethodology, setShowMethodology] = useState(false)

  useEffect(() => {
    analyzeClarity()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  const analyzeClarity = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/clarity-analytics/')
      if (response.ok) {
        const data = await response.json()
        const comparison = processApiDataForComparison(data, result)
        setComparisonData(comparison)
      } else {
        throw new Error('Failed to fetch analytics data')
      }
    } catch (error) {
      console.error('Error analyzing clarity:', error)
      setComparisonData({
        error: 'Unable to connect to analytics API. Please ensure the backend server is running.',
        fallback: true
      })
    } finally {
      setLoading(false)
    }
  }

  const processApiDataForComparison = (apiData) => {
    if (!apiData || !apiData.anchor_paper || !apiData.our_analyses || apiData.our_analyses.length === 0) {
      return { error: 'Insufficient data for comparison' }
    }

    const anchorMetrics = apiData.anchor_paper.metrics || {}
    const ourAnalyses = apiData.our_analyses || []

    const comparisonChartData = [
      {
        metric: 'Domain-Aware Clarity',
        anchor: anchorMetrics.clarity_score || 0,
        ourAvg: apiData.comparison_summary?.avg_clarity_score || 0,
        difference: apiData.comparison_summary?.clarity_vs_anchor || 0,
        calculation: 'Base clarity + domain expertise bonus + length appropriateness'
      },
      {
        metric: 'Domain Term Coverage',
        anchor: anchorMetrics.domain_coverage_percent || 0,
        ourAvg: apiData.comparison_summary?.avg_domain_coverage || 0,
        difference: 0,
        calculation: 'Higher % = More precise agricultural/medical terminology',
        isPercent: true
      }
    ]

    return {
      comparisonChartData,
      anchorMetrics,
      ourAnalyses,
      summaryStats: apiData.comparison_summary || {},
      anchorText: apiData.anchor_paper?.text || ''
    }
  }

  return (
    <div className="space-y-6">
      {showCalculationDetails && (
        <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-semibold text-blue-900 flex items-center gap-2">
              <Calculator className="w-6 h-6" />
              How Scores Are Calculated
            </h3>
            <button
              onClick={() => setShowMethodology(!showMethodology)}
              className="flex items-center gap-2 px-3 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
            >
              <Info className="w-4 h-4" />
              {showMethodology ? 'Hide' : 'Show'} Methodology
            </button>
          </div>

          {showMethodology && (
            <div className="grid grid-cols-1 gap-6">
              <div className="bg-white rounded-lg p-4 border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-3">Domain-Aware Clarity Calculation</h4>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-700">Base Flesch-Kincaid Score</span>
                    <span className="font-medium text-blue-600">60%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Domain Expertise Bonus</span>
                    <span className="font-medium text-blue-600">25%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-700">Length Appropriateness</span>
                    <span className="font-medium text-blue-600">15%</span>
                  </div>
                </div>
                <div className="mt-3 p-3 bg-blue-50 rounded text-sm">
                  <strong>Domain-Aware Formula:</strong> Uses Flesch-Kincaid with agricultural terms excluded from complexity calculations
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      <div className="card p-6">
        <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-blue-500" />
          Dataset Clarity Analysis
        </h3>
        <p className="text-gray-600 mb-4">
          Domain-aware clarity evaluation using agricultural glossary approach. Compares our disease analysis dataset against established standards.
        </p>

        {loading ? (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
            <span className="ml-3">Analyzing dataset clarity metrics...</span>
          </div>
        ) : comparisonData?.error ? (
          <div className="text-center py-8 text-red-500">
            <AlertCircle className="w-12 h-12 mx-auto mb-3 text-red-300" />
            <p className="font-semibold">{comparisonData.error}</p>
            {comparisonData.fallback && (
              <p className="text-sm text-gray-500 mt-2">
                Please ensure the backend server is running at http://localhost:8000
              </p>
            )}
          </div>
        ) : comparisonData ? (
          <>
            {/* Domain Glossary Information */}
            {comparisonData.summaryStats?.domain_glossary_stats && (
              <div className="mb-6 bg-gradient-to-r from-purple-50 to-indigo-50 rounded-lg p-4 border border-purple-200">
                <h4 className="text-lg font-semibold mb-3 flex items-center gap-2 text-purple-800">
                  <FileText className="w-5 h-5" />
                  Domain Glossary Approach
                </h4>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-purple-600">
                      {comparisonData.summaryStats.domain_glossary_stats.total_terms || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-600">Total Domain Terms</p>
                    <p className="text-xs text-gray-500">in glossary</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-indigo-600">
                      {(comparisonData.summaryStats?.avg_domain_coverage || 0).toFixed(1)}%
                    </p>
                    <p className="text-sm text-gray-600">Avg Domain Coverage</p>
                    <p className="text-xs text-gray-500">in our analyses</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-green-600">
                      {comparisonData.summaryStats.domain_glossary_stats.agricultural_terms || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-600">Agricultural Terms</p>
                    <p className="text-xs text-gray-500">excluded from penalty</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-blue-600">
                      {comparisonData.summaryStats.domain_glossary_stats.disease_terms || 'N/A'}
                    </p>
                    <p className="text-sm text-gray-600">Disease Terms</p>
                    <p className="text-xs text-gray-500">recognized as standard</p>
                  </div>
                </div>
                <div className="bg-white rounded-lg p-3 border border-purple-100">
                  <h5 className="font-semibold text-purple-700 mb-2">How Domain Glossary Improves Evaluation:</h5>
                  <ul className="text-sm text-gray-700 space-y-1">
                    <li><strong>Excludes agricultural terms</strong> from "complex word" counts (e.g., fungicide, pathogen, irrigation)</li>
                    <li><strong>Applies clarity metrics</strong> to remaining text only</li>
                    <li><strong>Rewards appropriate technical vocabulary</strong> instead of penalizing domain expertise</li>
                    <li><strong>Provides fair comparison</strong> between technical and general content</li>
                  </ul>
                </div>
              </div>
            )}

            {/* Dataset Overview */}
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-6 border border-blue-200 mb-6">
              <h4 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Target className="w-5 h-5 text-purple-600" />
                Dataset Overview
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">{comparisonData.summaryStats?.avg_clarity_score || 'N/A'}</p>
                  <p className="text-sm text-gray-600">Avg Clarity Score</p>
                  <p className="text-xs text-gray-500">vs {comparisonData.anchorMetrics?.clarity_score} (Anchor)</p>
                </div>
                <div className="text-center">
                  <p className={`text-2xl font-bold ${(comparisonData.summaryStats?.clarity_vs_anchor || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {(comparisonData.summaryStats?.clarity_vs_anchor || 0) > 0 ? '+' : ''}{comparisonData.summaryStats?.clarity_vs_anchor || 'N/A'}
                  </p>
                  <p className="text-sm text-gray-600">Clarity Difference</p>
                  <p className="text-xs text-gray-500">vs Anchor paper</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-purple-600">{(comparisonData.summaryStats?.avg_domain_coverage || 0).toFixed(1)}%</p>
                  <p className="text-sm text-gray-600">Domain Coverage</p>
                  <p className="text-xs text-gray-500">agricultural terms</p>
                </div>
                <div className="text-center">
                  <p className="text-2xl font-bold text-indigo-600">{comparisonData.summaryStats?.total_analyses || 0}</p>
                  <p className="text-sm text-gray-600">Total Analyses</p>
                  <p className="text-xs text-gray-500">in dataset</p>
                </div>
              </div>
            </div>

            {/* Comparison Chart */}
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-3 flex items-center gap-2 text-blue-700">
                <BarChart3 className="w-5 h-5 text-blue-600" />
                Clarity Comparison: Our Dataset vs Anchor Paper
              </h4>

              <div className="space-y-4">
                {comparisonData.comparisonChartData?.map((item, index) => (
                  <div key={index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium text-gray-700">{item.metric}</span>
                      <div className="flex gap-4 text-sm">
                        <span className="text-green-600">Anchor: {item.isPercent ? `${item.anchor.toFixed(1)}%` : item.anchor}</span>
                        <span className="text-blue-600">Our Avg: {item.isPercent ? `${item.ourAvg.toFixed(1)}%` : item.ourAvg}</span>
                      </div>
                    </div>
                    {item.calculation && (
                      <div className="text-xs text-gray-500 mb-2">
                        <strong>Calculation:</strong> {item.calculation}
                      </div>
                    )}

                    <div className="relative">
                      <div className="flex gap-2 h-8">
                        <div className="flex-1 relative">
                          <div className="h-full bg-gray-200 rounded"></div>
                          <div
                            className="absolute top-0 left-0 h-full bg-green-500 rounded transition-all duration-1000"
                            style={{ width: `${Math.min(100, (item.anchor / Math.max(item.anchor, item.ourAvg)) * 100)}%` }}
                          />
                          <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                            {item.isPercent ? `${item.anchor.toFixed(1)}%` : item.anchor}
                          </span>
                        </div>

                        <div className="flex-1 relative">
                          <div className="h-full bg-gray-200 rounded"></div>
                          <div
                            className="absolute top-0 left-0 h-full bg-blue-500 rounded transition-all duration-1000"
                            style={{ width: `${Math.min(100, (item.ourAvg / Math.max(item.anchor, item.ourAvg)) * 100)}%` }}
                          />
                          <span className="absolute inset-0 flex items-center justify-center text-xs font-semibold text-white">
                            {item.isPercent ? `${item.ourAvg.toFixed(1)}%` : item.ourAvg}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Performance Summary */}
            <div className="mb-6">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-xl font-bold mb-0 flex items-center gap-3">
                  <div className="p-2 bg-gradient-to-r from-purple-500 to-blue-500 rounded-lg">
                    <Zap className="w-6 h-6 text-white" />
                  </div>
                  <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">Performance Summary</span>
                </h4>

                <div className="flex items-center gap-2 bg-gradient-to-r from-green-50 to-emerald-50 px-4 py-2 rounded-full border border-green-200">
                  <Trophy className="w-4 h-4 text-green-600" />
                  <span className="text-sm font-semibold text-green-700">
                    {comparisonData?.ourAnalyses?.filter(a => (a.metrics?.clarity_score || 0) >= (comparisonData.anchorMetrics?.clarity_score || 0)).length || 0}/
                    {comparisonData?.ourAnalyses?.length || 0} Outperforming
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-white rounded-xl p-4 border border-blue-100 shadow-md">
                  <h6 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-green-600" />
                    Success Rate
                  </h6>
                  <p className="text-2xl font-bold text-green-600">
                    {((comparisonData?.ourAnalyses?.filter(a => (a.metrics?.clarity_score || 0) >= (comparisonData.anchorMetrics?.clarity_score || 0)).length || 0) / (comparisonData?.ourAnalyses?.length || 1) * 100).toFixed(0)}%
                  </p>
                  <p className="text-sm text-gray-600">analyses outperforming anchor</p>
                </div>

                <div className="bg-white rounded-xl p-4 border border-blue-100 shadow-md">
                  <h6 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-blue-600" />
                    Average Difference
                  </h6>
                  <p className={`text-2xl font-bold ${(comparisonData?.summaryStats?.avg_clarity_score || 0) >= (comparisonData?.anchorMetrics?.clarity_score || 0) ? 'text-green-600' : 'text-orange-600'}`}>
                    {((comparisonData?.summaryStats?.avg_clarity_score || 0) - (comparisonData?.anchorMetrics?.clarity_score || 0)) > 0 ? '+' : ''}
                    {((comparisonData?.summaryStats?.avg_clarity_score || 0) - (comparisonData?.anchorMetrics?.clarity_score || 0)).toFixed(1)}
                  </p>
                  <p className="text-sm text-gray-600">points vs anchor</p>
                </div>

                <div className="bg-white rounded-xl p-4 border border-blue-100 shadow-md">
                  <h6 className="font-semibold text-gray-800 mb-2 flex items-center gap-2">
                    <Star className="w-4 h-4 text-purple-600" />
                    Top Performer
                  </h6>
                  <p className="text-2xl font-bold text-purple-600">
                    {Math.max(...(comparisonData?.ourAnalyses?.map(a => a.metrics?.clarity_score || 0) || [0])).toFixed(1)}
                  </p>
                  <p className="text-sm text-gray-600">highest clarity score</p>
                </div>
              </div>
            </div>

            {/* Analysis Summary Table */}
            <div className="mb-6">
              <h4 className="text-lg font-semibold mb-3">Individual Analysis Details</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm border border-gray-200 rounded-lg">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="p-3 text-left border-b">Disease</th>
                      <th className="p-3 text-center border-b">Clarity Score</th>
                      <th className="p-3 text-center border-b">Word Count</th>
                      <th className="p-3 text-center border-b">vs Anchor</th>
                    </tr>
                  </thead>
                  <tbody>
                    {comparisonData.ourAnalyses?.slice(0, 5).map((analysis, index) => (
                      <tr key={index} className="hover:bg-gray-50">
                        <td className="p-3 border-b font-medium">{analysis.disease_name}</td>
                        <td className="p-3 border-b text-center">
                          <span className={`px-2 py-1 rounded text-xs font-semibold ${(analysis.metrics?.clarity_score || 0) >= (comparisonData.anchorMetrics?.clarity_score || 0)
                              ? 'bg-green-100 text-green-800'
                              : 'bg-yellow-100 text-yellow-800'
                            }`}>
                            {analysis.metrics?.clarity_score || 'N/A'}
                          </span>
                        </td>
                        <td className="p-3 border-b text-center">{analysis.metrics?.word_count || 'N/A'}</td>
                        <td className="p-3 border-b text-center">
                          {(analysis.metrics?.clarity_score || 0) >= (comparisonData.anchorMetrics?.clarity_score || 0) ? (
                            <CheckCircle className="w-4 h-4 text-green-600 mx-auto" />
                          ) : (
                            <AlertCircle className="w-4 h-4 text-yellow-600 mx-auto" />
                          )}
                        </td>
                      </tr>
                    ))}
                    <tr className="bg-blue-50 font-semibold">
                      <td className="p-3 border-b">Anchor Paper</td>
                      <td className="p-3 border-b text-center">{comparisonData.anchorMetrics?.clarity_score || 'N/A'}</td>
                      <td className="p-3 border-b text-center">{comparisonData.anchorMetrics?.word_count || 'N/A'}</td>
                      <td className="p-3 border-b text-center">
                        <span className="text-blue-600">Reference</span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            {/* Sample Text Comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-3 flex items-center gap-2">
                  <FileText className="w-4 h-4" />
                  Sample Analysis from Our Dataset
                </h4>
                <p className="text-sm text-blue-700 leading-relaxed max-h-40 overflow-y-auto">
                  {comparisonData.ourAnalyses?.[0]?.text || 'No analysis text available'}
                </p>
              </div>

              <div className="bg-green-50 rounded-lg p-4 border border-green-200">
                <h4 className="font-semibold text-green-800 mb-3 flex items-center gap-2">
                  <Database className="w-4 h-4" />
                  Anchor Paper Reference Text
                </h4>
                <p className="text-sm text-green-700 leading-relaxed max-h-40 overflow-y-auto">
                  {comparisonData.anchorText}
                </p>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <AlertCircle className="w-12 h-12 mx-auto mb-3 text-gray-300" />
            <p>No analysis available. Please analyze an image first.</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ClarityComparison

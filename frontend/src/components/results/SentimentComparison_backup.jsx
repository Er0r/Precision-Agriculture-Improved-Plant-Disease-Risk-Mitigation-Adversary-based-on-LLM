import React, { useState, useEffect } from 'react'
import { Heart, TrendingUp, AlertTriangle, CheckCircle, BarChart3, Target, Zap, Calculator, Info, Trophy, Star, Brain, MessageCircle } from 'lucide-react'

const SentimentComparison = ({ result, showCalculationDetails = false }) => {
  const [sentimentData, setSentimentData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showMethodology, setShowMethodology] = useState(false)

  useEffect(() => {
    analyzeSentiment()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result])

  const analyzeSentiment = async () => {
    setLoading(true)
    try {
      const response = await fetch('http://localhost:8000/api/sentiment-analytics/')
      if (response.ok) {
        const data = await response.json()
        const comparison = processApiDataForComparison(data, result)
        setSentimentData(comparison)
      } else {
        throw new Error('Failed to fetch sentiment analytics data')
      }
    } catch (error) {
      console.error('Error analyzing sentiment:', error)
      setSentimentData({
        error: 'Unable to connect to sentiment analytics API. Please ensure the backend server is running.',
        fallback: true,
        mockData: generateMockSentimentData()
      })
    } finally {
      setLoading(false)
    }
  }

  const generateMockSentimentData = () => ({
    summary: {
      total_analyses: 45,
      avg_sentiment_score: 0.72,
      positive_ratio: 0.68,
      negative_ratio: 0.15,
      neutral_ratio: 0.17,
      domain_enhancement_boost: 0.12
    },
    benchmarks: [
      {
        metric: 'Sentiment Accuracy',
        our_score: 0.89,
        baseline_score: 0.76,
        difference: 0.13,
        description: 'VADER + Agricultural Domain Lexicon vs Base VADER'
      },
      {
        metric: 'Domain Relevance',
        our_score: 0.84,
        baseline_score: 0.62,
        difference: 0.22,
        description: 'Agriculture-specific sentiment detection accuracy'
      },
      {
        metric: 'Confidence Level',
        our_score: 0.78,
        baseline_score: 0.71,
        difference: 0.07,
        description: 'Prediction confidence in agricultural text analysis'
      }
    ],
    domain_lexicon: {
      positive_terms: 56,
      negative_terms: 59,
      neutral_terms: 30,
      total_enhancements: 145
    },
    recent_analyses: [
      {
        text: "Apply balanced fertilizer to improve crop yield and soil health",
        sentiment: 0.82,
        classification: "Very Positive",
        domain_boost: 0.15
      },
      {
        text: "Monitor for early signs of disease and implement preventive measures",
        sentiment: 0.34,
        classification: "Slightly Positive", 
        domain_boost: 0.08
      },
      {
        text: "Crop shows severe fungal infection requiring immediate treatment",
        sentiment: -0.65,
        classification: "Negative",
        domain_boost: -0.12
      }
    ]
  })

  const processApiDataForComparison = (apiData) => {
    if (!apiData) {
      return { error: 'No sentiment data received from API' }
    }

    // Handle new API structure with anchor paper benchmarking
    if (apiData.anchor_paper && apiData.our_analyses) {
      return {
        anchor_paper: apiData.anchor_paper,
        our_analyses: apiData.our_analyses,
        benchmarks_vs_anchor: apiData.benchmarks_vs_anchor || [],
        aggregate_comparison: apiData.aggregate_comparison,
        validation_against_anchor: apiData.validation_against_anchor,
        benchmarks: apiData.benchmarks || [],
        domain_lexicon: apiData.domain_lexicon,
        recent_analyses: apiData.our_analyses || []
      }
    }

    // Fallback for old API structure
    if (apiData.summary) {
      return {
        summary: apiData.summary,
        benchmarks: apiData.benchmarks || [],
        domain_lexicon: apiData.domain_lexicon,
        recent_analyses: apiData.recent_analyses || []
      }
    }

    return { error: 'Invalid sentiment data structure received from API' }
  }

  const getSentimentColor = (score) => {
    if (score >= 0.5) return 'text-green-600 bg-green-50'
    if (score >= 0.1) return 'text-blue-600 bg-blue-50'
    if (score >= -0.1) return 'text-gray-600 bg-gray-50'
    if (score >= -0.5) return 'text-orange-600 bg-orange-50'
    return 'text-red-600 bg-red-50'
  }

  const getSentimentLabel = (score) => {
    if (score >= 0.5) return 'Very Positive'
    if (score >= 0.1) return 'Positive'
    if (score >= -0.1) return 'Neutral'
    if (score >= -0.5) return 'Negative'
    return 'Very Negative'
  }

  const renderSentimentContent = (data) => {
    // Handle new anchor paper benchmarking structure
    if (data.anchor_paper && data.our_analyses) {
      return (
        <>
          {/* Performance Verdict - Main Result */}
          {data.performance_verdict && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="w-5 h-5 text-green-500 mr-2" />
                System Performance vs Research Paper
              </h4>
              
              <div className={`border-2 rounded-lg p-6 ${
                data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                  ? 'border-green-500 bg-green-50'
                  : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                  ? 'border-red-500 bg-red-50'
                  : 'border-yellow-500 bg-yellow-50'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' && (
                      <CheckCircle className="w-8 h-8 text-green-600 mr-3" />
                    )}
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research' && (
                      <AlertTriangle className="w-8 h-8 text-red-600 mr-3" />
                    )}
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'comparable_to_research' && (
                      <Trophy className="w-8 h-8 text-yellow-600 mr-3" />
                    )}
                    <div>
                      <h5 className={`text-xl font-bold ${
                        data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                          ? 'text-green-800'
                          : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                          ? 'text-red-800'
                          : 'text-yellow-800'
                      }`}>
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' && 'Our System Outperforms Research Paper'}
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research' && 'Our System Underperforms Research Paper'}
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'comparable_to_research' && 'Our System Comparable to Research Paper'}
                      </h5>
                      <p className={`text-sm ${
                        data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                          ? 'text-green-700'
                          : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                          ? 'text-red-700'
                          : 'text-yellow-700'
                      }`}>
                        Research Alignment: {data.performance_verdict.overall_system_assessment.research_alignment_status.toUpperCase()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${
                      data.performance_verdict.sentiment_performance.performance_status === 'better' 
                        ? 'text-green-600'
                        : data.performance_verdict.sentiment_performance.performance_status === 'worse'
                        ? 'text-red-600'
                        : 'text-yellow-600'
                    }`}>
                      {data.performance_verdict.sentiment_performance.improvement_percentage >= 0 ? '+' : ''}
                      {data.performance_verdict.sentiment_performance.improvement_percentage.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-600">vs Research Paper</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-gray-900">
                      {(data.performance_verdict.sentiment_performance.our_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Our Avg Sentiment</div>
                  </div>
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-blue-600">
                      {(data.performance_verdict.sentiment_performance.anchor_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Research Paper</div>
                  </div>
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-purple-600">
                      {((data.performance_verdict.domain_relevance_performance.our_avg_domain_terms || 0) / (data.performance_verdict.domain_relevance_performance.anchor_domain_terms || 1) * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600">Domain Term Ratio</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Anchor Paper Section */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Trophy className="w-5 h-5 text-yellow-500 mr-2" />
              Research Paper Baseline (Anchor)
            </h4>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-700">
                    {(data.anchor_paper.sentiment_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-yellow-600">Sentiment Score</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-yellow-700">
                    {data.anchor_paper.classification}
                  </div>
                  <div className="text-sm text-yellow-600">Classification</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-yellow-700">
                    {data.anchor_paper.domain_terms_count}
                  </div>
                  <div className="text-sm text-yellow-600">Domain Terms</div>
                </div>
              </div>
              <p className="text-sm text-yellow-800 italic">
                "{data.anchor_paper.text.substring(0, 200)}..."
              </p>
            </div>
          </div>

          {/* Our Analyses vs Anchor Benchmarks */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 text-blue-500 mr-2" />
              Individual Analysis Benchmarks vs Research Paper
            </h4>
            
            {data.our_analyses.length > 0 ? (
              <div className="space-y-4">
                {data.our_analyses.slice(0, 10).map((analysis, index) => (
                  <div key={analysis.id || index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex-1">
                        <h5 className="font-medium text-gray-900">
                          {analysis.disease_name} (ID: {analysis.id})
                        </h5>
                        <p className="text-sm text-gray-600 mt-1">
                          "{analysis.text.substring(0, 150)}..."
                        </p>
                      </div>
                      <div className="ml-4 text-right">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          analysis.anchor_benchmark?.overall_similarity >= 0.8 
                            ? 'bg-green-100 text-green-800'
                            : analysis.anchor_benchmark?.overall_similarity >= 0.6
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {analysis.anchor_benchmark?.performance_assessment || 'Unknown'}
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-gray-900">
                          {(analysis.sentiment_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Our Score</div>
                      </div>
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-blue-600">
                          {((analysis.anchor_benchmark?.sentiment_similarity || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Sentiment Match</div>
                      </div>
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-purple-600">
                          {((analysis.anchor_benchmark?.domain_term_similarity || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Domain Match</div>
                      </div>
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-green-600">
                          {((analysis.anchor_benchmark?.overall_similarity || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Overall Match</div>
                      </div>
                    </div>
                    
                    {analysis.anchor_benchmark?.classification_match && (
                      <div className="mt-2 flex items-center text-sm text-green-600">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Classification matches anchor paper ({analysis.anchor_benchmark.analysis_sentiment_range})
                      </div>
                    )}
                    {!analysis.anchor_benchmark?.classification_match && (
                      <div className="mt-2 flex items-center text-sm text-orange-600">
                        <AlertTriangle className="w-4 h-4 mr-1" />
                        Different classification: {analysis.anchor_benchmark?.analysis_sentiment_range} vs {analysis.anchor_benchmark?.anchor_sentiment_range}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                No individual analyses available for benchmarking
              </div>
            )}
          </div>

          {/* Aggregate Statistics */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Target className="w-5 h-5 text-green-500 mr-2" />
              Aggregate Performance vs Research Standards
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-green-600">Avg Sentiment Similarity</p>
                    <p className="text-2xl font-bold text-green-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.avg_sentiment_similarity || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <Heart className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-purple-600">Domain Term Similarity</p>
                    <p className="text-2xl font-bold text-purple-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.avg_domain_similarity || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <Brain className="w-8 h-8 text-purple-500" />
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-blue-600">Classification Match Rate</p>
                    <p className="text-2xl font-bold text-blue-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.classification_match_rate || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-blue-500" />
                </div>
              </div>
            </div>
          </div>

          {/* Domain Lexicon Statistics */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <MessageCircle className="w-5 h-5 text-purple-500 mr-2" />
              Agricultural Domain Lexicon
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-700">{data.domain_lexicon?.positive_terms_count || 0}</div>
                <div className="text-sm text-green-600">Positive Terms</div>
              </div>
              <div className="text-center p-4 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-700">{data.domain_lexicon?.negative_terms_count || 0}</div>
                <div className="text-sm text-red-600">Negative Terms</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-700">{data.domain_lexicon?.neutral_terms_count || 0}</div>
                <div className="text-sm text-gray-600">Neutral Terms</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-700">
                  {(data.domain_lexicon?.positive_terms_count || 0) + (data.domain_lexicon?.negative_terms_count || 0) + (data.domain_lexicon?.neutral_terms_count || 0)}
                </div>
                <div className="text-sm text-purple-600">Total Enhancements</div>
              </div>
            </div>
          </div>
        </>
      )
    }

    // Fallback to old structure
    return renderLegacySentimentContent(data)
  }

  const renderLegacySentimentContent = (data) => (
    <>
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600">Avg Sentiment</p>
              <p className="text-2xl font-bold text-green-700">
                {(data.summary.avg_sentiment_score * 100).toFixed(1)}%
              </p>
            </div>
            <Heart className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600">Positive Ratio</p>
              <p className="text-2xl font-bold text-blue-700">
                {(data.summary.positive_ratio * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600">Domain Boost</p>
              <p className="text-2xl font-bold text-purple-700">
                +{(data.summary.domain_enhancement_boost * 100).toFixed(1)}%
              </p>
            </div>
            <Brain className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Analyses</p>
              <p className="text-2xl font-bold text-gray-700">{data.summary.total_analyses}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-gray-500" />
          </div>
        </div>
      </div>

      {/* Benchmark Comparison */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-gray-900 flex items-center">
            <Trophy className="w-5 h-5 text-yellow-500 mr-2" />
            Performance Benchmarks
          </h4>
          <button
            onClick={() => setShowMethodology(!showMethodology)}
            className="flex items-center text-sm text-primary-600 hover:text-primary-700"
          >
            <Info className="w-4 h-4 mr-1" />
            {showMethodology ? 'Hide' : 'Show'} Methodology
          </button>
        </div>

        {showMethodology && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <h5 className="font-medium text-blue-900 mb-2">Sentiment Analysis Methodology</h5>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• <strong>Base Algorithm:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)</li>
              <li>• <strong>Domain Enhancement:</strong> 145 agriculture-specific sentiment terms</li>
              <li>• <strong>Lexicon Composition:</strong> 56 positive, 59 negative, 30 neutral agricultural terms</li>
              <li>• <strong>Confidence Calculation:</strong> Compound score + domain relevance + context analysis</li>
              <li>• <strong>Classification Thresholds:</strong> Very Positive (≥0.5), Positive (≥0.1), Neutral (-0.1 to 0.1), Negative (≤-0.1)</li>
            </ul>
          </div>
        )}

        <div className="space-y-4">
          {data.benchmarks.map((benchmark, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h5 className="font-medium text-gray-900">{benchmark.metric}</h5>
                  <p className="text-sm text-gray-600">{benchmark.description}</p>
                </div>
                <div className="text-right">
                  <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    benchmark.difference > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {benchmark.difference > 0 ? '+' : ''}{(benchmark.difference * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Our Score</span>
                    <span className="font-medium">{(benchmark.our_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${benchmark.our_score * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Baseline</span>
                    <span className="font-medium">{(benchmark.baseline_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gray-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${benchmark.baseline_score * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Domain Lexicon Statistics */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MessageCircle className="w-5 h-5 text-purple-500 mr-2" />
          Agricultural Domain Lexicon
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-700">{data.domain_lexicon.positive_terms}</div>
            <div className="text-sm text-green-600">Positive Terms</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-700">{data.domain_lexicon.negative_terms}</div>
            <div className="text-sm text-red-600">Negative Terms</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-700">{data.domain_lexicon.neutral_terms}</div>
            <div className="text-sm text-gray-600">Neutral Terms</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-700">{data.domain_lexicon.total_enhancements}</div>
            <div className="text-sm text-purple-600">Total Enhancements</div>
          </div>
        </div>
      </div>

      {/* Recent Analysis Examples */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Zap className="w-5 h-5 text-yellow-500 mr-2" />
          Recent Sentiment Analyses
        </h4>
        <div className="space-y-3">
          {data.recent_analyses.map((analysis, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-start mb-2">
                <p className="text-gray-900 flex-1 mr-4">"{analysis.text}"</p>
                <div className="flex flex-col items-end space-y-1">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(analysis.sentiment)}`}>
                    {getSentimentLabel(analysis.sentiment)}
                  </span>
                  <span className="text-xs text-gray-500">
                    Score: {analysis.sentiment.toFixed(3)}
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <div className="w-3/4 bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      analysis.sentiment >= 0 ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{ 
                      width: `${Math.abs(analysis.sentiment) * 100}%`,
                      marginLeft: analysis.sentiment < 0 ? `${(1 - Math.abs(analysis.sentiment)) * 100}%` : '0'
                    }}
                  ></div>
                </div>
                <span className="text-xs text-purple-600 font-medium">
                  Domain: {analysis.domain_boost > 0 ? '+' : ''}{(analysis.domain_boost * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {showCalculationDetails && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h5 className="font-medium text-blue-900 mb-2 flex items-center">
            <Calculator className="w-4 h-4 mr-2" />
            Calculation Details
          </h5>
          <div className="text-sm text-blue-800 space-y-2">
            <p><strong>Sentiment Score Formula:</strong> Compound Score + Domain Adjustment + Context Weight</p>
            <p><strong>Domain Adjustment:</strong> Agricultural terms receive ±0.1 to ±0.3 sentiment boost based on context</p>
            <p><strong>Confidence Level:</strong> Based on lexicon coverage, domain relevance, and text length</p>
            <p><strong>Classification:</strong> Threshold-based classification with agricultural context awareness</p>
          </div>
        </div>
      )}
    </>
  )

  if (loading) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <span className="ml-3 text-gray-600">Analyzing sentiment patterns...</span>
        </div>
      </div>
    )
  }

  if (sentimentData?.error) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-6">
        <div className="flex items-center mb-4">
          <AlertTriangle className="w-5 h-5 text-yellow-500 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Sentiment Analytics</h3>
        </div>
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-6">
          <p className="text-yellow-800">{sentimentData.error}</p>
          {sentimentData.fallback && (
            <p className="text-yellow-700 mt-2 text-sm">Showing demo data for interface preview.</p>
          )}
        </div>
        {sentimentData.mockData && renderSentimentContent(sentimentData.mockData)}
      </div>
    )
  }

  const renderSentimentContent = (data) => {
    // Handle new anchor paper benchmarking structure
    if (data.anchor_paper && data.our_analyses) {
      return (
        <>
          {/* Performance Verdict - Main Result */}
          {data.performance_verdict && (
            <div className="mb-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <Target className="w-5 h-5 text-green-500 mr-2" />
                System Performance vs Research Paper
              </h4>
              
              <div className={`border-2 rounded-lg p-6 ${
                data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                  ? 'border-green-500 bg-green-50'
                  : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                  ? 'border-red-500 bg-red-50'
                  : 'border-yellow-500 bg-yellow-50'
              }`}>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center">
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' && (
                      <CheckCircle className="w-8 h-8 text-green-600 mr-3" />
                    )}
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research' && (
                      <AlertTriangle className="w-8 h-8 text-red-600 mr-3" />
                    )}
                    {data.performance_verdict.overall_system_assessment.system_verdict === 'comparable_to_research' && (
                      <Trophy className="w-8 h-8 text-yellow-600 mr-3" />
                    )}
                    <div>
                      <h5 className={`text-xl font-bold ${
                        data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                          ? 'text-green-800'
                          : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                          ? 'text-red-800'
                          : 'text-yellow-800'
                      }`}>
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' && 'Our System Outperforms Research Paper'}
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research' && 'Our System Underperforms Research Paper'}
                        {data.performance_verdict.overall_system_assessment.system_verdict === 'comparable_to_research' && 'Our System Comparable to Research Paper'}
                      </h5>
                      <p className={`text-sm ${
                        data.performance_verdict.overall_system_assessment.system_verdict === 'outperforms_research' 
                          ? 'text-green-700'
                          : data.performance_verdict.overall_system_assessment.system_verdict === 'underperforms_research'
                          ? 'text-red-700'
                          : 'text-yellow-700'
                      }`}>
                        Research Alignment: {data.performance_verdict.overall_system_assessment.research_alignment_status.toUpperCase()}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`text-2xl font-bold ${
                      data.performance_verdict.sentiment_performance.performance_status === 'better' 
                        ? 'text-green-600'
                        : data.performance_verdict.sentiment_performance.performance_status === 'worse'
                        ? 'text-red-600'
                        : 'text-yellow-600'
                    }`}>
                      {data.performance_verdict.sentiment_performance.improvement_percentage >= 0 ? '+' : ''}
                      {data.performance_verdict.sentiment_performance.improvement_percentage.toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-600">vs Research Paper</div>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-gray-900">
                      {(data.performance_verdict.sentiment_performance.our_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Our Avg Sentiment</div>
                  </div>
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-blue-600">
                      {(data.performance_verdict.sentiment_performance.anchor_score * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-600">Research Paper</div>
                  </div>
                  <div className="text-center p-3 bg-white rounded-lg">
                    <div className="text-lg font-bold text-purple-600">
                      {((data.performance_verdict.domain_relevance_performance.our_avg_domain_terms || 0) / (data.performance_verdict.domain_relevance_performance.anchor_domain_terms || 1) * 100).toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600">Domain Term Ratio</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Anchor Paper Section */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Trophy className="w-5 h-5 text-yellow-500 mr-2" />
              Research Paper Baseline (Anchor)
            </h4>
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-700">
                    {(data.anchor_paper.sentiment_score * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-yellow-600">Sentiment Score</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-yellow-700">
                    {data.anchor_paper.classification}
                  </div>
                  <div className="text-sm text-yellow-600">Classification</div>
                </div>
                <div className="text-center">
                  <div className="text-lg font-bold text-yellow-700">
                    {data.anchor_paper.domain_terms_count}
                  </div>
                  <div className="text-sm text-yellow-600">Domain Terms</div>
                </div>
              </div>
              <p className="text-sm text-yellow-800 italic">
                "{data.anchor_paper.text.substring(0, 200)}..."
              </p>
            </div>
          </div>

          {/* Our Analyses vs Anchor Benchmarks */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <BarChart3 className="w-5 h-5 text-blue-500 mr-2" />
              Individual Analysis Benchmarks vs Research Paper
            </h4>
            
            {data.our_analyses.length > 0 ? (
              <div className="space-y-4">
                {data.our_analyses.slice(0, 10).map((analysis, index) => (
                  <div key={analysis.id || index} className="bg-gray-50 rounded-lg p-4">
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex-1">
                        <h5 className="font-medium text-gray-900">
                          {analysis.disease_name} (ID: {analysis.id})
                        </h5>
                        <p className="text-sm text-gray-600 mt-1">
                          "{analysis.text.substring(0, 150)}..."
                        </p>
                      </div>
                      <div className="ml-4 text-right">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          analysis.anchor_benchmark?.overall_similarity >= 0.8 
                            ? 'bg-green-100 text-green-800'
                            : analysis.anchor_benchmark?.overall_similarity >= 0.6
                            ? 'bg-yellow-100 text-yellow-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {analysis.anchor_benchmark?.performance_assessment || 'Unknown'}
                        </span>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-gray-900">
                          {(analysis.sentiment_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Our Score</div>
                        <div className="text-xs text-blue-500">
                          {analysis.anchor_benchmark?.analysis_sentiment_range || 'Unknown'}
                        </div>
                      </div>
                      <div className="text-center p-2 bg-yellow-50 rounded border">
                        <div className="text-lg font-bold text-yellow-700">
                          {((analysis.anchor_benchmark?.anchor_sentiment_score || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Research Score</div>
                        <div className="text-xs text-yellow-600">
                          {analysis.anchor_benchmark?.anchor_sentiment_range || 'Unknown'}
                        </div>
                      </div>
                      <div className="text-center p-2 bg-white rounded">
                        <div className={`text-lg font-bold ${
                          (analysis.sentiment_score || 0) > (analysis.anchor_benchmark?.anchor_sentiment_score || 0) 
                            ? 'text-green-600' 
                            : (analysis.sentiment_score || 0) < (analysis.anchor_benchmark?.anchor_sentiment_score || 0)
                            ? 'text-red-600'
                            : 'text-gray-600'
                        }`}>
                          {((analysis.sentiment_score || 0) > (analysis.anchor_benchmark?.anchor_sentiment_score || 0)) && '↗ Better'}
                          {((analysis.sentiment_score || 0) < (analysis.anchor_benchmark?.anchor_sentiment_score || 0)) && '↘ Lower'}
                          {((analysis.sentiment_score || 0) === (analysis.anchor_benchmark?.anchor_sentiment_score || 0)) && '= Equal'}
                        </div>
                        <div className="text-xs text-gray-600">vs Research</div>
                        <div className="text-xs font-medium">
                          Δ {(((analysis.sentiment_score || 0) - (analysis.anchor_benchmark?.anchor_sentiment_score || 0)) * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="text-center p-2 bg-white rounded">
                        <div className="text-lg font-bold text-green-600">
                          {((analysis.anchor_benchmark?.overall_similarity || 0) * 100).toFixed(1)}%
                        </div>
                        <div className="text-xs text-gray-600">Overall Match</div>
                        <div className="text-xs text-purple-500">
                          Sent: {((analysis.anchor_benchmark?.sentiment_similarity || 0) * 100).toFixed(0)}%
                        </div>
                      </div>
                    </div>
                    
                    {/* Domain Terms Comparison */}
                    <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                      <div className="bg-blue-50 p-2 rounded">
                        <span className="font-medium text-blue-700">Our Domain Terms:</span>
                        <span className="ml-1 text-blue-600">
                          {analysis.anchor_benchmark?.analysis_domain_terms || 0} terms
                        </span>
                      </div>
                      <div className="bg-yellow-50 p-2 rounded">
                        <span className="font-medium text-yellow-700">Research Domain Terms:</span>
                        <span className="ml-1 text-yellow-600">
                          {analysis.anchor_benchmark?.anchor_domain_terms || 0} terms
                        </span>
                      </div>
                    </div>
                    
                    {analysis.anchor_benchmark?.classification_match && (
                      <div className="mt-2 flex items-center text-sm text-green-600">
                        <CheckCircle className="w-4 h-4 mr-1" />
                        Classification matches anchor paper ({analysis.anchor_benchmark.analysis_sentiment_range})
                      </div>
                    )}
                    {!analysis.anchor_benchmark?.classification_match && (
                      <div className="mt-2 flex items-center text-sm text-orange-600">
                        <AlertTriangle className="w-4 h-4 mr-1" />
                        Different classification: {analysis.anchor_benchmark?.analysis_sentiment_range} vs {analysis.anchor_benchmark?.anchor_sentiment_range}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                No individual analyses available for benchmarking
              </div>
            )}
          </div>

          {/* Aggregate Statistics */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <Target className="w-5 h-5 text-green-500 mr-2" />
              Aggregate Performance vs Research Standards
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-green-600">Avg Sentiment Similarity</p>
                    <p className="text-2xl font-bold text-green-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.avg_sentiment_similarity || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <Heart className="w-8 h-8 text-green-500" />
                </div>
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-purple-600">Domain Term Similarity</p>
                    <p className="text-2xl font-bold text-purple-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.avg_domain_similarity || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <Brain className="w-8 h-8 text-purple-500" />
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-blue-600">Classification Match Rate</p>
                    <p className="text-2xl font-bold text-blue-700">
                      {((data.aggregate_comparison?.anchor_similarity_stats?.classification_match_rate || 0) * 100).toFixed(1)}%
                    </p>
                  </div>
                  <CheckCircle className="w-8 h-8 text-blue-500" />
                </div>
              </div>
            </div>
          </div>

          {/* Domain Lexicon Statistics */}
          <div className="mb-6">
            <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
              <MessageCircle className="w-5 h-5 text-purple-500 mr-2" />
              Agricultural Domain Lexicon
            </h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-700">{data.domain_lexicon?.positive_terms_count || 0}</div>
                <div className="text-sm text-green-600">Positive Terms</div>
              </div>
              <div className="text-center p-4 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-700">{data.domain_lexicon?.negative_terms_count || 0}</div>
                <div className="text-sm text-red-600">Negative Terms</div>
              </div>
              <div className="text-center p-4 bg-gray-50 rounded-lg">
                <div className="text-2xl font-bold text-gray-700">{data.domain_lexicon?.neutral_terms_count || 0}</div>
                <div className="text-sm text-gray-600">Neutral Terms</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded-lg">
                <div className="text-2xl font-bold text-purple-700">
                  {(data.domain_lexicon?.positive_terms_count || 0) + (data.domain_lexicon?.negative_terms_count || 0) + (data.domain_lexicon?.neutral_terms_count || 0)}
                </div>
                <div className="text-sm text-purple-600">Total Enhancements</div>
              </div>
            </div>
          </div>
        </>
      )
    }

    // Fallback to old structure
    return renderLegacySentimentContent(data)
  }

  const renderLegacySentimentContent = (data) => (
    <>
      {/* Summary Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-r from-green-50 to-green-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-green-600">Avg Sentiment</p>
              <p className="text-2xl font-bold text-green-700">
                {(data.summary.avg_sentiment_score * 100).toFixed(1)}%
              </p>
            </div>
            <Heart className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-blue-50 to-blue-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-blue-600">Positive Ratio</p>
              <p className="text-2xl font-bold text-blue-700">
                {(data.summary.positive_ratio * 100).toFixed(1)}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-purple-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-purple-600">Domain Boost</p>
              <p className="text-2xl font-bold text-purple-700">
                +{(data.summary.domain_enhancement_boost * 100).toFixed(1)}%
              </p>
            </div>
            <Brain className="w-8 h-8 text-purple-500" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Analyses</p>
              <p className="text-2xl font-bold text-gray-700">{data.summary.total_analyses}</p>
            </div>
            <BarChart3 className="w-8 h-8 text-gray-500" />
          </div>
        </div>
      </div>

      {/* Benchmark Comparison */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-semibold text-gray-900 flex items-center">
            <Trophy className="w-5 h-5 text-yellow-500 mr-2" />
            Performance Benchmarks
          </h4>
          <button
            onClick={() => setShowMethodology(!showMethodology)}
            className="flex items-center text-sm text-primary-600 hover:text-primary-700"
          >
            <Info className="w-4 h-4 mr-1" />
            {showMethodology ? 'Hide' : 'Show'} Methodology
          </button>
        </div>

        {showMethodology && (
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
            <h5 className="font-medium text-blue-900 mb-2">Sentiment Analysis Methodology</h5>
            <ul className="text-sm text-blue-800 space-y-1">
              <li>• <strong>Base Algorithm:</strong> VADER (Valence Aware Dictionary and sEntiment Reasoner)</li>
              <li>• <strong>Domain Enhancement:</strong> 145 agriculture-specific sentiment terms</li>
              <li>• <strong>Lexicon Composition:</strong> 56 positive, 59 negative, 30 neutral agricultural terms</li>
              <li>• <strong>Confidence Calculation:</strong> Compound score + domain relevance + context analysis</li>
              <li>• <strong>Classification Thresholds:</strong> Very Positive (≥0.5), Positive (≥0.1), Neutral (-0.1 to 0.1), Negative (≤-0.1)</li>
            </ul>
          </div>
        )}

        <div className="space-y-4">
          {data.benchmarks.map((benchmark, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-start mb-2">
                <div>
                  <h5 className="font-medium text-gray-900">{benchmark.metric}</h5>
                  <p className="text-sm text-gray-600">{benchmark.description}</p>
                </div>
                <div className="text-right">
                  <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                    benchmark.difference > 0 ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  }`}>
                    {benchmark.difference > 0 ? '+' : ''}{(benchmark.difference * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Our Score</span>
                    <span className="font-medium">{(benchmark.our_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${benchmark.our_score * 100}%` }}
                    ></div>
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-600">Baseline</span>
                    <span className="font-medium">{(benchmark.baseline_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gray-400 h-2 rounded-full transition-all duration-500"
                      style={{ width: `${benchmark.baseline_score * 100}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Domain Lexicon Statistics */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <MessageCircle className="w-5 h-5 text-purple-500 mr-2" />
          Agricultural Domain Lexicon
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center p-4 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-700">{data.domain_lexicon.positive_terms}</div>
            <div className="text-sm text-green-600">Positive Terms</div>
          </div>
          <div className="text-center p-4 bg-red-50 rounded-lg">
            <div className="text-2xl font-bold text-red-700">{data.domain_lexicon.negative_terms}</div>
            <div className="text-sm text-red-600">Negative Terms</div>
          </div>
          <div className="text-center p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-700">{data.domain_lexicon.neutral_terms}</div>
            <div className="text-sm text-gray-600">Neutral Terms</div>
          </div>
          <div className="text-center p-4 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-700">{data.domain_lexicon.total_enhancements}</div>
            <div className="text-sm text-purple-600">Total Enhancements</div>
          </div>
        </div>
      </div>

      {/* Recent Analysis Examples */}
      <div className="mb-6">
        <h4 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Zap className="w-5 h-5 text-yellow-500 mr-2" />
          Recent Sentiment Analyses
        </h4>
        <div className="space-y-3">
          {data.recent_analyses.map((analysis, index) => (
            <div key={index} className="bg-gray-50 rounded-lg p-4">
              <div className="flex justify-between items-start mb-2">
                <p className="text-gray-900 flex-1 mr-4">"{analysis.text}"</p>
                <div className="flex flex-col items-end space-y-1">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getSentimentColor(analysis.sentiment)}`}>
                    {getSentimentLabel(analysis.sentiment)}
                  </span>
                  <span className="text-xs text-gray-500">
                    Score: {analysis.sentiment.toFixed(3)}
                  </span>
                </div>
              </div>
              <div className="flex justify-between items-center">
                <div className="w-3/4 bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full transition-all duration-500 ${
                      analysis.sentiment >= 0 ? 'bg-green-500' : 'bg-red-500'
                    }`}
                    style={{ 
                      width: `${Math.abs(analysis.sentiment) * 100}%`,
                      marginLeft: analysis.sentiment < 0 ? `${(1 - Math.abs(analysis.sentiment)) * 100}%` : '0'
                    }}
                  ></div>
                </div>
                <span className="text-xs text-purple-600 font-medium">
                  Domain: {analysis.domain_boost > 0 ? '+' : ''}{(analysis.domain_boost * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {showCalculationDetails && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h5 className="font-medium text-blue-900 mb-2 flex items-center">
            <Calculator className="w-4 h-4 mr-2" />
            Calculation Details
          </h5>
          <div className="text-sm text-blue-800 space-y-2">
            <p><strong>Sentiment Score Formula:</strong> Compound Score + Domain Adjustment + Context Weight</p>
            <p><strong>Domain Adjustment:</strong> Agricultural terms receive ±0.1 to ±0.3 sentiment boost based on context</p>
            <p><strong>Confidence Level:</strong> Based on lexicon coverage, domain relevance, and text length</p>
            <p><strong>Classification:</strong> Threshold-based classification with agricultural context awareness</p>
          </div>
        </div>
      )}
    </>
  )

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-6">
      <div className="flex items-center mb-6">
        <Heart className="w-6 h-6 text-primary-600 mr-3" />
        <h3 className="text-xl font-bold text-gray-900">Sentiment Analysis Benchmarks</h3>
        <div className="ml-auto flex items-center space-x-2">
          <span className="text-sm text-gray-500">VADER + Agricultural Domain</span>
          <Star className="w-4 h-4 text-yellow-500" />
        </div>
      </div>
      
      {sentimentData && renderSentimentContent(sentimentData.fallback ? sentimentData.mockData : sentimentData)}
    </div>
  )
}

export default SentimentComparison
